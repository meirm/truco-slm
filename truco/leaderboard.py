"""
Persistent Leaderboard & Round-Robin Tournament System for Argentine Truco Agents.

Tracks all players (System 1 SLM checkpoints across generations, Frozen Base Model,
Untrained System 1 SLM, Heuristic Bluff Agent, Random Agent, and Human players) with:
- Rank (#1, #2, ...)
- Player Name & Model Type / Architecture / Checkpoint
- Elo Rating (anchored so `JEV-Base (Gen 0)` = 1500)
- Total Matches Played, Wins, Losses, Win %
- Hands Played, Hand Wins, Hand Losses, Net Pts/Hand
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from truco.engine import Action, DECK, TrucoHandState
from truco.model import System1TrucoSLM
from truco.tokenizer import TrucoTokenizer
from truco.trainer import HeuristicAnchorAgent

LEADERBOARD_JSON_PATH = "checkpoints/leaderboard.json"


@dataclass
class PlayerStats:
    player_id: str
    display_name: str
    model_type: str
    checkpoint_path: str = "-"
    elo: float = 1500.0
    matches_played: int = 0
    matches_won: int = 0
    matches_lost: int = 0
    hands_played: int = 0
    hands_won: int = 0
    hands_lost: int = 0
    hands_tied: int = 0
    total_points_for: int = 0
    total_points_against: int = 0

    @property
    def match_win_rate(self) -> float:
        return (self.matches_won / self.matches_played * 100.0) if self.matches_played > 0 else 0.0

    @property
    def hand_win_rate(self) -> float:
        return (self.hands_won / self.hands_played * 100.0) if self.hands_played > 0 else 0.0

    @property
    def net_pts_per_hand(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return (self.total_points_for - self.total_points_against) / self.hands_played


class TrucoAgentWrapper:
    """Unified wrapper around any Truco player (System1TrucoSLM, Heuristic, Random)."""

    def __init__(
        self,
        stats: PlayerStats,
        act_fn: Callable[[TrucoHandState], Action],
    ) -> None:
        self.stats = stats
        self.act_fn = act_fn

    def act(self, state: TrucoHandState) -> Action:
        return self.act_fn(state)


def update_elo(elo_a: float, elo_b: float, score_a: float, k: float = 24.0) -> Tuple[float, float]:
    """Standard Elo update where `score_a` is 1.0 for A win, 0.0 for B win, 0.5 for draw."""
    exp_a = 1.0 / (1.0 + math.pow(10.0, (elo_b - elo_a) / 400.0))
    exp_b = 1.0 - exp_a
    new_a = elo_a + k * (score_a - exp_a)
    new_b = elo_b + k * ((1.0 - score_a) - exp_b)
    return new_a, new_b


class LeaderboardManager:
    def __init__(self, json_path: str = LEADERBOARD_JSON_PATH) -> None:
        self.json_path = json_path
        self.players: Dict[str, PlayerStats] = {}
        self.head_to_head: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.disabled_players: List[str] = []
        self.load()

    def load(self) -> None:
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.disabled_players = list(data.get("disabled_players", []))
                disabled_set = set(self.disabled_players)
                for pid, pdict in data.get("players", {}).items():
                    if pid not in disabled_set:
                        self.players[pid] = PlayerStats(**pdict)
                raw_h2h = data.get("head_to_head", {})
                self.head_to_head = {
                    pa: {pb: rec for pb, rec in row.items() if pb not in disabled_set}
                    for pa, row in raw_h2h.items()
                    if pa not in disabled_set
                }
            except Exception:
                self.players = {}
                self.head_to_head = {}
                self.disabled_players = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        data = {
            "players": {pid: asdict(p) for pid, p in self.players.items()},
            "head_to_head": self.head_to_head,
            "disabled_players": sorted(set(self.disabled_players)),
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _disable_pids(self, pids_to_remove: List[str]) -> List[str]:
        removed = []
        for pid in pids_to_remove:
            if pid not in self.disabled_players:
                self.disabled_players.append(pid)
            if pid in self.players:
                del self.players[pid]
                removed.append(pid)
            self.head_to_head.pop(pid, None)
        for row in self.head_to_head.values():
            for pid in pids_to_remove:
                row.pop(pid, None)
        self.save()
        return removed

    def remove_player(self, player_id: str) -> List[str]:
        """Removes a single model/variant from the active Leaderboard and marks it disabled."""
        return self._disable_pids([player_id])

    def prune_to_top_k(self, keep_k: int = 8) -> List[str]:
        """Keeps the top `keep_k` models ranked by Elo and removes the losing tail."""
        keep_k = max(2, int(keep_k))
        ranked = self.get_ranked_players()
        to_drop = [p.player_id for p in ranked[keep_k:]]
        return self._disable_pids(to_drop)

    def prune_to_core_8(self) -> List[str]:
        """
        Keeps the 8 distinct tactical & architectural models (4 Core + 4 Personality Specialists):
          - jev_base, jev_gen_16, jev_champion, heuristic_gaucho
          - jev_sheriff, jev_bluffer, jev_phisher, jev_kamikaze
        and removes the redundant intermediate generations + untrained/random tail.
        """
        core_keep = {
            "jev_base",
            "jev_gen_16",
            "jev_champion",
            "heuristic_gaucho",
            "jev_sheriff",
            "jev_bluffer",
            "jev_phisher",
            "jev_kamikaze",
        }
        # First re-enable any of the 8 core models if they were previously disabled
        self.disabled_players = [pid for pid in self.disabled_players if pid not in core_keep]
        to_drop = [pid for pid in list(self.players.keys()) if pid not in core_keep]
        return self._disable_pids(to_drop)

    def prune_below_win_rate(self, min_win_rate: float = 50.0) -> List[str]:
        """Removes all models whose match win rate is strictly below `min_win_rate`%."""
        ranked = self.get_ranked_players()
        # Always keep at least the top 2 models even if min_win_rate is set very high
        protected = {p.player_id for p in ranked[:2]}
        to_drop = [
            p.player_id
            for p in ranked
            if p.player_id not in protected and p.match_win_rate < min_win_rate
        ]
        return self._disable_pids(to_drop)

    def restore_all_players(self) -> None:
        """Clears the disabled list so all discovered checkpoints & bots reappear."""
        self.disabled_players = []
        self.save()

    def register_player(
        self,
        player_id: str,
        display_name: str,
        model_type: str,
        checkpoint_path: str = "-",
        initial_elo: float = 1500.0,
    ) -> PlayerStats:
        if player_id not in self.players:
            self.players[player_id] = PlayerStats(
                player_id=player_id,
                display_name=display_name,
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                elo=initial_elo,
            )
        else:
            self.players[player_id].display_name = display_name
            self.players[player_id].model_type = model_type
            self.players[player_id].checkpoint_path = checkpoint_path
        return self.players[player_id]

    def record_hand(self, pid_a: str, pid_b: str, pts_a: int, pts_b: int) -> None:
        pa = self.players[pid_a]
        pb = self.players[pid_b]
        pa.hands_played += 1
        pb.hands_played += 1
        pa.total_points_for += pts_a
        pa.total_points_against += pts_b
        pb.total_points_for += pts_b
        pb.total_points_against += pts_a

        if pts_a > pts_b:
            pa.hands_won += 1
            pb.hands_lost += 1
        elif pts_b > pts_a:
            pb.hands_won += 1
            pa.hands_lost += 1
        else:
            pa.hands_tied += 1
            pb.hands_tied += 1

    def record_match(self, pid_a: str, pid_b: str, winner_is_a: bool, k: float = 16.0) -> None:
        pa = self.players[pid_a]
        pb = self.players[pid_b]
        pa.matches_played += 1
        pb.matches_played += 1

        if winner_is_a:
            pa.matches_won += 1
            pb.matches_lost += 1
            score_a = 1.0
        else:
            pb.matches_won += 1
            pa.matches_lost += 1
            score_a = 0.0

        pa.elo, pb.elo = update_elo(pa.elo, pb.elo, score_a, k=k)

        # Update head-to-head matrix
        self.head_to_head.setdefault(pid_a, {}).setdefault(pid_b, {"w": 0, "l": 0})
        self.head_to_head.setdefault(pid_b, {}).setdefault(pid_a, {"w": 0, "l": 0})
        if winner_is_a:
            self.head_to_head[pid_a][pid_b]["w"] += 1
            self.head_to_head[pid_b][pid_a]["l"] += 1
        else:
            self.head_to_head[pid_a][pid_b]["l"] += 1
            self.head_to_head[pid_b][pid_a]["w"] += 1

    def get_ranked_players(self) -> List[PlayerStats]:
        return sorted(
            self.players.values(),
            key=lambda p: (p.elo, p.match_win_rate, p.net_pts_per_hand),
            reverse=True,
        )

    def format_table(self) -> str:
        ranked = self.get_ranked_players()
        lines = []
        sep = "=" * 118
        dash = "-" * 118
        lines.append(sep)
        lines.append(" 🏆 ARGENTINE TRUCO AI & PLAYER LEADERBOARD (15-Point Matches & Duplicate Hands)")
        lines.append(sep)
        header = (
            f"{'Rank':<5} | {'Player Name':<22} | {'Model / Architecture':<28} | "
            f"{'Elo':>6} | {'Matches':>7} | {'Wins':>5} | {'Losses':>6} | {'Win %':>6} | {'Net Pts/H':>9}"
        )
        lines.append(header)
        lines.append(dash)

        for idx, p in enumerate(ranked, start=1):
            rank_str = f"#{idx}"
            lines.append(
                f"{rank_str:<5} | {p.display_name:<22} | {p.model_type:<28} | "
                f"{p.elo:>6.0f} | {p.matches_played:>7} | {p.matches_won:>5} | "
                f"{p.matches_lost:>6} | {p.match_win_rate:>5.1f}% | {p.net_pts_per_hand:>+9.3f}"
            )
        lines.append(sep)

        # Render Head-to-Head (Who Played Whom) Matrix
        if self.head_to_head and len(ranked) > 1:
            lines.append("\n ⚔️  HEAD-TO-HEAD MATCHUP MATRIX (Row Wins - Row Losses vs Column Opponent)")
            col_headers = [f"#{i}" for i in range(1, len(ranked) + 1)]
            h2h_header = f"{'Rank & Player':<28} | " + " | ".join(f"{ch:^7}" for ch in col_headers)
            h2h_sep = "-" * len(h2h_header)
            lines.append(h2h_sep)
            lines.append(h2h_header)
            lines.append(h2h_sep)
            for r_idx, pa in enumerate(ranked, start=1):
                cells = []
                for c_idx, pb in enumerate(ranked, start=1):
                    if pa.player_id == pb.player_id:
                        cells.append(f"{'---':^7}")
                    else:
                        rec = self.head_to_head.get(pa.player_id, {}).get(pb.player_id, {"w": 0, "l": 0})
                        w, l = rec.get("w", 0), rec.get("l", 0)
                        if w == 0 and l == 0:
                            cells.append(f"{'0-0':^7}")
                        else:
                            cells.append(f"{f'{w}-{l}':^7}")
                row_label = f"#{r_idx:<2} {pa.display_name[:23]:<23}"
                lines.append(f"{row_label:<28} | " + " | ".join(cells))
            lines.append(h2h_sep)

        return "\n".join(lines)


@torch.no_grad()
def act_with_personality(
    model: System1TrucoSLM,
    state: TrucoHandState,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    persona: str = "standard",
    temperature: float = 0.05,
) -> Action:
    """
    Executes a System-1 SLM forward pass modulated by a tactical personality profile:
    - 'bluffer' ('El Mentiroso'): High-frequency strategic bluffer on weak hands; folds if re-raised with trash.
    - 'phisher' ('El Zorro'): Slow-plays monster hands on Trick 0 ('Achique' bait with weakest card) and snap-traps on Trick 1/2.
    - 'sheriff' ('El Comisario'): Smells bluffs (p_bluff_suspect) and Trick-1 slow-play traps (p_trap_suspect).
    - 'risk_taker' ('El Temerario'): Relentless stakes escalator (Truco / Retruco / Real Envido).
    """
    from truco.engine import NUM_ACTIONS
    from truco.tokenizer import compute_opponent_belief

    if persona == "standard":
        act, _, _ = model.act(state, tokenizer, device, temperature=temperature)
        return act

    p = state.current_player
    legal = state.legal_actions()
    if len(legal) == 1:
        return legal[0]

    toks = tokenizer.encode_observation(state, p)
    if len(toks) > tokenizer.max_seq_len:
        toks = [toks[0]] + toks[-(tokenizer.max_seq_len - 1):]

    mask = [False] * NUM_ACTIONS
    for a in legal:
        mask[int(a)] = True

    inp = torch.tensor([toks], dtype=torch.long, device=device)
    act_pos = torch.tensor([len(toks) - 1], dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.bool, device=device)
    logits, _ = model(inp, act_pos, mask_t)
    mod_logits = logits[0].clone()

    belief = compute_opponent_belief(state, p)
    p_win = float(belief["p_beat_opp_best"])
    p_env_win = float(belief["p_win_envido"])
    p_bluff_suspect = float(belief.get("p_bluff_suspect", 0.0))
    p_trap_suspect = float(belief.get("p_trap_suspect", 0.0))

    my_rem = [state.hands[p][s] for s in range(3) if not state.played_mask[p][s]]
    max_my_power = max((c.truco_power for c in my_rem), default=0)

    if persona == "bluffer":
        # El Mentiroso: Bluffs frequently when holding weak cards, but folds if re-raised with trash
        if p_win < 0.45:
            mod_logits[int(Action.CALL_TRUCO)] += 3.0
            mod_logits[int(Action.CALL_RETRUCO)] += 2.2
            if p_win < 0.34 and state.truco_level >= 1:
                mod_logits[int(Action.NO_QUIERO)] += 2.0
        if p_env_win < 0.40 and len(state.trick_winners) == 0:
            mod_logits[int(Action.CALL_ENVIDO)] += 2.6
            if p_env_win < 0.25 and len(state.envido_chain) >= 2:
                mod_logits[int(Action.NO_QUIERO)] += 2.0

    elif persona == "phisher":
        # El Zorro: Slow-plays ("Achique") Monster hands on Trick 0, then springs the trap!
        if max_my_power >= 12 or p_win >= 0.70:
            if len(state.trick_winners) == 0 and state.truco_level == 0 and int(state.pending) == 0:
                # Hide strength on Trick 0: suppress early Truco and throw weakest card (PLAY_CARD_2) as bait
                mod_logits[int(Action.CALL_TRUCO)] -= 3.5
                if Action.PLAY_CARD_2 in legal and len(my_rem) == 3:
                    mod_logits[int(Action.PLAY_CARD_2)] += 3.0
            else:
                # Opponent took the bait or we reached Trick 1/2: spring the trap!
                mod_logits[int(Action.CALL_TRUCO)] += 3.2
                mod_logits[int(Action.CALL_RETRUCO)] += 4.0
                mod_logits[int(Action.CALL_VALE_CUATRO)] += 4.0
                mod_logits[int(Action.QUIERO)] += 4.0

    elif persona == "sheriff":
        # El Comisario: Calls bluffs when p_bluff_suspect is high; respects slow-play traps when p_trap_suspect is high
        if p_bluff_suspect >= 0.35:
            mod_logits[int(Action.QUIERO)] += 3.2
            mod_logits[int(Action.CALL_RETRUCO)] += 1.8
            mod_logits[int(Action.NO_QUIERO)] -= 2.5
        if p_trap_suspect >= 0.35 and p_win < 0.78:
            mod_logits[int(Action.CALL_TRUCO)] -= 3.0
            mod_logits[int(Action.CALL_RETRUCO)] -= 3.0
            if int(state.pending) != 0 and p_win < 0.52:
                mod_logits[int(Action.NO_QUIERO)] += 2.2

    elif persona == "risk_taker":
        # El Temerario: Relentless stakes escalator
        for a_esc in (
            Action.CALL_TRUCO,
            Action.CALL_RETRUCO,
            Action.CALL_VALE_CUATRO,
            Action.CALL_REAL_ENVIDO,
            Action.CALL_FALTA_ENVIDO,
        ):
            mod_logits[int(a_esc)] += 2.3
        mod_logits[int(Action.QUIERO)] += 1.6
        mod_logits[int(Action.NO_QUIERO)] -= 2.0

    mod_logits = mod_logits.masked_fill(~mask_t[0], -1e9)
    if temperature <= 1e-5:
        idx = int(torch.argmax(mod_logits).item())
    else:
        probs = torch.softmax(mod_logits / temperature, dim=-1)
        idx = int(torch.multinomial(probs, num_samples=1).item())
    return Action(idx)


def build_roster(
    leaderboard: LeaderboardManager,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
) -> Dict[str, TrucoAgentWrapper]:
    """
    Discovers all available models (Saved System 1 SLM checkpoints, Base Model,
    Tactical Personality Bots, Untrained SLM, Heuristic Bluff Agent, Random Agent)
    and wraps them as tournament players.
    """
    roster: Dict[str, TrucoAgentWrapper] = {}
    disabled_set = set(leaderboard.disabled_players)

    def load_slm_agent(
        pid: str,
        name: str,
        model_desc: str,
        ckpt_path: Optional[str],
        init_elo: float,
        persona: str = "standard",
    ) -> None:
        if pid in disabled_set:
            return
        model = System1TrucoSLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=tokenizer.max_seq_len,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=384,
            dropout=0.0,
        )
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            # Fixed seed for untrained SLM so its behavior is reproducible
            torch.manual_seed(777)
            model.apply(model._init_weights)

        model.to(device).eval()
        stats = leaderboard.register_player(
            player_id=pid,
            display_name=name,
            model_type=model_desc,
            checkpoint_path=ckpt_path or "random_init",
            initial_elo=init_elo,
        )

        def _act(state: TrucoHandState, m: System1TrucoSLM = model, p_mode: str = persona) -> Action:
            return act_with_personality(m, state, tokenizer, device, persona=p_mode, temperature=0.05)

        roster[pid] = TrucoAgentWrapper(stats=stats, act_fn=_act)

    # 1. Frozen Base Model (Gen 0)
    base_ckpt = os.path.join(checkpoint_dir, "jev_base_slm.pt")
    if os.path.exists(base_ckpt):
        load_slm_agent(
            "jev_base",
            "JEV-Base (Gen 0)",
            "System1-SLM (4L SwiGLU Base)",
            base_ckpt,
            1500.0,
        )

    # 2. Generation Checkpoints (Gen 1..N)
    gen_files: List[Tuple[int, str]] = []
    if os.path.exists(checkpoint_dir):
        for fn in os.listdir(checkpoint_dir):
            if fn.startswith("jev_gen_") and fn.endswith("_slm.pt"):
                try:
                    g_num = int(fn.split("_")[2])
                    gen_files.append((g_num, os.path.join(checkpoint_dir, fn)))
                except ValueError:
                    pass
        gen_files.sort(key=lambda x: x[0])
        for g_num, g_path in gen_files:
            load_slm_agent(
                f"jev_gen_{g_num}",
                f"JEV-League (Gen {g_num})",
                f"System1-SLM (Dual-Play G{g_num})",
                g_path,
                1500.0,
            )

    # 3. Main Active Checkpoint
    main_ckpt = os.path.join(checkpoint_dir, "jev_truco_slm.pt")
    if os.path.exists(main_ckpt):
        load_slm_agent(
            "jev_champion",
            "JEV-Active (Latest)",
            "System1-SLM (4L SwiGLU Latest)",
            main_ckpt,
            1500.0,
        )

    # 4. Tactical Personality Specialists (powered by highest Gen SLM + Logit Modulation)
    best_persona_ckpt = gen_files[-1][1] if gen_files else (main_ckpt if os.path.exists(main_ckpt) else None)
    if best_persona_ckpt:
        load_slm_agent(
            "jev_bluffer",
            "El Mentiroso (Bluffer)",
            "SLM + Strategic Bluff Persona",
            best_persona_ckpt,
            1520.0,
            persona="bluffer",
        )
        load_slm_agent(
            "jev_phisher",
            "El Zorro (Phisher Trap)",
            "SLM + Slow-Play Achique Trap",
            best_persona_ckpt,
            1520.0,
            persona="phisher",
        )
        load_slm_agent(
            "jev_sheriff",
            "El Comisario (Anti-Bluff)",
            "SLM + Bayesian Bluff-Catcher",
            best_persona_ckpt,
            1520.0,
            persona="sheriff",
        )
        load_slm_agent(
            "jev_kamikaze",
            "El Temerario (High-Risk)",
            "SLM + Max-Pressure Escalator",
            best_persona_ckpt,
            1490.0,
            persona="risk_taker",
        )

    # 5. Untrained System 1 SLM (Zero-Shot Random Weights)
    load_slm_agent(
        "jev_untrained",
        "JEV-Untrained (Raw)",
        "System1-SLM (Untrained Init)",
        None,
        1250.0,
    )

    # 6. Heuristic Bluff Agent
    if "heuristic_gaucho" not in disabled_set:
        heur_engine = HeuristicAnchorAgent(random.Random(42))
        heur_stats = leaderboard.register_player(
            player_id="heuristic_gaucho",
            display_name="El Gaucho (Heuristic)",
            model_type="Rule-Based + Bluff Engine",
            checkpoint_path="truco/trainer.py",
            initial_elo=1480.0,
        )
        roster["heuristic_gaucho"] = TrucoAgentWrapper(stats=heur_stats, act_fn=heur_engine.act)

    # 7. Random Agent
    if "random_bot" not in disabled_set:
        rand_rng = random.Random(99)
        rand_stats = leaderboard.register_player(
            player_id="random_bot",
            display_name="Random Bot",
            model_type="Uniform Random Policy",
            checkpoint_path="-",
            initial_elo=1150.0,
        )
        roster["random_bot"] = TrucoAgentWrapper(
            stats=rand_stats,
            act_fn=lambda st: rand_rng.choice(st.legal_actions()),
        )

    return roster


def play_duplicate_match_pair(
    agent_a: TrucoAgentWrapper,
    agent_b: TrucoAgentWrapper,
    leaderboard: LeaderboardManager,
    rng: random.Random,
    target_score: int = 15,
    telemetry_cb: Optional[Callable[[Dict[str, object]], None]] = None,
) -> None:
    """
    Plays two mirrored 15-point matches between `agent_a` and `agent_b` using pre-sampled
    duplicate decks so both players experience the exact same sequence of card deals
    in swapped seats.
    """
    # Pre-sample up to 40 hands of 6 cards for this match pair
    deals = [rng.sample(DECK, 6) for _ in range(40)]

    for match_dir in (0, 1):
        t_game_start = time.perf_counter()
        decision_times_ms: List[float] = []
        scores = [0, 0]
        mano = 0
        deal_idx = 0

        while scores[0] < target_score and scores[1] < target_score:
            cards6 = deals[deal_idx % len(deals)]
            deal_idx += 1
            h0, h1 = cards6[:3], cards6[3:]

            state = TrucoHandState.new_hand(
                mano=mano,
                match_scores=(scores[0], scores[1]),
                target_score=target_score,
                dealt_cards=(h0, h1),
            )

            while not state.is_terminal:
                p = state.current_player
                t_dec = time.perf_counter()
                if (match_dir == 0 and p == 0) or (match_dir == 1 and p == 1):
                    act = agent_a.act(state)
                else:
                    act = agent_b.act(state)
                decision_times_ms.append((time.perf_counter() - t_dec) * 1000.0)
                state.step(act)

            pts0, pts1 = state.points_won[0], state.points_won[1]
            scores[0] += pts0
            scores[1] += pts1

            if match_dir == 0:
                leaderboard.record_hand(agent_a.stats.player_id, agent_b.stats.player_id, pts0, pts1)
            else:
                leaderboard.record_hand(agent_a.stats.player_id, agent_b.stats.player_id, pts1, pts0)

            mano = 1 - mano

        game_ms = (time.perf_counter() - t_game_start) * 1000.0
        if match_dir == 0:
            winner_is_a = scores[0] >= target_score
        else:
            winner_is_a = scores[1] >= target_score

        leaderboard.record_match(
            agent_a.stats.player_id,
            agent_b.stats.player_id,
            winner_is_a=winner_is_a,
            k=16.0,
        )

        if telemetry_cb is not None:
            winner_name = agent_a.stats.display_name if winner_is_a else agent_b.stats.display_name
            loser_name = agent_b.stats.display_name if winner_is_a else agent_a.stats.display_name
            avg_dec = sum(decision_times_ms) / max(1, len(decision_times_ms))
            telemetry_cb({
                "winner_name": winner_name,
                "loser_name": loser_name,
                "winner_pts": int(max(scores[0], scores[1])),
                "loser_pts": int(min(scores[0], scores[1])),
                "game_ms": round(game_ms, 2),
                "avg_decision_ms": round(avg_dec, 3),
                "decisions": len(decision_times_ms),
                "hands": deal_idx,
            })


def run_leaderboard_tournament(
    tokenizer: TrucoTokenizer,
    device: torch.device,
    rounds_per_pair: int = 15,
    reset: bool = False,
    telemetry_cb: Optional[Callable[[Dict[str, object]], None]] = None,
) -> LeaderboardManager:
    """
    Runs a full round-robin tournament across all registered models and saves/prints the leaderboard.
    Each `round_per_pair` plays 2 mirrored 15-point duplicate matches between every pair of players.
    """
    leaderboard = LeaderboardManager(LEADERBOARD_JSON_PATH)
    if reset:
        saved_disabled = list(leaderboard.disabled_players)
        leaderboard.players = {}
        leaderboard.head_to_head = {}
        leaderboard.disabled_players = saved_disabled
        leaderboard.save()

    roster = build_roster(leaderboard, tokenizer, device)
    pids = list(roster.keys())
    rng = random.Random(202610)

    total_pairs = len(pids) * (len(pids) - 1) // 2
    total_matches = max(1, rounds_per_pair * total_pairs * 2)
    completed_matches = 0
    print(f"Running Round-Robin Tournament across {len(pids)} models ({total_pairs} pairings, {rounds_per_pair * 2} matches/pair)...")

    def _wrapped_cb(ev: Dict[str, object]) -> None:
        nonlocal completed_matches
        completed_matches += 1
        ev["completed_matches"] = completed_matches
        ev["total_matches"] = total_matches
        ev["progress_fraction"] = completed_matches / total_matches
        if telemetry_cb is not None:
            telemetry_cb(ev)

    for r in range(rounds_per_pair):
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                agent_a = roster[pids[i]]
                agent_b = roster[pids[j]]
                play_duplicate_match_pair(
                    agent_a, agent_b, leaderboard, rng, telemetry_cb=_wrapped_cb
                )

        # Anchor `jev_base` at 1500 Elo so all ratings are directly interpretable relative to the Base Model
        if "jev_base" in leaderboard.players:
            drift = leaderboard.players["jev_base"].elo - 1500.0
            for p in leaderboard.players.values():
                p.elo -= drift

    leaderboard.save()
    return leaderboard
