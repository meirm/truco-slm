"""
HTTP API & React Web UI Server for JEV-Truco.

Provides endpoints for:
- `GET  /api/leaderboard`      : Ranked players, Elo, stats, and Head-to-Head matrix
- `GET  /api/training/status`  : Live status & logs of background King-of-the-Hill / Tournament jobs
- `POST /api/train/koth`       : Trigger King-of-the-Hill (Beat-the-Leader) training + Tournament refresh
- `POST /api/tournament`       : Trigger Round-Robin Leaderboard Tournament
- `POST /api/simulate`         : Simulate a full match or hand between any two bots and return
                                 every frame (cards, tricks, System-1 token prompt, action probabilities,
                                 value estimate, and speech bubbles) for the Video Replay Theater.
- `GET  /`                     : Serves the React Single-Page Web Application (`webui/index.html`)
"""

from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import torch

from truco.arena import run_championship_and_retrain_all, train_against_leader
from truco.engine import (
    Action,
    ACTION_NAMES,
    Card,
    DECK,
    PendingState,
    SUIT_NAMES,
    SUIT_SHORT,
    TrucoHandState,
)
from truco.leaderboard import (
    build_roster,
    LeaderboardManager,
    run_leaderboard_tournament,
)
from truco.model import System1TrucoSLM
from truco.rlhf import InteractivePlaySession, train_model_from_rlhf
from truco.tokenizer import compute_opponent_belief, TrucoTokenizer

BASE_CHECKPOINT = "checkpoints/jev_base_slm.pt"
DEFAULT_CHECKPOINT = "checkpoints/jev_truco_slm.pt"


class TrainingState:
    def __init__(self) -> None:
        self.is_running: bool = False
        self.job_type: str = "idle"
        self.progress_pct: float = 0.0
        self.logs: List[str] = [
            "Ready. Click 'Train vs #1 Leader (King of the Hill)' or 'Run Tournament' to start."
        ]
        self.last_result: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
        self.started_at: float = time.time()
        self.telemetry: Dict[str, Any] = {
            "games_completed": 0,
            "hands_completed": 0,
            "total_decisions": 0,
            "avg_decision_ms": 0.0,
            "last_decision_ms": 0.0,
            "avg_game_ms": 0.0,
            "last_game_ms": 0.0,
            "avg_hand_ms": 0.0,
            "elapsed_sec": 0.0,
            "game_history": [],
        }

    def reset_telemetry(self) -> None:
        with self.lock:
            self.started_at = time.time()
            self.telemetry = {
                "games_completed": 0,
                "hands_completed": 0,
                "total_decisions": 0,
                "avg_decision_ms": 0.0,
                "last_decision_ms": 0.0,
                "avg_game_ms": 0.0,
                "last_game_ms": 0.0,
                "avg_hand_ms": 0.0,
                "elapsed_sec": 0.0,
                "game_history": [],
            }

    def record_game_event(self, ev: Dict[str, Any], log_every: int = 8) -> None:
        with self.lock:
            t = self.telemetry
            g_count = int(t["games_completed"]) + 1
            h_add = int(ev.get("hands", 1))
            d_add = int(ev.get("decisions", 1))
            game_ms = float(ev.get("game_ms", 0.0))
            dec_ms = float(ev.get("avg_decision_ms", 0.0))

            prev_g = int(t["games_completed"])
            prev_d = int(t["total_decisions"])
            new_d = prev_d + d_add
            new_h = int(t["hands_completed"]) + h_add

            avg_game_ms = ((float(t["avg_game_ms"]) * prev_g) + game_ms) / max(1, g_count)
            avg_dec_ms = ((float(t["avg_decision_ms"]) * prev_d) + (dec_ms * d_add)) / max(1, new_d)
            avg_hand_ms = (avg_game_ms * g_count) / max(1, new_h)

            t["games_completed"] = g_count
            t["hands_completed"] = new_h
            t["total_decisions"] = new_d
            t["avg_decision_ms"] = round(avg_dec_ms, 3)
            t["last_decision_ms"] = round(dec_ms, 3)
            t["avg_game_ms"] = round(avg_game_ms, 2)
            t["last_game_ms"] = round(game_ms, 2)
            t["avg_hand_ms"] = round(avg_hand_ms, 2)
            t["elapsed_sec"] = round(time.time() - self.started_at, 1)

            hist: List[Dict[str, Any]] = t["game_history"]
            hist.append({
                "game_idx": g_count,
                "winner_name": str(ev.get("winner_name", "Winner")),
                "loser_name": str(ev.get("loser_name", "Loser")),
                "winner_pts": int(ev.get("winner_pts", 15)),
                "loser_pts": int(ev.get("loser_pts", 0)),
                "game_ms": round(game_ms, 2),
                "avg_decision_ms": round(dec_ms, 3),
                "hands": h_add,
            })
            if len(hist) > 50:
                t["game_history"] = hist[-50:]

            if g_count == 1 or g_count % log_every == 0:
                timestamp = time.strftime("%H:%M:%S")
                msg = (
                    f"[Game #{g_count}] {ev.get('winner_name')} def. {ev.get('loser_name')} "
                    f"({ev.get('winner_pts')}-{ev.get('loser_pts')}) | "
                    f"Game: {game_ms:.1f}ms ({h_add} hands) | Sys-1 Decision: {dec_ms:.2f}ms"
                )
                self.logs.append(f"[{timestamp}] {msg}")
                if len(self.logs) > 120:
                    self.logs = self.logs[-120:]

    def log(self, msg: str) -> None:
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {msg}")
            if len(self.logs) > 120:
                self.logs = self.logs[-120:]


GLOBAL_TRAIN_STATE = TrainingState()
_bundle_init_path = os.path.join(os.path.dirname(__file__), "docs", "data", "static_bundle.json")
if os.path.exists(_bundle_init_path):
    try:
        with open(_bundle_init_path, "r", encoding="utf-8") as _bf:
            _bdata = json.load(_bf).get("train_status", {})
        if _bdata:
            GLOBAL_TRAIN_STATE.job_type = _bdata.get("job_type", "idle")
            GLOBAL_TRAIN_STATE.progress_pct = float(_bdata.get("progress_pct", 0.0))
            GLOBAL_TRAIN_STATE.logs = list(_bdata.get("logs", GLOBAL_TRAIN_STATE.logs))
            if "telemetry" in _bdata:
                GLOBAL_TRAIN_STATE.telemetry = dict(_bdata["telemetry"])
    except Exception:
        pass
GLOBAL_PLAY_SESSION: Optional[InteractivePlaySession] = None
PLAY_LOCK = threading.Lock()


def get_play_session() -> InteractivePlaySession:
    global GLOBAL_PLAY_SESSION
    if GLOBAL_PLAY_SESSION is None:
        GLOBAL_PLAY_SESSION = InteractivePlaySession(get_device(), TrucoTokenizer())
    return GLOBAL_PLAY_SESSION


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Use CPU on macOS for multi-threaded HTTP + background tournament safety
    # (avoids IOGPUMetalCommandBuffer thread-safety aborts while keeping ~1.0ms System-1 latency)
    return torch.device("cpu")


def build_model(tokenizer: TrucoTokenizer) -> System1TrucoSLM:
    return System1TrucoSLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=384,
        dropout=0.0,
    )


def serialize_card(c: Card) -> Dict[str, Any]:
    return {
        "rank": c.rank,
        "suit": SUIT_NAMES[c.suit],
        "suit_short": SUIT_SHORT[c.suit],
        "short": c.short_str(),
        "name": str(c),
        "power": c.truco_power,
        "envido_val": c.envido_value,
    }


def format_action_label(state: TrucoHandState, player: int, action: Action) -> str:
    if action in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2):
        card = state.hands[player][int(action)]
        return f"Play {card} ({card.short_str()})"
    return ACTION_NAMES[action]


def snapshot_frame(
    state: TrucoHandState,
    hand_index: int,
    frame_index: int,
    match_scores: List[int],
    actor: Optional[int],
    action_taken: Optional[Action],
    action_label: str,
    speech_bubble: Optional[str],
    sys1_prompt: str,
    top_probs: List[Dict[str, Any]],
    value_est: Optional[float],
    event_banner: str,
) -> Dict[str, Any]:
    """Creates a complete visual frame for the React Video Replay Theater."""
    tricks_serialized = []
    for t_idx, trick in enumerate(state.tricks):
        winner = state.trick_winners[t_idx] if t_idx < len(state.trick_winners) else None
        tricks_serialized.append({
            "trick_index": t_idx,
            "winner": winner,
            "cards": [
                {
                    "player": tc.player,
                    "hand_slot": tc.hand_slot,
                    "card": serialize_card(tc.card),
                }
                for tc in trick
            ],
        })

    obs_player = actor if actor is not None else state.current_player
    actor_belief = compute_opponent_belief(state, obs_player)

    return {
        "hand_index": hand_index,
        "frame_index": frame_index,
        "match_scores": list(match_scores),
        "mano": state.mano,
        "current_player": state.current_player,
        "actor": actor,
        "action_code": action_taken.name if action_taken is not None else None,
        "action_label": action_label,
        "speech_bubble": speech_bubble,
        "event_banner": event_banner,
        "truco_level": state.truco_level,
        "truco_stakes": state.truco_stakes,
        "truco_owner": state.truco_owner,
        "pending_state": state.pending.name,
        "envido_resolved": state.envido_resolved,
        "envido_winner": state.envido_winner,
        "envido_awarded_points": state.envido_awarded_points,
        "is_terminal": state.is_terminal,
        "truco_winner": state.truco_winner,
        "points_won_hand": list(state.points_won),
        "hands": [
            [
                {
                    **serialize_card(c),
                    "played": state.played_mask[p_idx][s_idx],
                }
                for s_idx, c in enumerate(state.hands[p_idx])
            ]
            for p_idx in (0, 1)
        ],
        "envido_scores": list(state.envido_scores),
        "tricks": tricks_serialized,
        "sys1_prompt": sys1_prompt,
        "top_probs": top_probs,
        "value_est": value_est,
        "actor_belief": actor_belief,
    }


def simulate_recorded_game(
    player0_id: str,
    player1_id: str,
    num_hands: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulates `num_hands` hands between `player0_id` and `player1_id` and returns a complete
    list of video frames (deal frame, each turn decision with System-1 telemetry, and hand summary).
    """
    device = get_device()
    tokenizer = TrucoTokenizer()
    lb = LeaderboardManager()
    roster = build_roster(lb, tokenizer, device)

    if player0_id not in roster:
        player0_id = next(iter(roster.keys()))
    if player1_id not in roster:
        player1_id = list(roster.keys())[min(1, len(roster) - 1)]

    agent0 = roster[player0_id]
    agent1 = roster[player1_id]
    agents = [agent0, agent1]

    # Also load raw System1TrucoSLM instances if available so we can extract exact probability distributions & EVs
    slm_models: List[Optional[System1TrucoSLM]] = [None, None]
    for idx, ag in enumerate(agents):
        ckpt_p = ag.stats.checkpoint_path
        if ckpt_p and (ckpt_p.endswith(".pt") or ckpt_p == "random_init"):
            m = build_model(tokenizer)
            if os.path.exists(ckpt_p):
                ck = torch.load(ckpt_p, map_location=device, weights_only=False)
                m.load_state_dict(ck["model_state_dict"])
            else:
                torch.manual_seed(777)
                m.apply(m._init_weights)
            m.to(device).eval()
            slm_models[idx] = m

    rng = random.Random(seed if seed is not None else int(time.time() * 1000) % 1000000)
    frames: List[Dict[str, Any]] = []
    match_scores = [0, 0]
    frame_counter = 0

    for h_idx in range(1, num_hands + 1):
        mano = (h_idx - 1) % 2
        state = TrucoHandState.new_hand(
            mano=mano,
            match_scores=(match_scores[0], match_scores[1]),
            target_score=15,
            rng=rng,
        )

        # Initial frame: Cards dealt on table
        frames.append(
            snapshot_frame(
                state=state,
                hand_index=h_idx,
                frame_index=frame_counter,
                match_scores=match_scores,
                actor=None,
                action_taken=None,
                action_label="Cards Dealt",
                speech_bubble=None,
                sys1_prompt=tokenizer.format_prompt_text(state, state.current_player),
                top_probs=[],
                value_est=None,
                event_banner=f"Hand #{h_idx} Started — {agents[mano].stats.display_name} is Mano",
            )
        )
        frame_counter += 1

        while not state.is_terminal:
            p = state.current_player
            prompt_str = tokenizer.format_prompt_text(state, p)
            top_probs_list: List[Dict[str, Any]] = []
            val_est: Optional[float] = None

            if slm_models[p] is not None:
                chosen, prob_dict, val_est = slm_models[p].act(state, tokenizer, device, temperature=0.05)
                sorted_p = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
                for act_item, pr in sorted_p[:4]:
                    top_probs_list.append({
                        "action": format_action_label(state, p, act_item),
                        "prob": round(pr * 100.0, 1),
                    })
            else:
                chosen = agents[p].act(state)
                top_probs_list.append({
                    "action": format_action_label(state, p, chosen),
                    "prob": 100.0,
                })

            act_label = format_action_label(state, p, chosen)

            # Determine dramatic Truco speech bubble for calls/responses
            bubble: Optional[str] = None
            if chosen not in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2):
                bubble_map = {
                    Action.CALL_ENVIDO: "¡ENVIDO!",
                    Action.CALL_REAL_ENVIDO: "¡REAL ENVIDO!",
                    Action.CALL_FALTA_ENVIDO: "¡FALTA ENVIDO!",
                    Action.CALL_TRUCO: "¡TRUCO!",
                    Action.CALL_RETRUCO: "¡QUIERO RETRUCO!",
                    Action.CALL_VALE_CUATRO: "¡QUIERO VALE CUATRO!",
                    Action.QUIERO: "¡QUIERO!",
                    Action.NO_QUIERO: "No quiero...",
                    Action.IRSE_AL_MAZO: "Me voy al mazo.",
                }
                bubble = bubble_map.get(chosen, ACTION_NAMES[chosen])

            state.step(chosen)

            banner = f"{agents[p].stats.display_name}: {act_label}"
            if state.is_terminal:
                match_scores[0] += state.points_won[0]
                match_scores[1] += state.points_won[1]
                banner += (
                    f"  |  Hand Finished! +{state.points_won[0]} pts for {agent0.stats.display_name}, "
                    f"+{state.points_won[1]} pts for {agent1.stats.display_name}"
                )

            frames.append(
                snapshot_frame(
                    state=state,
                    hand_index=h_idx,
                    frame_index=frame_counter,
                    match_scores=match_scores,
                    actor=p,
                    action_taken=chosen,
                    action_label=act_label,
                    speech_bubble=bubble,
                    sys1_prompt=prompt_str,
                    top_probs=top_probs_list,
                    value_est=round(val_est, 2) if val_est is not None else None,
                    event_banner=banner,
                )
            )
            frame_counter += 1

    return {
        "player0": {
            "id": agent0.stats.player_id,
            "name": agent0.stats.display_name,
            "model_type": agent0.stats.model_type,
            "elo": round(agent0.stats.elo),
        },
        "player1": {
            "id": agent1.stats.player_id,
            "name": agent1.stats.display_name,
            "model_type": agent1.stats.model_type,
            "elo": round(agent1.stats.elo),
        },
        "final_scores": match_scores,
        "num_hands": num_hands,
        "frames": frames,
    }


def _background_koth_worker(generations: int, hands_per_gen: int, epochs: int, tournament_rounds: int) -> None:
    try:
        GLOBAL_TRAIN_STATE.reset_telemetry()
        GLOBAL_TRAIN_STATE.is_running = True
        GLOBAL_TRAIN_STATE.job_type = "King of the Hill (Challenge Leader)"
        GLOBAL_TRAIN_STATE.progress_pct = 5.0

        device = get_device()
        tokenizer = TrucoTokenizer()
        lb = LeaderboardManager()

        ranked = [
            p for p in lb.get_ranked_players()
            if os.path.exists(p.checkpoint_path) and p.checkpoint_path.endswith(".pt")
        ]
        if ranked:
            leader_stats = ranked[0]
            leader_ckpt_path = leader_stats.checkpoint_path
            leader_name = f"{leader_stats.display_name} ({leader_stats.elo:.0f} Elo)"
        else:
            leader_ckpt_path = BASE_CHECKPOINT if os.path.exists(BASE_CHECKPOINT) else DEFAULT_CHECKPOINT
            leader_name = "JEV-Base (Gen 0)"

        GLOBAL_TRAIN_STATE.log(f"Selected #1 Leaderboard Champion: {leader_name}")

        leader_model = build_model(tokenizer)
        if os.path.exists(leader_ckpt_path):
            l_ckpt = torch.load(leader_ckpt_path, map_location=device, weights_only=False)
            leader_model.load_state_dict(l_ckpt["model_state_dict"])

        base_model = build_model(tokenizer)
        if os.path.exists(BASE_CHECKPOINT):
            b_ckpt = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
            base_model.load_state_dict(b_ckpt["model_state_dict"])
        else:
            base_model.load_state_dict(leader_model.state_dict())

        extra_opponents: List[System1TrucoSLM] = []
        for runner_up in ranked[1:4]:
            opp_m = build_model(tokenizer)
            r_ckpt = torch.load(runner_up.checkpoint_path, map_location=device, weights_only=False)
            opp_m.load_state_dict(r_ckpt["model_state_dict"])
            extra_opponents.append(opp_m)

        challenger_model = build_model(tokenizer)
        challenger_model.load_state_dict(leader_model.state_dict())

        for g in range(1, generations + 1):
            GLOBAL_TRAIN_STATE.log(
                f"Generation {g}/{generations}: Rolling out {hands_per_gen} counterfactual hands vs {leader_name} + Top Gauntlet..."
            )
            gen_base_pct = 5.0 + 65.0 * ((g - 1) / max(1, generations))
            gen_span_pct = 65.0 / max(1, generations)

            def _koth_telemetry(ev: Dict[str, Any], b_pct: float = gen_base_pct, s_pct: float = gen_span_pct) -> None:
                GLOBAL_TRAIN_STATE.record_game_event(ev, log_every=4)
                frac = float(ev.get("progress_fraction", 0.0))
                GLOBAL_TRAIN_STATE.progress_pct = round(b_pct + s_pct * frac * 0.85, 1)

            h2h = train_against_leader(
                challenger_model=challenger_model,
                leader_model=leader_model,
                leader_name=leader_name,
                base_model=base_model,
                tokenizer=tokenizer,
                device=device,
                extra_opponents=extra_opponents,
                rounds=hands_per_gen,
                epochs=epochs,
                batch_size=64,
                lr=5e-4,
                telemetry_cb=_koth_telemetry,
            )

            existing_gens = []
            for fn in os.listdir("checkpoints"):
                if fn.startswith("jev_gen_") and fn.endswith("_slm.pt"):
                    try:
                        existing_gens.append(int(fn.split("_")[2]))
                    except ValueError:
                        pass
            next_gen = (max(existing_gens) + 1) if existing_gens else 1
            gen_path = f"checkpoints/jev_gen_{next_gen}_slm.pt"
            torch.save({"model_state_dict": challenger_model.state_dict(), "gen": next_gen}, gen_path)
            torch.save({"model_state_dict": challenger_model.state_dict(), "gen": next_gen}, DEFAULT_CHECKPOINT)

            GLOBAL_TRAIN_STATE.log(
                f"Gen {next_gen} vs {leader_name} -> Net Pts/Hand: {h2h.net_pts_per_hand:+.3f} | "
                f"Match Win: {h2h.match_win_rate*100:.1f}% (Elo diff: {h2h.elo_delta:+.0f})"
            )

            if h2h.net_pts_per_hand > 0.02 or h2h.match_win_rate >= 0.52:
                extra_opponents.insert(0, build_model(tokenizer))
                extra_opponents[0].load_state_dict(leader_model.state_dict())
                leader_model.load_state_dict(challenger_model.state_dict())
                leader_name = f"JEV-League (Gen {next_gen}) [New King]"
                GLOBAL_TRAIN_STATE.log(f"👑 DETHRONED! Promoted JEV-League (Gen {next_gen}) to new Leader!")

            GLOBAL_TRAIN_STATE.progress_pct = round(5.0 + 65.0 * (g / generations), 1)

        if tournament_rounds > 0:
            GLOBAL_TRAIN_STATE.log(f"Running Round-Robin Tournament ({tournament_rounds} duplicate rounds/pair) to update Leaderboard...")

            def _tourn_telemetry(ev: Dict[str, Any]) -> None:
                GLOBAL_TRAIN_STATE.record_game_event(ev, log_every=15)
                frac = float(ev.get("progress_fraction", 0.0))
                GLOBAL_TRAIN_STATE.progress_pct = round(70.0 + 29.0 * frac, 1)

            run_leaderboard_tournament(
                tokenizer, device, rounds_per_pair=tournament_rounds, reset=True, telemetry_cb=_tourn_telemetry
            )

        GLOBAL_TRAIN_STATE.progress_pct = 100.0
        GLOBAL_TRAIN_STATE.log("King-of-the-Hill Training & Leaderboard update complete!")
    except Exception as e:
        GLOBAL_TRAIN_STATE.log(f"Error during training: {e}")
    finally:
        GLOBAL_TRAIN_STATE.is_running = False
        GLOBAL_TRAIN_STATE.job_type = "idle"


def _background_tournament_worker(rounds_per_pair: int, reset: bool) -> None:
    try:
        GLOBAL_TRAIN_STATE.reset_telemetry()
        GLOBAL_TRAIN_STATE.is_running = True
        GLOBAL_TRAIN_STATE.job_type = "Round-Robin Tournament"
        GLOBAL_TRAIN_STATE.progress_pct = 5.0
        GLOBAL_TRAIN_STATE.log(f"Starting Round-Robin Tournament ({rounds_per_pair} duplicate match rounds per pair)...")
        device = get_device()
        tokenizer = TrucoTokenizer()

        def _tourn_telemetry(ev: Dict[str, Any]) -> None:
            GLOBAL_TRAIN_STATE.record_game_event(ev, log_every=12)
            frac = float(ev.get("progress_fraction", 0.0))
            GLOBAL_TRAIN_STATE.progress_pct = round(5.0 + 94.0 * frac, 1)

        run_leaderboard_tournament(
            tokenizer, device, rounds_per_pair=rounds_per_pair, reset=reset, telemetry_cb=_tourn_telemetry
        )
        GLOBAL_TRAIN_STATE.progress_pct = 100.0
        GLOBAL_TRAIN_STATE.log("Tournament complete! Leaderboard rankings updated.")
    except Exception as e:
        GLOBAL_TRAIN_STATE.log(f"Tournament error: {e}")
    finally:
        GLOBAL_TRAIN_STATE.is_running = False
        GLOBAL_TRAIN_STATE.job_type = "idle"


def _background_championship_retrain_worker(championship_rounds: int, epochs: int) -> None:
    try:
        GLOBAL_TRAIN_STATE.reset_telemetry()
        GLOBAL_TRAIN_STATE.is_running = True
        GLOBAL_TRAIN_STATE.job_type = "Championship + Collective All-Model Retraining"
        GLOBAL_TRAIN_STATE.progress_pct = 5.0
        phase = {"step": 1}

        def _log_with_progress(msg: str) -> None:
            GLOBAL_TRAIN_STATE.log(msg)
            if msg.startswith("[Step 2/4]"):
                phase["step"] = 2
                GLOBAL_TRAIN_STATE.progress_pct = 40.0
            elif msg.startswith("[Step 3/4]"):
                phase["step"] = 3
                GLOBAL_TRAIN_STATE.progress_pct = 45.0
            elif msg.startswith("  ["):
                GLOBAL_TRAIN_STATE.progress_pct = min(74.0, GLOBAL_TRAIN_STATE.progress_pct + 2.5)
            elif msg.startswith("[Step 4/4]"):
                phase["step"] = 4
                GLOBAL_TRAIN_STATE.progress_pct = 75.0

        def _champ_telemetry(ev: Dict[str, Any]) -> None:
            GLOBAL_TRAIN_STATE.record_game_event(ev, log_every=12)
            frac = float(ev.get("progress_fraction", 0.0))
            if phase["step"] == 1:
                GLOBAL_TRAIN_STATE.progress_pct = round(5.0 + 34.0 * frac, 1)
            elif phase["step"] == 4:
                GLOBAL_TRAIN_STATE.progress_pct = round(75.0 + 24.0 * frac, 1)

        device = get_device()
        tokenizer = TrucoTokenizer()
        summary = run_championship_and_retrain_all(
            tokenizer=tokenizer,
            device=device,
            championship_rounds=championship_rounds,
            epochs=epochs,
            log_fn=_log_with_progress,
            telemetry_cb=_champ_telemetry,
        )
        GLOBAL_TRAIN_STATE.last_result = summary
        GLOBAL_TRAIN_STATE.progress_pct = 100.0
    except Exception as e:
        GLOBAL_TRAIN_STATE.log(f"Championship & Retrain error: {e}")
    finally:
        GLOBAL_TRAIN_STATE.is_running = False
        GLOBAL_TRAIN_STATE.job_type = "idle"


class TrucoWebRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serialize_leaderboard(self, lb: LeaderboardManager) -> Dict[str, Any]:
        ranked = lb.get_ranked_players()
        players_list = []
        for idx, p in enumerate(ranked, start=1):
            players_list.append({
                "rank": idx,
                "player_id": p.player_id,
                "display_name": p.display_name,
                "model_type": p.model_type,
                "checkpoint_path": p.checkpoint_path,
                "elo": round(p.elo),
                "matches_played": p.matches_played,
                "matches_won": p.matches_won,
                "matches_lost": p.matches_lost,
                "match_win_rate": round(p.match_win_rate, 1),
                "hands_played": p.hands_played,
                "hands_won": p.hands_won,
                "hands_lost": p.hands_lost,
                "net_pts_per_hand": round(p.net_pts_per_hand, 3),
            })
        return {
            "players": players_list,
            "head_to_head": lb.head_to_head,
            "disabled_players": lb.disabled_players,
            "disabled_count": len(lb.disabled_players),
        }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/leaderboard":
            lb = LeaderboardManager()
            if not lb.players:
                device = get_device()
                tokenizer = TrucoTokenizer()
                build_roster(lb, tokenizer, device)
            self._send_json(self._serialize_leaderboard(lb))
            return

        if parsed.path == "/api/training/status":
            with GLOBAL_TRAIN_STATE.lock:
                if GLOBAL_TRAIN_STATE.is_running:
                    GLOBAL_TRAIN_STATE.telemetry["elapsed_sec"] = round(
                        time.time() - GLOBAL_TRAIN_STATE.started_at, 1
                    )
                self._send_json({
                    "is_running": GLOBAL_TRAIN_STATE.is_running,
                    "job_type": GLOBAL_TRAIN_STATE.job_type,
                    "progress_pct": GLOBAL_TRAIN_STATE.progress_pct,
                    "logs": list(GLOBAL_TRAIN_STATE.logs),
                    "last_result": GLOBAL_TRAIN_STATE.last_result,
                    "telemetry": dict(GLOBAL_TRAIN_STATE.telemetry),
                })
            return

        if parsed.path == "/api/play/state":
            with PLAY_LOCK:
                session = get_play_session()
                if not session.state:
                    session.start_match("jev_gen_16")
                    by_short = {c.short_str(): c for c in DECK}
                    h_human = [by_short["7E"], by_short["7O"], by_short["4C"]]
                    h_ai = [by_short["1B"], by_short["12E"], by_short["11B"]]
                    session.state = TrucoHandState.new_hand(mano=1, match_scores=(6, 4), dealt_cards=(h_human, h_ai))
                    session.state.step(Action.CALL_ENVIDO)
                    session.state.step(Action.QUIERO)
                    session.state.step(Action.PLAY_CARD_1)
                self._send_json(session.serialize_session())
            return

        if parsed.path.startswith("/docs/"):
            rel_path = parsed.path.lstrip("/")
            full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))
            root_dir = os.path.abspath(os.path.dirname(__file__))
            if full_path.startswith(root_dir) and os.path.isfile(full_path):
                with open(full_path, "rb") as f:
                    content = f.read()
                ctype = "application/json; charset=utf-8" if full_path.endswith(".json") else (
                    "image/gif" if full_path.endswith(".gif") else "image/png"
                )
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # Serve index.html for root or UI paths
        ui_path = os.path.join(os.path.dirname(__file__), "webui", "index.html")
        if os.path.exists(ui_path):
            with open(ui_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_json({"error": "webui/index.html not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if parsed.path == "/api/play/start":
            with PLAY_LOCK:
                session = get_play_session()
                opp_id = body.get("opponent_id", "jev_champion")
                hide_id = bool(body.get("hide_identity", False))
                self._send_json(session.start_match(opp_id, hide_identity=hide_id))
            return

        if parsed.path == "/api/play/hide-identity":
            with PLAY_LOCK:
                session = get_play_session()
                hide_id = bool(body.get("hide_identity", False))
                self._send_json(session.set_hide_identity(hide_id))
            return

        if parsed.path == "/api/play/action":
            with PLAY_LOCK:
                session = get_play_session()
                act_idx = int(body.get("action_idx", 0))
                self._send_json(session.step_human(act_idx))
            return

        if parsed.path == "/api/play/next-hand":
            with PLAY_LOCK:
                session = get_play_session()
                self._send_json(session.start_next_hand())
            return

        if parsed.path == "/api/play/feedback":
            with PLAY_LOCK:
                session = get_play_session()
                turn_id = body.get("turn_id", "")
                fb_type = body.get("feedback_type", "thumbs_up")
                pref_idx = body.get("preferred_action_idx")
                self._send_json(session.submit_turn_feedback(turn_id, fb_type, pref_idx))
            return

        if parsed.path == "/api/play/train-rlhf":
            with PLAY_LOCK:
                session = get_play_session()
                epochs = int(body.get("epochs", 8))
                res = train_model_from_rlhf(session, epochs=epochs)
                self._send_json(res)
            return

        if parsed.path == "/api/train/koth":
            if GLOBAL_TRAIN_STATE.is_running:
                self._send_json({"error": "A training or tournament job is already running."}, status=409)
                return
            gens = int(body.get("generations", 1))
            hands = int(body.get("hands_per_gen", 250))
            epochs = int(body.get("epochs", 4))
            t_rounds = int(body.get("tournament_rounds", 4))
            t = threading.Thread(
                target=_background_koth_worker,
                args=(gens, hands, epochs, t_rounds),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "job": "King of the Hill"})
            return

        if parsed.path == "/api/train/championship-retrain":
            if GLOBAL_TRAIN_STATE.is_running:
                self._send_json({"error": "A training or tournament job is already running."}, status=409)
                return
            c_rounds = int(body.get("championship_rounds", 2))
            epochs = int(body.get("epochs", 5))
            t = threading.Thread(
                target=_background_championship_retrain_worker,
                args=(c_rounds, epochs),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "job": "Championship + Retrain All Models"})
            return

        if parsed.path == "/api/tournament":
            if GLOBAL_TRAIN_STATE.is_running:
                self._send_json({"error": "A job is already running."}, status=409)
                return
            rounds = int(body.get("rounds", 5))
            reset = bool(body.get("reset", True))
            t = threading.Thread(
                target=_background_tournament_worker,
                args=(rounds, reset),
                daemon=True,
            )
            t.start()
            self._send_json({"status": "started", "job": "Round-Robin Tournament"})
            return

        if parsed.path == "/api/leaderboard/prune":
            device = get_device()
            tokenizer = TrucoTokenizer()
            lb = LeaderboardManager()
            build_roster(lb, tokenizer, device)
            mode = body.get("mode", "top_k")
            removed: List[str] = []
            if mode == "top_k":
                keep_k = int(body.get("keep_k", 8))
                removed = lb.prune_to_top_k(keep_k=keep_k)
            elif mode == "core_8":
                removed = lb.prune_to_core_8()
                build_roster(lb, tokenizer, device)
            elif mode == "win_rate":
                min_wr = float(body.get("min_win_rate", 50.0))
                removed = lb.prune_below_win_rate(min_win_rate=min_wr)
            elif mode == "remove_one":
                pid = str(body.get("player_id", ""))
                if pid:
                    removed = lb.remove_player(pid)
            elif mode == "restore_all":
                lb.restore_all_players()
                build_roster(lb, tokenizer, device)
            payload = self._serialize_leaderboard(lb)
            payload["removed"] = removed
            self._send_json(payload)
            return

        if parsed.path == "/api/simulate":
            p0 = body.get("player0_id", "jev_champion")
            p1 = body.get("player1_id", "jev_base")
            num_hands = max(1, min(15, int(body.get("num_hands", 4))))
            sim_data = simulate_recorded_game(p0, p1, num_hands=num_hands)
            self._send_json(sim_data)
            return

        self._send_json({"error": "Unknown endpoint"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="JEV-Truco React Web UI & REST Server")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TrucoWebRequestHandler)
    print("=" * 72)
    print(f" 🃏 JEV-Truco React Web UI & Video Replay Server running at:")
    print(f"    http://{args.host}:{args.port}")
    print("=" * 72)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
