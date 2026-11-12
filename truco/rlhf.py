"""
Interactive Human-in-the-Loop (RLHF / Direct Action Critique + Human Imitation) Engine
for JEV-Truco.

Allows the user to:
1. Pick any top model from the Leaderboard (`jev_champion`, `jev_gen_8`, `jev_base`, etc.)
   and play live 15-point Argentine Truco hands against it in the React Web UI.
2. Automatically records every human decision `(token_ids, legal_mask, human_action)` as
   an expert human demonstration.
3. On every AI turn, records the AI's state and action so the user can give one-click
   Human Feedback:
   - 👍 **Approve Move (`+1`)**: Reinforces the AI's chosen action.
   - 👎 **Critique / Override Move (`-1` + Better Action)**: Penalizes the AI's blunder and
     trains the AI's System 1 policy head directly toward the human's preferred action.
4. Supports instant online fine-tuning (`train_from_human_feedback`) that updates the selected
   model (or saves a new RLHF generation checkpoint) and refreshes the Leaderboard.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from truco.engine import (
    Action,
    ACTION_NAMES,
    Card,
    NUM_ACTIONS,
    PendingState,
    SUIT_NAMES,
    SUIT_SHORT,
    TrucoHandState,
)
from truco.leaderboard import build_roster, LeaderboardManager
from truco.model import System1TrucoSLM
from truco.tokenizer import compute_opponent_belief, TrucoTokenizer

RLHF_DATASET_PATH = "checkpoints/human_feedback.json"
DEFAULT_CHECKPOINT = "checkpoints/jev_truco_slm.pt"


@dataclass
class FeedbackItem:
    item_id: str
    timestamp: str
    source: str  # "human_play", "thumbs_up", "override", "hindsight_reveal"
    model_id: str
    prompt_text: str
    token_ids: List[int]
    legal_actions: List[int]
    chosen_action: int
    preferred_action: int
    weight: float
    hand_reward: float = 0.0


class RLHFBuffer:
    def __init__(self, path: str = RLHF_DATASET_PATH) -> None:
        self.path = path
        self.items: List[FeedbackItem] = []
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.items = [FeedbackItem(**x) for x in raw.get("items", [])]
            except Exception:
                self.items = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"items": [asdict(x) for x in self.items]}, f, indent=2)

    def add_item(self, item: FeedbackItem) -> None:
        # Replace if same item_id already exists (e.g. user changed hindsight/thumbs up to explicit override)
        self.items = [x for x in self.items if x.item_id != item.item_id]
        self.items.append(item)
        self.save()


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


class InteractivePlaySession:
    """
    Manages a live Human (Player 0) vs Selected AI Model (Player 1) match session,
    recording every AI move so the human can inspect and critique/override it in real time,
    computing Bayesian hidden-card beliefs during play, and running End-of-Hand Hindsight
    Counterfactual Replay once all 6 cards are known.
    """

    def __init__(self, device: torch.device, tokenizer: TrucoTokenizer) -> None:
        self.device = device
        self.tokenizer = tokenizer
        self.rlhf_buffer = RLHFBuffer()

        self.opponent_id: str = "jev_champion"
        self.opponent_name: str = "JEV-Active (Latest)"
        self.opponent_model_type: str = "System1-SLM"
        self.opponent_elo: int = 1600
        self.opponent_ckpt: str = DEFAULT_CHECKPOINT
        self.opponent_model: Optional[System1TrucoSLM] = None
        self.hide_identity: bool = False

        self.match_scores: List[int] = [0, 0]
        self.hand_number: int = 0
        self.mano: int = 0
        self.state: Optional[TrucoHandState] = None
        self.match_over: bool = False
        self.match_winner: Optional[int] = None

        # History of turns in the current hand (for UI display & human feedback buttons)
        self.hand_log: List[Dict[str, Any]] = []
        # Cloned state snapshots per AI turn for End-of-Hand Hindsight Counterfactual Replay
        self._ai_turn_snapshots: Dict[str, TrucoHandState] = {}
        # Pending human trajectory in current hand (rewarded once hand finishes)
        self.pending_human_steps: List[FeedbackItem] = []

    def set_hide_identity(self, hide_identity: bool) -> Dict[str, Any]:
        self.hide_identity = bool(hide_identity)
        return self.serialize_session()

    def start_match(self, opponent_id: str, hide_identity: bool = False) -> Dict[str, Any]:
        lb = LeaderboardManager()
        roster = build_roster(lb, self.tokenizer, self.device)
        self.hide_identity = bool(hide_identity)

        if opponent_id == "mystery" or (self.hide_identity and opponent_id == "mystery"):
            # Sample a competitive opponent (including Tactical Personality Specialists & Top SLMs)
            candidates = [
                pid for pid in roster.keys()
                if pid not in ("human_player", "random_bot", "jev_untrained")
            ]
            opponent_id = random.choice(candidates) if candidates else next(iter(roster.keys()))
        elif opponent_id not in roster:
            opponent_id = next(iter(roster.keys()))

        opp_wrapper = roster[opponent_id]
        self.opponent_id = opponent_id
        self.opponent_name = opp_wrapper.stats.display_name
        self.opponent_model_type = opp_wrapper.stats.model_type
        self.opponent_elo = round(opp_wrapper.stats.elo)
        self.opponent_ckpt = opp_wrapper.stats.checkpoint_path

        # Load neural model if checkpoint exists
        self.opponent_model = System1TrucoSLM(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=self.tokenizer.max_seq_len,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=384,
            dropout=0.0,
        )
        if self.opponent_ckpt and os.path.exists(self.opponent_ckpt) and self.opponent_ckpt.endswith(".pt"):
            ck = torch.load(self.opponent_ckpt, map_location=self.device, weights_only=False)
            self.opponent_model.load_state_dict(ck["model_state_dict"])
        self.opponent_model.to(self.device).eval()
        self.opp_wrapper = opp_wrapper

        self.match_scores = [0, 0]
        self.hand_number = 0
        self.mano = 0
        self.match_over = False
        self.match_winner = None

        return self.start_next_hand()

    def start_next_hand(self) -> Dict[str, Any]:
        if self.match_scores[0] >= 15 or self.match_scores[1] >= 15:
            self.match_over = True
            return self.serialize_session()

        self.hand_number += 1
        self.mano = (self.hand_number - 1) % 2
        self.state = TrucoHandState.new_hand(
            mano=self.mano,
            match_scores=(self.match_scores[0], self.match_scores[1]),
            target_score=15,
        )
        self.hand_log = []
        self._ai_turn_snapshots = {}
        self.pending_human_steps = []

        # If AI is Mano (Player 1), let AI act until it is Human's turn (Player 0) or hand ends
        self._advance_ai_turns()
        return self.serialize_session()

    def step_human(self, action_idx: int) -> Dict[str, Any]:
        if self.state is None or self.state.is_terminal:
            return self.serialize_session()
        if self.state.current_player != 0:
            return self.serialize_session()

        act = Action(action_idx)
        legal = self.state.legal_actions()
        if act not in legal:
            raise ValueError(f"Illegal human action {act}")

        # Record human demonstration state before stepping
        toks = self.tokenizer.encode_observation(self.state, 0)
        if len(toks) > self.tokenizer.max_seq_len:
            toks = [toks[0]] + toks[-(self.tokenizer.max_seq_len - 1):]
        prompt_str = self.tokenizer.format_prompt_text(self.state, 0)
        act_label = format_action_label(self.state, 0, act)

        item_id = f"hum_{int(time.time()*1000)}_{len(self.hand_log)}"
        fb_item = FeedbackItem(
            item_id=item_id,
            timestamp=time.strftime("%H:%M:%S"),
            source="human_play",
            model_id=self.opponent_id,
            prompt_text=prompt_str,
            token_ids=toks,
            legal_actions=[int(a) for a in legal],
            chosen_action=int(act),
            preferred_action=int(act),
            weight=1.0,
        )
        self.pending_human_steps.append(fb_item)

        self.hand_log.append({
            "turn_id": item_id,
            "actor": 0,
            "actor_name": "You (Human)",
            "action_idx": int(act),
            "action_label": act_label,
            "speech_bubble": self._speech_bubble_for(act),
            "legal_options": [],
            "feedback_status": "human_demo",
        })

        self.state.step(act)
        if self.state.is_terminal:
            self._finalize_hand()
        else:
            self._advance_ai_turns()

        return self.serialize_session()

    def _advance_ai_turns(self) -> None:
        assert self.state is not None
        while not self.state.is_terminal and self.state.current_player == 1:
            legal = self.state.legal_actions()
            toks = self.tokenizer.encode_observation(self.state, 1)
            if len(toks) > self.tokenizer.max_seq_len:
                toks = [toks[0]] + toks[-(self.tokenizer.max_seq_len - 1):]
            prompt_str = self.tokenizer.format_prompt_text(self.state, 1)
            belief = compute_opponent_belief(self.state, 1)

            top_probs_list: List[Dict[str, Any]] = []
            val_est: Optional[float] = None
            if self.opponent_model is not None and self.opponent_ckpt.endswith(".pt"):
                raw_chosen, prob_dict, val_est = self.opponent_model.act(
                    self.state, self.tokenizer, self.device, temperature=0.05
                )
                # Use wrapper action for tactical personality bots (El Mentiroso, El Zorro, El Comisario, El Temerario)
                if self.opponent_id in ("jev_bluffer", "jev_phisher", "jev_sheriff", "jev_kamikaze"):
                    chosen = self.opp_wrapper.act(self.state)
                else:
                    chosen = raw_chosen
                for a_item, pr in sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True):
                    top_probs_list.append({
                        "action_idx": int(a_item),
                        "label": format_action_label(self.state, 1, a_item),
                        "prob": round(pr * 100.0, 1),
                    })
            else:
                chosen = self.opp_wrapper.act(self.state)
                for a_item in legal:
                    top_probs_list.append({
                        "action_idx": int(a_item),
                        "label": format_action_label(self.state, 1, a_item),
                        "prob": 100.0 if a_item == chosen else 0.0,
                    })

            act_label = format_action_label(self.state, 1, chosen)
            turn_id = f"ai_{int(time.time()*1000)}_{len(self.hand_log)}"
            self._ai_turn_snapshots[turn_id] = self.state.clone()

            # Save snapshot of AI turn so the Human can click Thumbs Up or Critique/Override!
            legal_options = [
                {
                    "action_idx": int(a),
                    "label": format_action_label(self.state, 1, a),
                }
                for a in legal
            ]
            self.hand_log.append({
                "turn_id": turn_id,
                "actor": 1,
                "actor_name": self.opponent_name,
                "action_idx": int(chosen),
                "action_label": act_label,
                "speech_bubble": self._speech_bubble_for(chosen),
                "prompt_text": prompt_str,
                "token_ids": toks,
                "legal_actions": [int(a) for a in legal],
                "legal_options": legal_options,
                "top_probs": top_probs_list,
                "value_est": round(val_est, 2) if val_est is not None else None,
                "ai_belief": belief,
                "feedback_status": None,
                "preferred_action_idx": int(chosen),
                "hindsight": None,
            })

            self.state.step(chosen)
            if self.state.is_terminal:
                self._finalize_hand()
                break

    def _run_hindsight_rollout(self, branch_state: TrucoHandState, first_ai_action: Action) -> float:
        """
        Deterministically rolls out a counterfactual AI action from `branch_state` against
        the human's ACTUAL 3 cards (`branch_state.hands[0]`) to see what would have happened
        at the end of the hand once all hidden cards are revealed.
        Returns net points for AI (`pts_ai - pts_human`).
        """
        sim = branch_state.clone()
        sim.step(first_ai_action)
        safety = 0
        while not sim.is_terminal and safety < 35:
            safety += 1
            cur_p = sim.current_player
            if self.opponent_model is not None and self.opponent_ckpt.endswith(".pt"):
                a_next, _, _ = self.opponent_model.act(
                    sim, self.tokenizer, self.device, temperature=0.0
                )
            else:
                a_next = self.opp_wrapper.act(sim)
            sim.step(a_next)
        return float(sim.points_won[1] - sim.points_won[0])

    def _finalize_hand(self) -> None:
        assert self.state is not None
        pts0, pts1 = self.state.points_won[0], self.state.points_won[1]
        self.match_scores[0] += pts0
        self.match_scores[1] += pts1

        # 1. Record human demonstration steps that won net points (or tied/won hand) into RLHF buffer
        net_human = float(pts0 - pts1)
        net_ai_actual = float(pts1 - pts0)
        for h_item in self.pending_human_steps:
            h_item.hand_reward = net_human
            if net_human >= 0:
                h_item.weight = 1.0 + 0.25 * net_human
                self.rlhf_buffer.add_item(h_item)
        self.pending_human_steps.clear()

        # 2. END-OF-HAND FULL-CARD HINDSIGHT COUNTERFACTUAL REPLAY:
        # Now that the hand is over, the AI inspects all 3 of the human's actual cards
        # (`self.state.hands[0]`) and evaluates every legal move it could have made at each turn.
        human_cards_str = ", ".join(c.short_str() for c in self.state.hands[0])
        for entry in self.hand_log:
            if entry["actor"] != 1:
                continue
            t_id = entry["turn_id"]
            snap = self._ai_turn_snapshots.get(t_id)
            if snap is None:
                continue

            chosen_act = Action(entry["action_idx"])
            evals: Dict[int, float] = {}
            for a_idx in entry["legal_actions"]:
                a_enum = Action(a_idx)
                if a_enum == chosen_act:
                    # Compare against both actual realized outcome and simulated rollout
                    ro_val = self._run_hindsight_rollout(snap, a_enum)
                    evals[a_idx] = 0.5 * net_ai_actual + 0.5 * ro_val
                else:
                    evals[a_idx] = self._run_hindsight_rollout(snap, a_enum)

            best_a_idx = max(evals.keys(), key=lambda k: evals[k])
            best_val = evals[best_a_idx]
            chosen_val = evals.get(int(chosen_act), net_ai_actual)
            gain = round(best_val - chosen_val, 2)
            best_label = format_action_label(snap, 1, Action(best_a_idx))

            entry["hindsight"] = {
                "human_cards_revealed": human_cards_str,
                "best_action_idx": int(best_a_idx),
                "best_action_label": best_label,
                "chosen_ev": round(chosen_val, 2),
                "best_ev": round(best_val, 2),
                "regret_gain": gain,
            }

            # If Hindsight reveals a clear mistake (gain >= 1.0 pt against human's actual cards)
            # and human hasn't manually overridden this turn yet, automatically store a hindsight sample!
            if gain >= 1.0 and best_a_idx != int(chosen_act) and entry.get("feedback_status") is None:
                entry["feedback_status"] = "hindsight_auto"
                entry["preferred_action_idx"] = int(best_a_idx)
                fb_hindsight = FeedbackItem(
                    item_id=t_id,
                    timestamp=time.strftime("%H:%M:%S"),
                    source="hindsight_reveal",
                    model_id=self.opponent_id,
                    prompt_text=entry["prompt_text"],
                    token_ids=entry["token_ids"],
                    legal_actions=entry["legal_actions"],
                    chosen_action=int(chosen_act),
                    preferred_action=int(best_a_idx),
                    weight=min(3.0, 1.0 + 0.5 * gain),
                    hand_reward=best_val,
                )
                self.rlhf_buffer.add_item(fb_hindsight)

        # 3. Update Leaderboard hand stats
        lb = LeaderboardManager()
        lb.register_player("human_player", "You (Human)", "Human Mind (RLHF Teacher)", "-", 1500.0)
        if self.opponent_id in lb.players:
            lb.record_hand("human_player", self.opponent_id, pts0, pts1)

        if self.match_scores[0] >= 15 or self.match_scores[1] >= 15:
            self.match_over = True
            self.match_winner = 0 if self.match_scores[0] >= 15 else 1
            if self.opponent_id in lb.players:
                lb.record_match("human_player", self.opponent_id, winner_is_a=(self.match_winner == 0))
        lb.save()

    def submit_turn_feedback(
        self,
        turn_id: str,
        feedback_type: str,  # "thumbs_up" | "override"
        preferred_action_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Records explicit human feedback on a specific AI turn (`turn_id`) into `RLHFBuffer`.
        """
        for entry in self.hand_log:
            if entry["turn_id"] == turn_id and entry["actor"] == 1:
                chosen_idx = int(entry["action_idx"])
                pref_idx = int(preferred_action_idx) if preferred_action_idx is not None else chosen_idx
                entry["feedback_status"] = feedback_type
                entry["preferred_action_idx"] = pref_idx

                fb = FeedbackItem(
                    item_id=turn_id,
                    timestamp=time.strftime("%H:%M:%S"),
                    source=feedback_type,
                    model_id=self.opponent_id,
                    prompt_text=entry["prompt_text"],
                    token_ids=entry["token_ids"],
                    legal_actions=entry["legal_actions"],
                    chosen_action=chosen_idx,
                    preferred_action=pref_idx,
                    weight=2.5 if feedback_type == "override" else 1.5,
                    hand_reward=2.0 if feedback_type == "override" else 1.0,
                )
                self.rlhf_buffer.add_item(fb)
                break

        return self.serialize_session()

    def _speech_bubble_for(self, act: Action) -> Optional[str]:
        if act in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2):
            return None
        if act == Action.QUIERO and self.state is not None and self.state.pending == PendingState.ENVIDO:
            mano_p = self.state.mano
            pie_p = 1 - mano_p
            m_score = self.state.envido_scores[mano_p]
            p_score = self.state.envido_scores[pie_p]
            m_lbl = "You" if mano_p == 0 else "AI"
            p_lbl = "You" if pie_p == 0 else "AI"
            if p_score > m_score:
                return f"¡QUIERO! ({m_lbl}: {m_score} → {p_lbl}: ¡{p_score} son mejores!)"
            return f"¡QUIERO! ({m_lbl}: {m_score} → {p_lbl}: 'Son buenas')"
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
        return bubble_map.get(act, ACTION_NAMES[act])

    def serialize_session(self) -> Dict[str, Any]:
        st = self.state
        if st is None:
            return {"active": False, "rlhf_count": len(self.rlhf_buffer.items)}

        reveal_opp = st.is_terminal
        live_ai_belief = compute_opponent_belief(st, 1)
        live_human_belief = compute_opponent_belief(st, 0)
        legal_human = []
        if not st.is_terminal and st.current_player == 0:
            for a in st.legal_actions():
                legal_human.append({
                    "action_idx": int(a),
                    "label": format_action_label(st, 0, a),
                    "is_card": a in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2),
                    "slot": int(a) if a in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2) else None,
                })

        tricks_serialized = []
        for t_idx, trick in enumerate(st.tricks):
            winner = st.trick_winners[t_idx] if t_idx < len(st.trick_winners) else None
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

        is_masked = bool(self.hide_identity and not self.match_over)
        revealed_at_end = bool(self.hide_identity and self.match_over)

        masked_hand_log = []
        for entry in self.hand_log:
            e_copy = dict(entry)
            if e_copy.get("actor") == 1:
                e_copy["actor_name"] = "🕵️ Mystery Opponent" if is_masked else self.opponent_name
            masked_hand_log.append(e_copy)

        opp_envido_speech: Optional[str] = None
        if st.envido_resolved and st.envido_revealed_scores[st.mano] is not None:
            if st.mano == 0:
                # Usuario es Mano (cantó primero), Máquina es Pie
                if st.envido_son_buenas:
                    opp_envido_speech = "Son buenas."
                elif st.envido_revealed_scores[1] is not None:
                    opp_envido_speech = f"¡{st.envido_revealed_scores[1]} son mejores!"
            else:
                # Máquina es Mano (cantó primero)
                opp_envido_speech = f"¡Tengo {st.envido_revealed_scores[1]}!"

        # Keep AI's unplayed cards hidden if AI is Pie and said "Son buenas" (never sang its Envido points)
        show_opp_unplayed = bool(
            st.is_terminal
            and not (st.mano == 0 and st.envido_son_buenas)
            and st.envido_revealed_scores[1] is not None
        )

        return {
            "active": True,
            "hide_identity": self.hide_identity,
            "opponent": {
                "id": "mystery" if is_masked else self.opponent_id,
                "name": "🕵️ Mystery Opponent" if is_masked else self.opponent_name,
                "elo": "????" if is_masked else self.opponent_elo,
                "model_type": "Identity Hidden Until Match Ends" if is_masked else self.opponent_model_type,
                "ckpt": "hidden" if is_masked else self.opponent_ckpt,
                "is_masked": is_masked,
                "revealed": revealed_at_end,
            },
            "hand_number": self.hand_number,
            "mano": st.mano,
            "match_scores": list(self.match_scores),
            "match_over": self.match_over,
            "match_winner": self.match_winner,
            "is_terminal": st.is_terminal,
            "current_player": st.current_player,
            "truco_stakes": st.truco_stakes,
            "truco_level": st.truco_level,
            "pending_state": st.pending.name,
            "points_won_hand": list(st.points_won),
            "envido_resolved": st.envido_resolved,
            "envido_winner": st.envido_winner,
            "envido_awarded_points": st.envido_awarded_points,
            "human_envido": st.envido_scores[0],
            "human_son_buenas": bool(st.envido_son_buenas and st.mano == 1),
            "opp_envido": st.envido_revealed_scores[1],
            "opp_son_buenas": bool(st.envido_son_buenas and st.mano == 0),
            "opp_envido_speech": opp_envido_speech,
            "live_ai_belief": live_ai_belief,
            "live_human_belief": live_human_belief,
            "hands": [
                [
                    {**serialize_card(c), "played": st.played_mask[0][s_idx]}
                    for s_idx, c in enumerate(st.hands[0])
                ],
                [
                    {
                        **serialize_card(c),
                        "played": st.played_mask[1][s_idx],
                        "hidden": not (show_opp_unplayed or st.played_mask[1][s_idx]),
                    }
                    for s_idx, c in enumerate(st.hands[1])
                ],
            ],
            "tricks": tricks_serialized,
            "legal_human_actions": legal_human,
            "hand_log": masked_hand_log,
            "rlhf_stats": {
                "total_samples": len(self.rlhf_buffer.items),
                "overrides": sum(1 for x in self.rlhf_buffer.items if x.source == "override"),
                "hindsight_reveals": sum(1 for x in self.rlhf_buffer.items if x.source == "hindsight_reveal"),
                "thumbs_up": sum(1 for x in self.rlhf_buffer.items if x.source == "thumbs_up"),
                "human_demos": sum(1 for x in self.rlhf_buffer.items if x.source == "human_play"),
            },
        }


def train_model_from_rlhf(
    session: InteractivePlaySession,
    epochs: int = 8,
    lr: float = 6e-4,
) -> Dict[str, Any]:
    """
    Fine-tunes the selected opponent model on all collected Human Feedback items:
    - Direct Human Overrides (`override`): High-weight cross-entropy toward `preferred_action`
      + unlikelihood penalty on the AI's blundered `chosen_action`.
    - Thumbs Up (`thumbs_up`): Reinforces `chosen_action`.
    - Winning Human Play Demos (`human_play`): Imitation learning on human moves that won points.
    """
    buffer = session.rlhf_buffer
    buffer.load()
    if len(buffer.items) == 0:
        return {"error": "No human feedback samples collected yet. Play a hand or critique an AI move first!"}

    device = session.device
    tokenizer = session.tokenizer

    model = System1TrucoSLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=384,
        dropout=0.0,
    )
    ckpt_path = session.opponent_ckpt if os.path.exists(session.opponent_ckpt) else DEFAULT_CHECKPOINT
    if os.path.exists(ckpt_path) and ckpt_path.endswith(".pt"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

    model.to(device)
    ref_model = System1TrucoSLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=384,
        dropout=0.0,
    ).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    items = list(buffer.items)
    B = len(items)
    max_len = max(len(x.token_ids) for x in items)

    inp = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
    act_pos = torch.zeros(B, dtype=torch.long, device=device)
    masks = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=device)
    pref_actions = torch.zeros(B, dtype=torch.long, device=device)
    bad_actions = torch.zeros(B, dtype=torch.long, device=device)
    is_override = torch.zeros(B, dtype=torch.bool, device=device)
    weights = torch.zeros(B, dtype=torch.float32, device=device)
    target_vals = torch.zeros(B, dtype=torch.float32, device=device)

    for i, it in enumerate(items):
        L = len(it.token_ids)
        inp[i, :L] = torch.tensor(it.token_ids, dtype=torch.long, device=device)
        act_pos[i] = L - 1
        for a_idx in it.legal_actions:
            masks[i, a_idx] = True
        pref_actions[i] = it.preferred_action
        bad_actions[i] = it.chosen_action
        is_override[i] = (it.source in ("override", "hindsight_reveal")) and (it.preferred_action != it.chosen_action)
        weights[i] = it.weight
        target_vals[i] = it.hand_reward

    model.train()
    initial_loss = 0.0
    final_loss = 0.0

    for ep in range(1, epochs + 1):
        logits, pred_v = model(inp, act_pos, masks)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        with torch.no_grad():
            ref_logits, _ = ref_model(inp, act_pos, masks)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # 1. Weighted Cross-Entropy toward human-preferred action
        chosen_lp = log_probs.gather(1, pref_actions.unsqueeze(1)).squeeze(1)
        ce_loss = -(weights * chosen_lp).mean()

        # 2. Direct Preference / Unlikelihood penalty on overridden AI blunder
        if is_override.any():
            bad_probs = probs.gather(1, bad_actions.unsqueeze(1)).squeeze(1)
            unlikelihood = -torch.log(torch.clamp(1.0 - bad_probs[is_override], min=1e-4)).mean()
        else:
            unlikelihood = torch.tensor(0.0, device=device)

        # 3. KL anchor to reference policy so model retains general play strength on other states
        kl_div = (probs * (log_probs.masked_fill(~masks, 0.0) - ref_log_probs.masked_fill(~masks, 0.0))).sum(dim=-1).mean()
        v_loss = F.smooth_l1_loss(pred_v, target_vals)

        loss = ce_loss + 0.5 * unlikelihood + 0.15 * v_loss + 0.25 * kl_div
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if ep == 1:
            initial_loss = float(loss.item())
        final_loss = float(loss.item())

    model.eval()
    session.opponent_model = model

    # Save updated weights to the selected checkpoint AND `jev_truco_slm.pt`
    save_target = ckpt_path if ckpt_path.endswith(".pt") else DEFAULT_CHECKPOINT
    torch.save({"model_state_dict": model.state_dict(), "rlhf_samples": B}, save_target)
    torch.save({"model_state_dict": model.state_dict(), "rlhf_samples": B}, DEFAULT_CHECKPOINT)

    return {
        "status": "ok",
        "trained_on_samples": B,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "updated_checkpoint": save_target,
    }
