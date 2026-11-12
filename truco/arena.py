"""
Dual-Instance Self-Play & League Evaluation Arena for System 1 Truco SLMs.

Maintains multiple separate instances of `System1TrucoSLM`:
1. `base_model` (Frozen Base Yardstick): Never updated once created; used to measure
   how much the active learner has improved over training generations.
2. `learner_model` (Active Policy/Value Network): Trains by playing hands against
   separate opponent instances (`opponent_model` and historical snapshots).
3. `opponent_model` (Sparring Instance): A distinct `System1TrucoSLM` instance that
   is periodically updated when `learner_model` significantly outperforms it.

Uses duplicate dealing (every dealt 6-card hand is played twice with seat-swapped
Mano/Pie positions) so card-luck variance cancels out during evaluation against `base_model`.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from truco.engine import Action, Card, DECK, NUM_ACTIONS, TrucoHandState
from truco.model import System1TrucoSLM
from truco.tokenizer import TrucoTokenizer


@dataclass
class HeadToHeadResult:
    hands_played: int
    net_pts_per_hand: float
    hand_win_rate: float
    match_win_rate: float
    elo_delta: float


@torch.no_grad()
def evaluate_head_to_head(
    model_a: System1TrucoSLM,
    model_b: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    num_duplicate_pairs: int = 250,
    num_matches: int = 60,
    seed: int = 12345,
) -> HeadToHeadResult:
    """
    Evaluates `model_a` vs `model_b` using duplicate hands (same cards dealt with seats swapped)
    plus full 15-point matches.
    """
    model_a.eval()
    model_b.eval()
    rng = random.Random(seed)

    total_net_pts_a = 0.0
    wins_a = 0
    ties = 0
    total_hands = num_duplicate_pairs * 2

    for _ in range(num_duplicate_pairs):
        sampled = rng.sample(DECK, 6)
        h0, h1 = sampled[:3], sampled[3:]
        s0, s1 = rng.randint(0, 12), rng.randint(0, 12)

        # Direction 1: model_a is Player 0 (Mano), model_b is Player 1 (Pie)
        state1 = TrucoHandState.new_hand(mano=0, match_scores=(s0, s1), dealt_cards=(h0, h1))
        while not state1.is_terminal:
            if state1.current_player == 0:
                act, _, _ = model_a.act(state1, tokenizer, device, temperature=0.0)
            else:
                act, _, _ = model_b.act(state1, tokenizer, device, temperature=0.0)
            state1.step(act)
        diff1 = state1.payoff(0)

        # Direction 2: model_b is Player 0 (Mano), model_a is Player 1 (Pie) with exact same cards
        state2 = TrucoHandState.new_hand(mano=0, match_scores=(s0, s1), dealt_cards=(h0, h1))
        while not state2.is_terminal:
            if state2.current_player == 0:
                act, _, _ = model_b.act(state2, tokenizer, device, temperature=0.0)
            else:
                act, _, _ = model_a.act(state2, tokenizer, device, temperature=0.0)
            state2.step(act)
        diff2 = state2.payoff(1)

        pair_net = diff1 + diff2
        total_net_pts_a += pair_net
        for d in (diff1, diff2):
            if d > 0:
                wins_a += 1
            elif d == 0:
                ties += 1

    # Also play full 15-point matches
    match_wins_a = 0
    for m in range(num_matches):
        scores = [0, 0]
        mano = m % 2
        while scores[0] < 15 and scores[1] < 15:
            st = TrucoHandState.new_hand(mano=mano, match_scores=(scores[0], scores[1]), rng=rng)
            while not st.is_terminal:
                if st.current_player == 0:
                    act, _, _ = model_a.act(st, tokenizer, device, temperature=0.05)
                else:
                    act, _, _ = model_b.act(st, tokenizer, device, temperature=0.05)
                st.step(act)
            scores[0] += st.points_won[0]
            scores[1] += st.points_won[1]
            mano = 1 - mano
        if scores[0] >= 15:
            match_wins_a += 1

    hand_wr = (wins_a + 0.5 * ties) / total_hands
    match_wr = match_wins_a / max(1, num_matches)
    clipped_wr = min(0.99, max(0.01, match_wr))
    elo_delta = 400.0 * float(np.log10(clipped_wr / (1.0 - clipped_wr)))

    return HeadToHeadResult(
        hands_played=total_hands,
        net_pts_per_hand=total_net_pts_a / total_hands,
        hand_win_rate=hand_wr,
        match_win_rate=match_wr,
        elo_delta=elo_delta,
    )


@dataclass
class RolloutStep:
    token_ids: List[int]
    legal_mask: np.ndarray
    action_idx: int
    old_log_prob: float
    old_value: float
    reward: float


def run_dual_instance_training(
    base_model: System1TrucoSLM,
    learner_model: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    generations: int = 8,
    hands_per_gen: int = 500,
    ppo_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 5e-4,
    checkpoint_dir: str = "checkpoints",
) -> List[Dict[str, float]]:
    """
    Trains `learner_model` by playing against a separate `opponent_model` instance (and a pool
    of past snapshots), evaluating against the frozen `base_model` after every generation.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    learner_model.to(device)

    # Create a separate second instance (`sparring_model`) initialized from `learner_model`
    sparring_model = copy.deepcopy(learner_model).to(device)
    sparring_model.eval()
    for p in sparring_model.parameters():
        p.requires_grad_(False)

    # Detect highest existing generation checkpoint so repeated runs accumulate (Gen 6, 7, 8...)
    existing_gens: List[int] = []
    for fn in os.listdir(checkpoint_dir):
        if fn.startswith("jev_gen_") and fn.endswith("_slm.pt"):
            try:
                existing_gens.append(int(fn.split("_")[2]))
            except ValueError:
                pass
    start_gen = max(existing_gens) if existing_gens else 0

    # Keep a historical snapshot pool (including base_model and past saved generation snapshots)
    snapshot_pool: List[System1TrucoSLM] = [base_model]
    for g_num in sorted(existing_gens)[-4:]:
        g_path = os.path.join(checkpoint_dir, f"jev_gen_{g_num}_slm.pt")
        try:
            snap = copy.deepcopy(learner_model).to(device)
            snap_ckpt = torch.load(g_path, map_location=device, weights_only=False)
            snap.load_state_dict(snap_ckpt["model_state_dict"])
            snap.eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            snapshot_pool.append(snap)
        except Exception:
            pass
    snapshot_pool.append(copy.deepcopy(sparring_model))

    optimizer = torch.optim.AdamW(learner_model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(2026 + start_gen * 100)
    history_log: List[Dict[str, float]] = []

    # Save frozen base model if not already saved
    base_ckpt_path = os.path.join(checkpoint_dir, "jev_base_slm.pt")
    if not os.path.exists(base_ckpt_path):
        torch.save({"model_state_dict": base_model.state_dict()}, base_ckpt_path)

    # Current baseline evaluation before this run's new generations
    init_eval = evaluate_head_to_head(learner_model, base_model, tokenizer, device)
    print(
        f"[Gen {start_gen} / Resume State] vs Base Model -> Net Pts/Hand: {init_eval.net_pts_per_hand:+.3f} | "
        f"Hand Win: {init_eval.hand_win_rate*100:.1f}% | "
        f"15-pt Match Win: {init_eval.match_win_rate*100:.1f}% (Elo: {init_eval.elo_delta:+.0f})"
    )
    history_log.append({
        "generation": float(start_gen),
        "net_pts_vs_base": init_eval.net_pts_per_hand,
        "hand_wr_vs_base": init_eval.hand_win_rate,
        "match_wr_vs_base": init_eval.match_win_rate,
        "elo_vs_base": init_eval.elo_delta,
    })

    end_gen = start_gen + generations
    for gen in range(start_gen + 1, end_gen + 1):
        learner_model.eval()
        rollouts: List[RolloutStep] = []

        # Collect duplicate pairs of hands where Learner (Instance 1) plays against Opponent (Instance 2).
        # Playing every dealt 6-card hand twice (once as Mano, once as Pie) and using the difference
        # in payoff (`r_seat0 - r_seat1`) cancels out raw card-luck variance!
        num_pairs = max(1, hands_per_gen // 2)
        for pair_idx in range(num_pairs):
            if rng.random() < 0.70:
                opp_instance = sparring_model
            else:
                opp_instance = rng.choice(snapshot_pool)

            sampled = rng.sample(DECK, 6)
            h0, h1 = sampled[:3], sampled[3:]
            s0, s1 = rng.randint(0, 13), rng.randint(0, 13)

            pair_trajs: List[List[Tuple[List[int], np.ndarray, int, float, float]]] = [[], []]
            pair_payoffs: List[float] = [0.0, 0.0]

            for seat in (0, 1):
                state = TrucoHandState.new_hand(mano=0, match_scores=(s0, s1), dealt_cards=(h0, h1))
                while not state.is_terminal:
                    p = state.current_player
                    if p == seat:
                        toks = tokenizer.encode_observation(state, p)
                        if len(toks) > tokenizer.max_seq_len:
                            toks = [toks[0]] + toks[-(tokenizer.max_seq_len - 1):]
                        legal = state.legal_actions()
                        mask = np.zeros(NUM_ACTIONS, dtype=bool)
                        for a in legal:
                            mask[int(a)] = True

                        with torch.no_grad():
                            inp = torch.tensor([toks], dtype=torch.long, device=device)
                            act_pos = torch.tensor([len(toks) - 1], dtype=torch.long, device=device)
                            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                            logits, val_t = learner_model(inp, act_pos, mask_t)
                            probs = F.softmax(logits[0] / 0.7, dim=-1)
                            log_probs = F.log_softmax(logits[0], dim=-1)
                            act_idx = int(torch.multinomial(probs, 1).item())
                            old_lp = float(log_probs[act_idx].item())
                            old_v = float(val_t[0].item())

                        pair_trajs[seat].append((toks, mask, act_idx, old_lp, old_v))
                        state.step(Action(act_idx))
                    else:
                        opp_act, _, _ = opp_instance.act(state, tokenizer, device, temperature=0.1)
                        state.step(opp_act)

                pair_payoffs[seat] = state.payoff(seat)

            # Variance-reduced duplicate return: compare Learner's payoff in Seat 0 against
            # Opponent's payoff in Seat 0 with the exact same cards!
            # (Note: when Learner is Seat 1, Opponent was Seat 0, so Opponent's Seat 0 payoff is -pair_payoffs[1])
            vr_return_0 = 0.5 * (pair_payoffs[0] - (-pair_payoffs[1]))
            vr_return_1 = 0.5 * (pair_payoffs[1] - (-pair_payoffs[0]))

            for toks, mask, act_idx, old_lp, old_v in pair_trajs[0]:
                rollouts.append(
                    RolloutStep(
                        token_ids=toks,
                        legal_mask=mask,
                        action_idx=act_idx,
                        old_log_prob=old_lp,
                        old_value=old_v,
                        reward=vr_return_0,
                    )
                )
            for toks, mask, act_idx, old_lp, old_v in pair_trajs[1]:
                rollouts.append(
                    RolloutStep(
                        token_ids=toks,
                        legal_mask=mask,
                        action_idx=act_idx,
                        old_log_prob=old_lp,
                        old_value=old_v,
                        reward=vr_return_1,
                    )
                )

        # --- PPO / Actor-Critic Update on Learner Instance (with Sparring KL Anchor) ---
        learner_model.train()
        for _ in range(ppo_epochs):
            rng.shuffle(rollouts)
            for start in range(0, len(rollouts), batch_size):
                batch = rollouts[start : start + batch_size]
                if len(batch) < 8:
                    continue
                B = len(batch)
                max_len = max(len(x.token_ids) for x in batch)

                inp = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
                act_pos = torch.zeros(B, dtype=torch.long, device=device)
                masks = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=device)
                actions = torch.zeros(B, dtype=torch.long, device=device)
                old_lps = torch.zeros(B, dtype=torch.float32, device=device)
                returns = torch.zeros(B, dtype=torch.float32, device=device)
                advs = torch.zeros(B, dtype=torch.float32, device=device)

                for i, step in enumerate(batch):
                    L = len(step.token_ids)
                    inp[i, :L] = torch.tensor(step.token_ids, dtype=torch.long, device=device)
                    act_pos[i] = L - 1
                    masks[i] = torch.from_numpy(step.legal_mask).to(device)
                    actions[i] = step.action_idx
                    old_lps[i] = step.old_log_prob
                    returns[i] = step.reward
                    advs[i] = step.reward - 0.5 * step.old_value

                advs = (advs - advs.mean()) / (advs.std() + 1e-6)

                logits, values = learner_model(inp, act_pos, masks)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)

                with torch.no_grad():
                    ref_logits, _ = sparring_model(inp, act_pos, masks)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                new_lps = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(new_lps - old_lps)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 0.85, 1.15) * advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL divergence penalty to prevent catastrophic forgetting of CFR prior
                kl_div = (probs * (log_probs.masked_fill(~masks, 0.0) - ref_log_probs.masked_fill(~masks, 0.0))).sum(dim=-1).mean()
                value_loss = F.smooth_l1_loss(values, returns)

                loss = policy_loss + 0.4 * value_loss + 0.08 * kl_div
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(learner_model.parameters(), 0.5)
                optimizer.step()

        # --- Evaluate Learner vs Sparring Instance (Instance 1 vs Instance 2) ---
        vs_sparring = evaluate_head_to_head(
            learner_model, sparring_model, tokenizer, device, num_duplicate_pairs=150, num_matches=30, seed=100 + gen
        )
        promoted = False
        if vs_sparring.net_pts_per_hand > 0.05:
            # Learner beats current sparring instance -> promote learner weights to sparring_model!
            sparring_model.load_state_dict(copy.deepcopy(learner_model.state_dict()))
            snapshot_pool.append(copy.deepcopy(sparring_model))
            if len(snapshot_pool) > 6:
                # Always keep index 0 (base_model) and most recent snapshots
                snapshot_pool.pop(1)
            promoted = True

        # --- Measure Learner vs Frozen Base Model ---
        vs_base = evaluate_head_to_head(
            learner_model, base_model, tokenizer, device, num_duplicate_pairs=250, num_matches=60, seed=12345
        )
        promo_tag = " [PROMOTED NEW SPARRING OPPONENT]" if promoted else ""
        print(
            f"[Gen {gen}/{end_gen}] vs Base Model -> Net Pts/Hand: {vs_base.net_pts_per_hand:+.3f} | "
            f"Hand Win: {vs_base.hand_win_rate*100:.1f}% | "
            f"15-pt Match Win: {vs_base.match_win_rate*100:.1f}% (Elo: {vs_base.elo_delta:+.0f})"
            f" | vs Sparring: {vs_sparring.net_pts_per_hand:+.3f} pts/h{promo_tag}"
        )

        history_log.append({
            "generation": float(gen),
            "net_pts_vs_base": vs_base.net_pts_per_hand,
            "hand_wr_vs_base": vs_base.hand_win_rate,
            "match_wr_vs_base": vs_base.match_win_rate,
            "elo_vs_base": vs_base.elo_delta,
        })

        # Save latest learner checkpoint
        gen_ckpt_path = os.path.join(checkpoint_dir, f"jev_gen_{gen}_slm.pt")
        torch.save({"model_state_dict": learner_model.state_dict(), "gen": gen}, gen_ckpt_path)

    # Also update main checkpoint if final learner beats base
    final_ckpt_path = os.path.join(checkpoint_dir, "jev_truco_slm.pt")
    torch.save(
        {
            "model_state_dict": learner_model.state_dict(),
            "history_log": history_log,
        },
        final_ckpt_path,
    )
    return history_log


@dataclass
class CounterfactualSample:
    token_ids: List[int]
    legal_mask: np.ndarray
    target_policy: np.ndarray
    target_value: float


@torch.no_grad()
def _rollout_to_end(
    state: TrucoHandState,
    challenger_seat: int,
    challenger: System1TrucoSLM,
    leader: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    opp_persona: str = "standard",
) -> float:
    """Fast greedy rollout to the end of the hand from `state`, returning challenger's net payoff."""
    from truco.leaderboard import act_with_personality

    sim = state.clone()
    while not sim.is_terminal:
        if sim.current_player == challenger_seat:
            act, _, _ = challenger.act(sim, tokenizer, device, temperature=0.05)
        else:
            if opp_persona == "standard":
                act, _, _ = leader.act(sim, tokenizer, device, temperature=0.05)
            else:
                act = act_with_personality(leader, sim, tokenizer, device, persona=opp_persona, temperature=0.05)
        sim.step(act)
    return sim.payoff(challenger_seat)


def train_against_leader(
    challenger_model: System1TrucoSLM,
    leader_model: System1TrucoSLM,
    leader_name: str,
    base_model: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    extra_opponents: Optional[List[System1TrucoSLM]] = None,
    rounds: int = 1200,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 5e-4,
    checkpoint_dir: str = "checkpoints",
    telemetry_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> HeadToHeadResult:
    """
    Trains `challenger_model` against `leader_model` (#1 ranked model on the Leaderboard)
    and a Gauntlet of top runners-up + Tactical Personality Specialists (`bluffer`, `phisher`,
    `sheriff`, `risk_taker`) using Counterfactual Multi-Action Rollouts + Leader Policy Prior.

    Why this works without overfitting to a single bot:
    1. Initializes/anchors to the #1 Leader's policy (`leader_probs`).
    2. At each state, rolls out every legal action against a gauntlet of top leaders and tactical
       personas (`El Mentiroso`, `El Zorro`, `El Comisario`) to measure true counterfactual advantage `A(s, a) = Q(s, a) - V(s)`.
    3. Updates the policy using Advantage-Weighted Policy Improvement (`pi_new(a|s) ~ pi_leader(a|s) * exp(A(s,a) / tau)`),
       which is the exact closed-form solution to KL-constrained reward maximization (AWAC / MPO / Nash-MD).
    """
    from truco.leaderboard import act_with_personality

    challenger_model.to(device)
    leader_model.to(device).eval()
    base_model.to(device).eval()

    for p in leader_model.parameters():
        p.requires_grad_(False)
    for p in base_model.parameters():
        p.requires_grad_(False)

    gauntlet: List[System1TrucoSLM] = [leader_model, leader_model, base_model]
    if extra_opponents:
        for opp in extra_opponents:
            opp.to(device).eval()
            for p in opp.parameters():
                p.requires_grad_(False)
            gauntlet.append(opp)

    persona_pool = ["standard", "standard", "bluffer", "phisher", "sheriff", "risk_taker"]
    rng = random.Random(7777)
    samples: List[CounterfactualSample] = []

    print(f"  [1/3] Collecting Advantage-Weighted Counterfactual Rollouts vs Top-Leader & Tactical Persona Gauntlet ({rounds} hands)...")
    match_scores = [0, 0]
    t_game_start = time.perf_counter()
    game_dec_times: List[float] = []
    hands_in_curr_game = 0

    for h in range(rounds):
        challenger_seat = h % 2
        mano = (h // 2) % 2
        opp_model = rng.choice(gauntlet)
        opp_persona = rng.choice(persona_pool)

        s0 = match_scores[0] if challenger_seat == 0 else match_scores[1]
        s1 = match_scores[1] if challenger_seat == 0 else match_scores[0]

        state = TrucoHandState.new_hand(mano=mano, match_scores=(s0, s1), rng=rng)
        while not state.is_terminal:
            p = state.current_player
            if p == challenger_seat:
                t_dec = time.perf_counter()
                legal = state.legal_actions()
                toks = tokenizer.encode_observation(state, p)
                if len(toks) > tokenizer.max_seq_len:
                    toks = [toks[0]] + toks[-(tokenizer.max_seq_len - 1):]

                mask = np.zeros(NUM_ACTIONS, dtype=bool)
                for a in legal:
                    mask[int(a)] = True

                # Query Leader's prior policy at this state (System-1 single-pass forward latency)
                with torch.no_grad():
                    inp_t = torch.tensor([toks], dtype=torch.long, device=device)
                    act_pos_t = torch.tensor([len(toks) - 1], dtype=torch.long, device=device)
                    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
                    leader_logits, _ = leader_model(inp_t, act_pos_t, mask_t)
                    leader_probs = F.softmax(leader_logits[0], dim=-1).cpu().numpy()
                game_dec_times.append((time.perf_counter() - t_dec) * 1000.0)

                # Evaluate every legal action via 2 counterfactual rollouts (vs Leader & vs Gauntlet Persona opp)
                q_vals = np.zeros(len(legal), dtype=np.float64)
                for idx_a, a in enumerate(legal):
                    next_st = state.clone().step(a)
                    r1 = _rollout_to_end(next_st, challenger_seat, challenger_model, leader_model, tokenizer, device, opp_persona="standard")
                    r2 = _rollout_to_end(next_st, challenger_seat, challenger_model, opp_model, tokenizer, device, opp_persona=opp_persona)
                    q_vals[idx_a] = 0.5 * (r1 + r2)

                # Expected value under Leader policy & counterfactual advantages
                prior_legal = np.array([max(1e-4, leader_probs[int(a)]) for a in legal], dtype=np.float64)
                prior_legal /= np.sum(prior_legal)
                v_state = float(np.sum(prior_legal * q_vals))
                advantages = q_vals - v_state

                # Closed-form KL-regularized policy improvement: pi*(a) \propto pi_leader(a) * exp(adv(a) / tau)
                adv_scaled = np.clip(advantages / 1.0, -4.0, 4.0)
                unnorm_pi = prior_legal * np.exp(adv_scaled)
                unnorm_pi /= np.sum(unnorm_pi)

                target_pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for idx_a, a in enumerate(legal):
                    target_pi[int(a)] = float(unnorm_pi[idx_a])

                best_v = float(np.sum(unnorm_pi * q_vals))
                samples.append(
                    CounterfactualSample(
                        token_ids=toks,
                        legal_mask=mask,
                        target_policy=target_pi,
                        target_value=best_v,
                    )
                )

                # Advance trajectory
                if rng.random() < 0.15:
                    chosen = rng.choice(legal)
                else:
                    chosen = rng.choices(legal, weights=unnorm_pi.tolist(), k=1)[0]
                state.step(chosen)
            else:
                t_dec = time.perf_counter()
                opp_act = act_with_personality(opp_model, state, tokenizer, device, persona=opp_persona, temperature=0.05)
                game_dec_times.append((time.perf_counter() - t_dec) * 1000.0)
                state.step(opp_act)

        # Accumulate points in the ongoing 15-pt training game
        pts_chal = state.points_won[challenger_seat]
        pts_opp = state.points_won[1 - challenger_seat]
        match_scores[0] += pts_chal
        match_scores[1] += pts_opp
        hands_in_curr_game += 1

        if match_scores[0] >= 15 or match_scores[1] >= 15 or (h == rounds - 1 and hands_in_curr_game > 0):
            game_ms = (time.perf_counter() - t_game_start) * 1000.0
            chal_won = match_scores[0] >= match_scores[1]
            if telemetry_cb is not None:
                avg_dec = sum(game_dec_times) / max(1, len(game_dec_times))
                telemetry_cb({
                    "winner_name": "Challenger" if chal_won else leader_name.split(" (")[0],
                    "loser_name": leader_name.split(" (")[0] if chal_won else "Challenger",
                    "winner_pts": int(max(match_scores[0], match_scores[1])),
                    "loser_pts": int(min(match_scores[0], match_scores[1])),
                    "game_ms": round(game_ms, 2),
                    "avg_decision_ms": round(avg_dec, 3),
                    "decisions": len(game_dec_times),
                    "hands": hands_in_curr_game,
                    "hands_done_total": h + 1,
                    "hands_target_total": rounds,
                    "progress_fraction": (h + 1) / max(1, rounds),
                })
            match_scores = [0, 0]
            t_game_start = time.perf_counter()
            game_dec_times = []
            hands_in_curr_game = 0

    print(f"  [2/3] Collected {len(samples):,} states. Updating System 1 Policy & Value Heads...")
    optimizer = torch.optim.AdamW(challenger_model.parameters(), lr=lr, weight_decay=1e-4)
    challenger_model.train()

    for ep in range(1, epochs + 1):
        rng.shuffle(samples)
        tot_p_loss = 0.0
        tot_v_loss = 0.0
        n_batches = 0

        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            B = len(batch)
            max_len = max(len(x.token_ids) for x in batch)

            inp = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
            act_pos = torch.zeros(B, dtype=torch.long, device=device)
            masks = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=device)
            target_pis = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
            target_vs = torch.zeros(B, dtype=torch.float32, device=device)

            for i, s in enumerate(batch):
                L = len(s.token_ids)
                inp[i, :L] = torch.tensor(s.token_ids, dtype=torch.long, device=device)
                act_pos[i] = L - 1
                masks[i] = torch.from_numpy(s.legal_mask).to(device)
                target_pis[i] = torch.from_numpy(s.target_policy).to(device)
                target_vs[i] = s.target_value

            logits, pred_v = challenger_model(inp, act_pos, masks)
            log_probs = F.log_softmax(logits, dim=-1)

            p_loss = -(target_pis * log_probs).sum(dim=-1).mean()
            v_loss = F.smooth_l1_loss(pred_v, target_vs)
            loss = p_loss + 0.25 * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(challenger_model.parameters(), 0.5)
            optimizer.step()

            tot_p_loss += float(p_loss.item())
            tot_v_loss += float(v_loss.item())
            n_batches += 1

        print(
            f"        Epoch {ep}/{epochs} | Policy CE: {tot_p_loss/max(1, n_batches):.4f} | "
            f"Value Loss: {tot_v_loss/max(1, n_batches):.4f}"
        )

    print(f"  [3/3] Evaluating Challenger vs Leader '{leader_name}' (500 duplicate hands + 60 matches)...")
    h2h = evaluate_head_to_head(challenger_model, leader_model, tokenizer, device, num_duplicate_pairs=250, num_matches=60)
    print(
        f"        Result vs '{leader_name}' -> Net Pts/Hand: {h2h.net_pts_per_hand:+.3f} | "
        f"Hand Win: {h2h.hand_win_rate*100:.1f}% | "
        f"15-pt Match Win: {h2h.match_win_rate*100:.1f}% (Elo diff: {h2h.elo_delta:+.0f})"
    )
    return h2h


def run_championship_and_retrain_all(
    tokenizer: TrucoTokenizer,
    device: torch.device,
    championship_rounds: int = 3,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 5e-4,
    log_fn: Optional[Any] = None,
    telemetry_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    1. Runs a Round-Robin Championship across all Leaderboard bots while recording every
       turn and running End-of-Hand Full-Card Hindsight Counterfactual Replay on the played hands.
    2. Combines the harvested Championship Hindsight dataset with any collected Human RLHF
       feedback (`checkpoints/human_feedback.json`).
    3. Retrains ALL trainable neural SLM models (`jev_champion` and `jev_gen_1..N`, keeping
       `jev_base` frozen as our immutable Gen-0 reference yardstick) so every model learns
       the new Bayesian hidden-card belief tokens (`BELIEF_WIN_*`, `BELIEF_THREAT_*`, `BELIEF_ENV_*`)
       and fixes the mistakes exposed in the Championship.
    4. Runs a Post-Retraining Championship to measure the updated Leaderboard Elo rankings!
    """
    from truco.leaderboard import build_roster, LeaderboardManager, run_leaderboard_tournament

    def _emit(msg: str) -> None:
        print(msg)
        if log_fn is not None:
            log_fn(msg)

    lb = LeaderboardManager()
    roster = build_roster(lb, tokenizer, device)
    player_ids = [pid for pid in roster.keys() if pid != "human_player"]

    # Load #1 Leader neural model to guide fast hindsight rollouts
    ranked_pts = [
        p for p in lb.get_ranked_players()
        if p.checkpoint_path and os.path.exists(p.checkpoint_path) and p.checkpoint_path.endswith(".pt")
    ]
    rollout_policy = System1TrucoSLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=tokenizer.max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=384,
        dropout=0.0,
    ).to(device)
    if ranked_pts:
        ck = torch.load(ranked_pts[0].checkpoint_path, map_location=device, weights_only=False)
        rollout_policy.load_state_dict(ck["model_state_dict"])
    rollout_policy.eval()

    _emit(
        f"[Step 1/4] Playing Championship across {len(player_ids)} bots ({championship_rounds} duplicate match rounds/pair) "
        f"& harvesting Full-Card Hindsight rollouts..."
    )

    rng = random.Random(2026)
    harvested_samples: List[CounterfactualSample] = []
    total_hands_played = 0
    total_pairs = len(player_ids) * (len(player_ids) - 1) // 2
    total_step1_matches = max(1, total_pairs * championship_rounds * 2)
    step1_matches_done = 0

    for i in range(len(player_ids)):
        for j in range(i + 1, len(player_ids)):
            id_a, id_b = player_ids[i], player_ids[j]
            bot_a, bot_b = roster[id_a], roster[id_b]

            for r in range(championship_rounds):
                for first_mano in (0, 1):
                    t_game_start = time.perf_counter()
                    game_dec_times: List[float] = []
                    hands_in_game = 0
                    scores = [0, 0]
                    mano = first_mano
                    while scores[0] < 15 and scores[1] < 15:
                        state = TrucoHandState.new_hand(
                            mano=mano,
                            match_scores=(scores[0], scores[1]),
                            target_score=15,
                            rng=rng,
                        )
                        hand_snapshots: List[Tuple[TrucoHandState, int, List[int], List[Action], Action]] = []

                        while not state.is_terminal:
                            cur_p = state.current_player
                            legal = state.legal_actions()
                            toks = tokenizer.encode_observation(state, cur_p)
                            if len(toks) > tokenizer.max_seq_len:
                                toks = [toks[0]] + toks[-(tokenizer.max_seq_len - 1):]
                            t_dec = time.perf_counter()
                            act = bot_a.act(state) if cur_p == 0 else bot_b.act(state)
                            game_dec_times.append((time.perf_counter() - t_dec) * 1000.0)
                            if len(legal) > 1 and rng.random() < 0.25:
                                hand_snapshots.append((state.clone(), cur_p, toks, legal, act))
                            state.step(act)

                        pts0, pts1 = state.points_won[0], state.points_won[1]
                        scores[0] += pts0
                        scores[1] += pts1
                        total_hands_played += 1
                        hands_in_game += 1

                        # End-of-Hand Full-Card Hindsight Counterfactual Replay on sampled turns
                        for snap, seat, toks, legal, chosen_act in hand_snapshots:
                            mask = np.zeros(NUM_ACTIONS, dtype=bool)
                            for a in legal:
                                mask[int(a)] = True

                            q_vals = np.zeros(len(legal), dtype=np.float64)
                            for idx_a, a in enumerate(legal):
                                next_st = snap.clone().step(a)
                                q_vals[idx_a] = _rollout_to_end(
                                    next_st, seat, rollout_policy, rollout_policy, tokenizer, device
                                )

                            # Softmax advantage-weighted target policy over full-card hindsight Q-values
                            q_centered = q_vals - np.max(q_vals)
                            unnorm_pi = np.exp(np.clip(q_centered / 0.8, -5.0, 5.0))
                            unnorm_pi /= np.sum(unnorm_pi)

                            target_pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                            for idx_a, a in enumerate(legal):
                                target_pi[int(a)] = float(unnorm_pi[idx_a])

                            best_v = float(np.sum(unnorm_pi * q_vals))
                            harvested_samples.append(
                                CounterfactualSample(
                                    token_ids=toks,
                                    legal_mask=mask,
                                    target_policy=target_pi,
                                    target_value=best_v,
                                )
                            )

                        mano = 1 - mano

                    game_ms = (time.perf_counter() - t_game_start) * 1000.0
                    step1_matches_done += 1
                    if telemetry_cb is not None:
                        a_won = scores[0] >= scores[1]
                        avg_dec = sum(game_dec_times) / max(1, len(game_dec_times))
                        telemetry_cb({
                            "winner_name": bot_a.stats.display_name if a_won else bot_b.stats.display_name,
                            "loser_name": bot_b.stats.display_name if a_won else bot_a.stats.display_name,
                            "winner_pts": int(max(scores[0], scores[1])),
                            "loser_pts": int(min(scores[0], scores[1])),
                            "game_ms": round(game_ms, 2),
                            "avg_decision_ms": round(avg_dec, 3),
                            "decisions": len(game_dec_times),
                            "hands": hands_in_game,
                            "progress_fraction": step1_matches_done / total_step1_matches,
                        })

    # Also fold in any Human Feedback / Hindsight Reveal items from `checkpoints/human_feedback.json`
    rlhf_added = 0
    rlhf_path = "checkpoints/human_feedback.json"
    if os.path.exists(rlhf_path):
        try:
            with open(rlhf_path, "r", encoding="utf-8") as f:
                raw_fb = json.load(f).get("items", [])
            for item in raw_fb:
                mask = np.zeros(NUM_ACTIONS, dtype=bool)
                for a_idx in item.get("legal_actions", []):
                    mask[int(a_idx)] = True
                pref_idx = int(item.get("preferred_action", 0))
                if not mask[pref_idx]:
                    continue
                target_pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
                target_pi[pref_idx] = 1.0
                # Duplicate human/hindsight items 2x for higher priority
                for _ in range(2):
                    harvested_samples.append(
                        CounterfactualSample(
                            token_ids=item["token_ids"],
                            legal_mask=mask,
                            target_policy=target_pi,
                            target_value=float(item.get("hand_reward", 1.0)),
                        )
                    )
                rlhf_added += 1
        except Exception:
            pass

    _emit(
        f"[Step 2/4] Harvested {len(harvested_samples):,} hindsight decision states from {total_hands_played:,} "
        f"Championship hands (+ {rlhf_added} human/hindsight RLHF states)."
    )

    # Identify all trainable neural checkpoints (keep `jev_base_slm.pt` frozen as the Gen-0 baseline benchmark!)
    trainable_ckpts: List[Tuple[str, str]] = []
    seen_paths = set()
    for pid, wrapper in roster.items():
        ckpt_p = wrapper.stats.checkpoint_path
        if (
            pid != "jev_base"
            and ckpt_p
            and ckpt_p.endswith(".pt")
            and os.path.exists(ckpt_p)
            and ckpt_p not in seen_paths
        ):
            seen_paths.add(ckpt_p)
            trainable_ckpts.append((wrapper.stats.display_name, ckpt_p))

    _emit(f"[Step 3/4] Retraining all {len(trainable_ckpts)} neural models on Championship + Hindsight output...")

    retrained_summaries: List[Dict[str, Any]] = []
    for idx_m, (disp_name, ckpt_path) in enumerate(trainable_ckpts, start=1):
        model = System1TrucoSLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=tokenizer.max_seq_len,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=384,
            dropout=0.0,
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])

        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        model.train()

        init_loss = 0.0
        final_loss = 0.0
        for ep in range(1, epochs + 1):
            rng.shuffle(harvested_samples)
            ep_loss = 0.0
            n_b = 0
            for start in range(0, len(harvested_samples), batch_size):
                batch = harvested_samples[start : start + batch_size]
                B = len(batch)
                max_len = max(len(x.token_ids) for x in batch)

                inp = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
                act_pos = torch.zeros(B, dtype=torch.long, device=device)
                masks = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=device)
                target_pis = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
                target_vs = torch.zeros(B, dtype=torch.float32, device=device)

                for b_i, s in enumerate(batch):
                    L = len(s.token_ids)
                    inp[b_i, :L] = torch.tensor(s.token_ids, dtype=torch.long, device=device)
                    act_pos[b_i] = L - 1
                    masks[b_i] = torch.from_numpy(s.legal_mask).to(device)
                    target_pis[b_i] = torch.from_numpy(s.target_policy).to(device)
                    target_vs[b_i] = s.target_value

                logits, pred_v = model(inp, act_pos, masks)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)

                with torch.no_grad():
                    ref_logits, _ = ref_model(inp, act_pos, masks)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                p_loss = -(target_pis * log_probs).sum(dim=-1).mean()
                v_loss = F.smooth_l1_loss(pred_v, target_vs)
                # Anchor gently to model's prior policy so each generation keeps its distinct play style while improving
                kl_div = (probs * (log_probs.masked_fill(~masks, 0.0) - ref_log_probs.masked_fill(~masks, 0.0))).sum(dim=-1).mean()
                loss = p_loss + 0.25 * v_loss + 0.12 * kl_div

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                ep_loss += float(loss.item())
                n_b += 1

            avg_ep_loss = ep_loss / max(1, n_b)
            if ep == 1:
                init_loss = avg_ep_loss
            final_loss = avg_ep_loss

        ck["model_state_dict"] = model.state_dict()
        torch.save(ck, ckpt_path)
        _emit(
            f"  [{idx_m}/{len(trainable_ckpts)}] Retrained {disp_name} ({ckpt_path}) | "
            f"Loss: {init_loss:.4f} -> {final_loss:.4f}"
        )
        retrained_summaries.append({
            "name": disp_name,
            "checkpoint": ckpt_path,
            "initial_loss": round(init_loss, 4),
            "final_loss": round(final_loss, 4),
        })

    _emit(f"[Step 4/4] Running Post-Retraining Championship ({championship_rounds + 1} rounds/pair) to rank all updated models...")
    run_leaderboard_tournament(
        tokenizer, device, rounds_per_pair=championship_rounds + 1, reset=True, telemetry_cb=telemetry_cb
    )

    lb_after = LeaderboardManager()
    top3 = [
        f"#{idx} {p.display_name} ({p.elo:.0f} Elo, {p.net_pts_per_hand:+.2f} pts/h)"
        for idx, p in enumerate(lb_after.get_ranked_players()[:3], start=1)
    ]
    _emit(f"🏆 Championship & Collective Retraining Complete! New Top 3: {' | '.join(top3)}")

    return {
        "hands_harvested": total_hands_played,
        "states_trained": len(harvested_samples),
        "models_retrained": len(retrained_summaries),
        "retrained_models": retrained_summaries,
        "new_top3": top3,
    }

