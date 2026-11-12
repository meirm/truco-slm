"""
Training pipeline for the System 1 Truco SLM ("JEV-Truco"):

Stage 1: Outcome-Sampling Monte Carlo Counterfactual Regret Minimization (OS-MCCFR)
  - Explores imperfect-information game trees of Argentine Truco
  - Computes regret-matched Nash approximation strategies and state values
  - Collects diverse game states paired with their CFR target policies and values

Stage 2: System 1 SLM Distillation & Self-Play RL (Actor-Critic / PPO)
  - Trains `System1TrucoSLM` via soft cross-entropy (KL divergence) against CFR/expert policies
    and MSE on state values
  - Refines the System 1 model via batched self-play rollouts with Generalized Advantage
    Estimation (GAE) and PPO-style policy gradient updates
"""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from truco.engine import Action, NUM_ACTIONS, PendingState, TrucoHandState
from truco.model import System1TrucoSLM
from truco.tokenizer import TrucoTokenizer, infoset_key


@dataclass
class InfoNode:
    """Stores cumulative regrets and strategy sum for an abstracted Information Set."""
    regret_sum: np.ndarray
    strategy_sum: np.ndarray
    value_sum: float = 0.0
    visits: int = 0

    @classmethod
    def create(cls) -> InfoNode:
        return cls(
            regret_sum=np.zeros(NUM_ACTIONS, dtype=np.float64),
            strategy_sum=np.zeros(NUM_ACTIONS, dtype=np.float64),
        )

    def get_strategy(self, legal_actions: List[Action], realization_weight: float = 1.0) -> np.ndarray:
        strategy = np.zeros(NUM_ACTIONS, dtype=np.float64)
        norm_sum = 0.0
        for a in legal_actions:
            idx = int(a)
            r = max(0.0, self.regret_sum[idx])
            strategy[idx] = r
            norm_sum += r

        if norm_sum > 0.0:
            strategy /= norm_sum
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                strategy[int(a)] = uniform

        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self, legal_actions: List[Action]) -> np.ndarray:
        avg = np.zeros(NUM_ACTIONS, dtype=np.float64)
        norm_sum = 0.0
        for a in legal_actions:
            idx = int(a)
            norm_sum += self.strategy_sum[idx]

        if norm_sum > 0.0:
            for a in legal_actions:
                avg[int(a)] = self.strategy_sum[int(a)] / norm_sum
        else:
            uniform = 1.0 / len(legal_actions)
            for a in legal_actions:
                avg[int(a)] = uniform
        return avg


class HeuristicAnchorAgent:
    """
    Strong rule-based & bluff-capable Argentine Truco baseline used for evaluation
    and initial prior shaping.
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng or random.Random()

    def policy_distribution(self, state: TrucoHandState) -> np.ndarray:
        legal = state.legal_actions()
        probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
        p = state.current_player
        env_score = state.envido_scores[p]
        avail_slots = [s for s in range(3) if not state.played_mask[p][s]]
        avail_powers = [state.hands[p][s].truco_power for s in avail_slots]
        avg_power = sum(avail_powers) / max(1, len(avail_powers))
        max_power = max(avail_powers) if avail_powers else 0

        # 1. Responding to Envido
        if state.pending == PendingState.ENVIDO:
            if env_score >= 30 and Action.CALL_REAL_ENVIDO in legal:
                probs[int(Action.CALL_REAL_ENVIDO)] = 0.65
                probs[int(Action.QUIERO)] = 0.35
            elif env_score >= 28 and Action.CALL_ENVIDO in legal:
                probs[int(Action.CALL_ENVIDO)] = 0.55
                probs[int(Action.QUIERO)] = 0.45
            elif env_score >= 25:
                probs[int(Action.QUIERO)] = 0.85
                probs[int(Action.NO_QUIERO)] = 0.15
            elif env_score >= 22:
                probs[int(Action.QUIERO)] = 0.35
                probs[int(Action.NO_QUIERO)] = 0.65
            else:
                probs[int(Action.NO_QUIERO)] = 0.92
                probs[int(Action.QUIERO)] = 0.08
            return self._normalize_legal(probs, legal)

        # 2. Responding to Truco
        if state.pending == PendingState.TRUCO:
            # Check if we should call Envido first ("El Envido está primero")
            if Action.CALL_ENVIDO in legal and env_score >= 27:
                probs[int(Action.CALL_ENVIDO)] = 0.75
                probs[int(Action.QUIERO)] = 0.25
                return self._normalize_legal(probs, legal)

            if max_power >= 12 or avg_power >= 9.5:
                for esc in (Action.CALL_RETRUCO, Action.CALL_VALE_CUATRO):
                    if esc in legal:
                        probs[int(esc)] = 0.45
                probs[int(Action.QUIERO)] = 0.55
            elif max_power >= 9 or avg_power >= 7.0:
                probs[int(Action.QUIERO)] = 0.75
                probs[int(Action.NO_QUIERO)] = 0.25
            else:
                probs[int(Action.NO_QUIERO)] = 0.80
                probs[int(Action.QUIERO)] = 0.20  # Occasional bluff accept
            return self._normalize_legal(probs, legal)

        # 3. Normal turn: Check if we should call Envido in Trick 0
        if Action.CALL_ENVIDO in legal:
            if env_score >= 31:
                probs[int(Action.CALL_REAL_ENVIDO)] = 0.50
                probs[int(Action.CALL_ENVIDO)] = 0.40
            elif env_score >= 26:
                probs[int(Action.CALL_ENVIDO)] = 0.75
            elif env_score <= 19 and self.rng.random() < 0.12:
                # Tactical bluff Envido
                probs[int(Action.CALL_ENVIDO)] = 0.25

        # 4. Check if we should call Truco / Retruco / Vale Cuatro
        for t_act in (Action.CALL_TRUCO, Action.CALL_RETRUCO, Action.CALL_VALE_CUATRO):
            if t_act in legal:
                if max_power >= 11 or avg_power >= 9.0:
                    probs[int(t_act)] += 0.45
                elif len(state.tricks) >= 2 and self.rng.random() < 0.18:
                    # Late trick bluff
                    probs[int(t_act)] += 0.25

        # 5. Card selection heuristics (play lowest winning card if responding, or strong card if leading)
        current_trick = state.tricks[-1]
        if len(current_trick) == 1:
            opp_power = current_trick[0].card.truco_power
            winning_slots = [s for s in avail_slots if state.hands[p][s].truco_power > opp_power]
            tying_slots = [s for s in avail_slots if state.hands[p][s].truco_power == opp_power]
            if winning_slots:
                # Matá con la más chica (win with the weakest winning card)
                best_slot = min(winning_slots, key=lambda s: state.hands[p][s].truco_power)
                probs[best_slot] += 0.70
            elif tying_slots:
                probs[tying_slots[0]] += 0.60
            else:
                # Throw lowest card
                lowest_slot = min(avail_slots, key=lambda s: state.hands[p][s].truco_power)
                probs[lowest_slot] += 0.70
        else:
            # Leading the trick: in Trick 0, often play medium/strong card to win primera
            sorted_slots = sorted(avail_slots, key=lambda s: state.hands[p][s].truco_power, reverse=True)
            for i, s in enumerate(sorted_slots):
                probs[s] += 0.50 if i == 0 else (0.30 if i == 1 else 0.15)

        return self._normalize_legal(probs, legal)

    def _normalize_legal(self, probs: np.ndarray, legal: List[Action]) -> np.ndarray:
        masked = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal:
            masked[int(a)] = max(1e-4, probs[int(a)])
        masked /= masked.sum()
        return masked

    def act(self, state: TrucoHandState) -> Action:
        dist = self.policy_distribution(state)
        idx = int(np.random.choice(NUM_ACTIONS, p=dist))
        return Action(idx)


class MCCFRTrainer:
    """
    External/Outcome-Sampling Monte Carlo CFR for Argentine Truco.
    Builds an equilibrium-approximating table of Information Sets and generates
    high-quality training targets for the System 1 SLM.
    """

    def __init__(self, seed: int = 42) -> None:
        self.nodes: Dict[Tuple[int, ...], InfoNode] = {}
        self.rng = random.Random(seed)
        self.heuristic = HeuristicAnchorAgent(self.rng)

    def get_node(self, state: TrucoHandState) -> InfoNode:
        key = infoset_key(state)
        node = self.nodes.get(key)
        if node is None:
            node = InfoNode.create()
            # Warm-start initial strategy sum slightly with heuristic prior so rare states play sensibly
            prior = self.heuristic.policy_distribution(state)
            node.strategy_sum += 2.0 * prior
            node.regret_sum += 0.5 * prior
            self.nodes[key] = node
        return node

    def cfr_traverse(
        self,
        state: TrucoHandState,
        traverser: int,
        pi_i: float,
        pi_opp: float,
        depth: int = 0,
    ) -> float:
        if state.is_terminal:
            return state.payoff(traverser)

        p = state.current_player
        legal = state.legal_actions()
        node = self.get_node(state)
        strategy = node.get_strategy(legal, realization_weight=pi_i if p == traverser else pi_opp)

        if p == traverser:
            # Traverser node: explore all legal actions at shallow depth, or sample subset
            action_utils = np.zeros(NUM_ACTIONS, dtype=np.float64)
            node_util = 0.0
            for a in legal:
                idx = int(a)
                next_state = state.clone().step(a)
                u = self.cfr_traverse(
                    next_state,
                    traverser,
                    pi_i * strategy[idx],
                    pi_opp,
                    depth + 1,
                )
                action_utils[idx] = u
                node_util += strategy[idx] * u

            # Counterfactual regret update
            for a in legal:
                idx = int(a)
                regret = action_utils[idx] - node_util
                node.regret_sum[idx] += pi_opp * regret

            node.value_sum += node_util
            node.visits += 1
            return node_util
        else:
            # Opponent node: external sampling (sample 1 action according to opponent's current strategy)
            probs = [strategy[int(a)] for a in legal]
            chosen = self.rng.choices(legal, weights=probs, k=1)[0]
            idx = int(chosen)
            next_state = state.clone().step(chosen)
            u = self.cfr_traverse(
                next_state,
                traverser,
                pi_i,
                pi_opp * strategy[idx],
                depth + 1,
            )
            node.value_sum += -u
            node.visits += 1
            return u

    def run_iterations(self, num_iterations: int) -> None:
        for i in range(num_iterations):
            mano = i % 2
            s0 = self.rng.randint(0, 14)
            s1 = self.rng.randint(0, 14)
            root = TrucoHandState.new_hand(mano=mano, match_scores=(s0, s1), rng=self.rng)
            for traverser in (0, 1):
                self.cfr_traverse(root.clone(), traverser=traverser, pi_i=1.0, pi_opp=1.0)


@dataclass
class System1Sample:
    token_ids: List[int]
    legal_mask: np.ndarray
    target_policy: np.ndarray
    target_value: float


def generate_distillation_dataset(
    cfr: MCCFRTrainer,
    tokenizer: TrucoTokenizer,
    num_hands: int = 1500,
) -> List[System1Sample]:
    """
    Generates a dataset of `(observation_tokens, legal_mask, target_policy, realized_and_cfr_value)`
    for training the System 1 SLM.
    """
    samples: List[System1Sample] = []
    rng = cfr.rng

    for h in range(num_hands):
        mano = h % 2
        s0 = rng.randint(0, 14)
        s1 = rng.randint(0, 14)
        state = TrucoHandState.new_hand(mano=mano, match_scores=(s0, s1), rng=rng)

        traj: List[Tuple[int, List[int], np.ndarray, np.ndarray, Optional[InfoNode]]] = []
        while not state.is_terminal:
            p = state.current_player
            legal = state.legal_actions()
            toks = tokenizer.encode_observation(state, p)
            if len(toks) > tokenizer.max_seq_len:
                toks = [toks[0]] + toks[-(tokenizer.max_seq_len - 1):]

            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            for a in legal:
                mask[int(a)] = True

            node = cfr.get_node(state)
            avg_strat = node.get_average_strategy(legal)

            traj.append((p, toks, mask, avg_strat, node))

            # Epsilon-greedy exploration during trajectory collection
            if rng.random() < 0.15:
                chosen = rng.choice(legal)
            else:
                chosen = rng.choices(legal, weights=[avg_strat[int(a)] for a in legal], k=1)[0]
            state.step(chosen)

        for p, toks, mask, avg_strat, node in traj:
            realized_val = state.payoff(p)
            cfr_val = (node.value_sum / node.visits) if (node and node.visits > 0) else realized_val
            blended_val = 0.5 * realized_val + 0.5 * cfr_val
            samples.append(
                System1Sample(
                    token_ids=toks,
                    legal_mask=mask,
                    target_policy=avg_strat.astype(np.float32),
                    target_value=float(blended_val),
                )
            )

    return samples


def collate_samples(
    batch: List[System1Sample],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(s.token_ids) for s in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    act_positions = torch.zeros(B, dtype=torch.long, device=device)
    legal_masks = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=device)
    target_policies = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
    target_values = torch.zeros(B, dtype=torch.float32, device=device)

    for i, s in enumerate(batch):
        L = len(s.token_ids)
        input_ids[i, :L] = torch.tensor(s.token_ids, dtype=torch.long, device=device)
        act_positions[i] = L - 1
        legal_masks[i] = torch.from_numpy(s.legal_mask).to(device)
        target_policies[i] = torch.from_numpy(s.target_policy).to(device)
        target_values[i] = s.target_value

    return input_ids, act_positions, legal_masks, target_policies, target_values


def train_system1_slm(
    model: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    cfr_iterations: int = 800,
    distill_hands: int = 1500,
    distill_epochs: int = 6,
    self_play_episodes: int = 800,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Full 2-stage training pipeline:
    1. MCCFR Nash approximation + Knowledge Distillation into `System1TrucoSLM`
    2. Self-Play Actor-Critic (PPO) policy/value refinement
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[Stage 1/2] Running Outcome-Sampling MCCFR ({cfr_iterations} iterations)...")
    cfr = MCCFRTrainer(seed=42)
    cfr.run_iterations(cfr_iterations)
    print(f"  -> Discovered {len(cfr.nodes):,} unique Truco Information Sets.")

    print(f"[Stage 1/2] Generating {distill_hands} hands of CFR equilibrium trajectories...")
    dataset = generate_distillation_dataset(cfr, tokenizer, num_hands=distill_hands)
    print(f"  -> Collected {len(dataset):,} state-action-value training positions.")

    model.train()
    last_policy_loss = 0.0
    last_value_loss = 0.0

    for epoch in range(1, distill_epochs + 1):
        random.shuffle(dataset)
        total_p_loss = 0.0
        total_v_loss = 0.0
        n_batches = 0

        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            inp, act_pos, mask, target_pi, target_v = collate_samples(batch, tokenizer.pad_id, device)

            logits, pred_v = model(inp, act_pos, mask)
            log_probs = F.log_softmax(logits, dim=-1)
            # Soft cross-entropy with CFR policy
            policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(pred_v, target_v)
            loss = policy_loss + 0.25 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_p_loss += float(policy_loss.item())
            total_v_loss += float(value_loss.item())
            n_batches += 1

        last_policy_loss = total_p_loss / max(1, n_batches)
        last_value_loss = total_v_loss / max(1, n_batches)
        print(
            f"  Epoch {epoch}/{distill_epochs} | Policy CE: {last_policy_loss:.4f} | Value Loss: {last_value_loss:.4f}"
        )

    # --- Stage 2: Self-Play Actor-Critic Refinement ---
    print(f"[Stage 2/2] Running {self_play_episodes} Self-Play RL hands...")
    rl_optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
    rng = random.Random(101)

    rl_buffer: List[Tuple[List[int], np.ndarray, int, float, float]] = []
    for ep in range(1, self_play_episodes + 1):
        mano = ep % 2
        s0 = rng.randint(0, 14)
        s1 = rng.randint(0, 14)
        state = TrucoHandState.new_hand(mano=mano, match_scores=(s0, s1), rng=rng)
        ep_steps: List[Tuple[int, List[int], np.ndarray, int, float]] = []

        while not state.is_terminal:
            p = state.current_player
            legal = state.legal_actions()
            toks = tokenizer.encode_observation(state, p)
            if len(toks) > tokenizer.max_seq_len:
                toks = [toks[0]] + toks[-(tokenizer.max_seq_len - 1):]

            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            for a in legal:
                mask[int(a)] = True

            act, _, val = model.act(state, tokenizer, device, temperature=0.8)
            ep_steps.append((p, toks, mask, int(act), val))
            state.step(act)

        for p, toks, mask, act_idx, old_val in ep_steps:
            ret = state.payoff(p)
            adv = ret - old_val
            rl_buffer.append((toks, mask, act_idx, ret, adv))

        if len(rl_buffer) >= batch_size:
            model.train()
            B = len(rl_buffer)
            max_len = max(len(x[0]) for x in rl_buffer)
            inp = torch.full((B, max_len), tokenizer.pad_id, dtype=torch.long, device=device)
            act_pos = torch.zeros(B, dtype=torch.long, device=device)
            masks = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=device)
            actions = torch.zeros(B, dtype=torch.long, device=device)
            returns = torch.zeros(B, dtype=torch.float32, device=device)
            advs = torch.zeros(B, dtype=torch.float32, device=device)

            for i, (toks, mask, act_idx, ret, adv) in enumerate(rl_buffer):
                L = len(toks)
                inp[i, :L] = torch.tensor(toks, dtype=torch.long, device=device)
                act_pos[i] = L - 1
                masks[i] = torch.from_numpy(mask).to(device)
                actions[i] = act_idx
                returns[i] = ret
                advs[i] = adv

            # Normalize advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)

            logits, values = model(inp, act_pos, masks)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs.masked_fill(~masks, 0.0)).sum(dim=-1).mean()

            pg_loss = -(chosen_log_probs * advs).mean()
            v_loss = F.smooth_l1_loss(values, returns)
            loss = pg_loss + 0.5 * v_loss - 0.02 * entropy

            rl_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            rl_optimizer.step()
            rl_buffer.clear()

    return {
        "cfr_infosets": float(len(cfr.nodes)),
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
    }


@torch.no_grad()
def evaluate_agent(
    model: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    device: torch.device,
    num_hands: int = 500,
) -> Dict[str, float]:
    """
    Evaluates the System 1 SLM against both a Random agent and the HeuristicAnchorAgent.
    """
    model.eval()
    rng = random.Random(999)
    heuristic = HeuristicAnchorAgent(rng)

    # 1. vs Random
    pts_model_rand = 0.0
    wins_rand = 0
    for h in range(num_hands):
        state = TrucoHandState.new_hand(mano=h % 2, rng=rng)
        while not state.is_terminal:
            if state.current_player == 0:
                act, _, _ = model.act(state, tokenizer, device, temperature=0.0)
            else:
                act = rng.choice(state.legal_actions())
            state.step(act)
        diff = state.payoff(0)
        pts_model_rand += diff
        if diff > 0:
            wins_rand += 1

    # 2. vs HeuristicAnchorAgent
    pts_model_heur = 0.0
    wins_heur = 0
    for h in range(num_hands):
        state = TrucoHandState.new_hand(mano=h % 2, rng=rng)
        while not state.is_terminal:
            if state.current_player == 0:
                act, _, _ = model.act(state, tokenizer, device, temperature=0.0)
            else:
                act = heuristic.act(state)
            state.step(act)
        diff = state.payoff(0)
        pts_model_heur += diff
        if diff > 0:
            wins_heur += 1

    return {
        "avg_pts_vs_random": pts_model_rand / num_hands,
        "win_rate_vs_random": wins_rand / num_hands,
        "avg_pts_vs_heuristic": pts_model_heur / num_hands,
        "win_rate_vs_heuristic": wins_heur / num_hands,
    }
