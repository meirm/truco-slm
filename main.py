"""
CLI entrypoint to train, benchmark, or interactively play Argentine Truco against
the System 1 Small Language Model ("JEV-Truco").
"""

from __future__ import annotations
import argparse
import os
import time
from typing import Optional

import torch

from truco.arena import (
    evaluate_head_to_head,
    run_championship_and_retrain_all,
    run_dual_instance_training,
    train_against_leader,
)
from truco.engine import Action, ACTION_NAMES, TrucoHandState
from truco.leaderboard import LeaderboardManager, run_leaderboard_tournament
from truco.model import System1TrucoSLM
from truco.tokenizer import TrucoTokenizer
from truco.trainer import evaluate_agent, train_system1_slm

DEFAULT_CHECKPOINT = "checkpoints/jev_truco_slm.pt"
BASE_CHECKPOINT = "checkpoints/jev_base_slm.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
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


def run_training(args: argparse.Namespace) -> None:
    device = get_device()
    tokenizer = TrucoTokenizer()
    model = build_model(tokenizer)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resuming & accumulating knowledge from existing checkpoint: {args.checkpoint}")

    num_params = sum(p.numel() for p in model.parameters())
    print("=" * 68)
    print(" JEV-Truco: System 1 Small Language Model (Zero-Thinking Action Head)")
    print("=" * 68)
    print(f" Device         : {device}")
    print(f" Vocabulary Size: {tokenizer.vocab_size} tokens")
    print(f" Model Params   : {num_params:,} parameters (4-layer SwiGLU Transformer)")
    print("-" * 68)

    t0 = time.perf_counter()
    metrics = train_system1_slm(
        model=model,
        tokenizer=tokenizer,
        cfr_iterations=args.cfr_iters,
        distill_hands=args.distill_hands,
        distill_epochs=args.epochs,
        self_play_episodes=args.self_play_episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s.")

    # Benchmark System 1 latency & game strength
    t_lat0 = time.perf_counter()
    dummy_state = TrucoHandState.new_hand()
    for _ in range(100):
        model.act(dummy_state, tokenizer, device, temperature=0.0)
    avg_latency_ms = ((time.perf_counter() - t_lat0) / 100.0) * 1000.0

    eval_stats = evaluate_agent(model, tokenizer, device, num_hands=400)
    print("-" * 68)
    print(" Evaluation Results (400 duplicate hands):")
    print(f"  * System 1 Inference Latency : {avg_latency_ms:.2f} ms / decision (1 forward pass)")
    print(f"  * vs Random Agent            : {eval_stats['avg_pts_vs_random']:+.2f} net pts/hand "
          f"(Win rate: {eval_stats['win_rate_vs_random']*100:.1f}%)")
    print(f"  * vs Heuristic Bluff Agent   : {eval_stats['avg_pts_vs_heuristic']:+.2f} net pts/hand "
          f"(Win rate: {eval_stats['win_rate_vs_heuristic']*100:.1f}%)")
    print("-" * 68)

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "metrics": metrics,
            "eval_stats": eval_stats,
        },
        args.checkpoint,
    )
    print(f"Saved System 1 SLM checkpoint to {args.checkpoint}")

    if args.demo:
        run_demo_hand(model, tokenizer, device)


def run_league(args: argparse.Namespace) -> None:
    """
    Runs Dual-Instance Self-Play (Instance 1 vs Instance 2) and periodically
    evaluates the active learner against the frozen Base Model.
    Accumulates knowledge across runs by loading the latest learner checkpoint (`args.checkpoint`).
    """
    device = get_device()
    tokenizer = TrucoTokenizer()

    base_model = build_model(tokenizer)
    learner_model = build_model(tokenizer)

    # 1. Load the frozen Base Model (Gen 0 yardstick)
    if os.path.exists(BASE_CHECKPOINT):
        ckpt = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded frozen Base Model (Gen 0 yardstick) from {BASE_CHECKPOINT}")
    elif os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Snapshot current {args.checkpoint} as frozen Base Model ({BASE_CHECKPOINT})")
    else:
        print("Initialized fresh Base Model.")

    # 2. Load the latest trained Learner checkpoint so knowledge accumulates across runs!
    if os.path.exists(args.checkpoint):
        latest_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        learner_model.load_state_dict(latest_ckpt["model_state_dict"])
        print(f"Resumed Active Learner from latest checkpoint: {args.checkpoint}")
    else:
        learner_model.load_state_dict(base_model.state_dict())

    print("=" * 78)
    print(" Dual-Instance Arena Training: Instance 1 (Learner) vs Instance 2 (Sparring)")
    print(" Benchmarking after every generation against Frozen Base Model")
    print("=" * 78)

    t0 = time.perf_counter()
    history = run_dual_instance_training(
        base_model=base_model,
        learner_model=learner_model,
        tokenizer=tokenizer,
        device=device,
        generations=args.generations,
        hands_per_gen=args.hands_per_gen,
        ppo_epochs=4,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    elapsed = time.perf_counter() - t0

    print("-" * 78)
    print(f"Dual-instance training finished in {elapsed:.2f}s.")
    final_log = history[-1]
    print(
        f"Final Learner vs Frozen Base Model:\n"
        f"  * Net Points / Hand : {final_log['net_pts_vs_base']:+.3f} pts/hand\n"
        f"  * Duplicate Hand Win: {final_log['hand_wr_vs_base']*100:.1f}%\n"
        f"  * 15-pt Match Win   : {final_log['match_wr_vs_base']*100:.1f}% "
        f"(Elo gain: {final_log['elo_vs_base']:+.0f})"
    )
    print("=" * 78)


def run_demo_hand(
    model: System1TrucoSLM,
    tokenizer: TrucoTokenizer,
    device: torch.device,
) -> None:
    print("\n" + "=" * 68)
    print(" Sample System 1 Self-Play Hand Trace (Zero-CoT Single-Token Policy)")
    print("=" * 68)
    state = TrucoHandState.new_hand(mano=0, match_scores=(8, 7))
    print(f"Player 0 (Mano) Hand : {[str(c) for c in state.hands[0]]} (Envido: {state.envido_scores[0]})")
    print(f"Player 1 (Pie)  Hand : {[str(c) for c in state.hands[1]]} (Envido: {state.envido_scores[1]})")
    print("-" * 68)

    step_num = 1
    while not state.is_terminal:
        p = state.current_player
        prompt_str = tokenizer.format_prompt_text(state, p)
        chosen, probs, val = model.act(state, tokenizer, device, temperature=0.0)

        # Format top-3 action probabilities
        top_actions = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_str = ", ".join(
            f"{ format_action(state, p, a) }: {prob*100:.1f}%"
            for a, prob in top_actions
        )
        print(f"[Turn {step_num}] Player {p} System-1 Prompt: {prompt_str}")
        print(f"         -> Action: {format_action(state, p, chosen)} | Value Est: {val:+.2f} pts | Top Policy: [{top_str}]")
        state.step(chosen)
        step_num += 1

    print("-" * 68)
    print(f"Hand Result: Player 0 won {state.points_won[0]} pts | Player 1 won {state.points_won[1]} pts")
    if state.envido_resolved:
        print(f"  Envido Winner: Player {state.envido_winner} (+{state.envido_awarded_points} pts)")
    print(f"  Truco Winner : Player {state.truco_winner} (+{state.truco_stakes} pts)")
    print("=" * 68)


def format_action(state: TrucoHandState, player: int, action: Action) -> str:
    if action in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2):
        card = state.hands[player][int(action)]
        return f"Play {card} ({card.short_str()})"
    return ACTION_NAMES[action]


def play_interactive(args: argparse.Namespace) -> None:
    device = get_device()
    tokenizer = TrucoTokenizer()
    model = build_model(tokenizer)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print(f"Loaded trained System 1 SLM from {args.checkpoint}")
    else:
        print("No checkpoint found. Running quick training first...")
        run_training(args)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

    scores = [0, 0]
    mano = 0
    target = 15

    print("\n" + "=" * 68)
    print(" Interactive Argentine Truco vs System 1 SLM (You are Player 0)")
    print("=" * 68)

    while scores[0] < target and scores[1] < target:
        state = TrucoHandState.new_hand(mano=mano, match_scores=(scores[0], scores[1]), target_score=target)
        print(f"\n--- NEW HAND | Match Score: You {scores[0]} - {scores[1]} JEV-SLM | Mano: {'You' if mano == 0 else 'JEV-SLM'} ---")
        print(f"Your Cards: {[f'[{i}] {c}' for i, c in enumerate(state.hands[0])]} (Your Envido: {state.envido_scores[0]})")

        while not state.is_terminal:
            p = state.current_player
            if p == 0:
                legal = state.legal_actions()
                print("\nYour turn! Legal actions:")
                for idx, a in enumerate(legal):
                    print(f"  ({idx}) {format_action(state, 0, a)}")
                choice_idx: Optional[int] = None
                while choice_idx is None:
                    raw = input("Select action index: ").strip()
                    if raw.lower() in ("q", "quit", "exit"):
                        print("Exiting match.")
                        return
                    if raw.isdigit() and 0 <= int(raw) < len(legal):
                        choice_idx = int(raw)
                chosen = legal[choice_idx]
                print(f"You chose: {format_action(state, 0, chosen)}")
                state.step(chosen)
            else:
                chosen, probs, val = model.act(state, tokenizer, device, temperature=0.1)
                print(f"JEV-SLM plays: {format_action(state, 1, chosen)}  (System 1 confidence: {probs[chosen]*100:.1f}%, EV: {val:+.2f})")
                state.step(chosen)

        scores[0] += state.points_won[0]
        scores[1] += state.points_won[1]
        print(f"--> Hand ended! Points won this hand: You +{state.points_won[0]}, JEV-SLM +{state.points_won[1]}")
        print(f"    Opponent's cards were: {[str(c) for c in state.hands[1]]} (Envido: {state.envido_scores[1]})")
        mano = 1 - mano

    print("\n" + "=" * 68)
    if scores[0] >= target:
        print(f" CONGRATULATIONS! You won the match {scores[0]} to {scores[1]}!")
    else:
        print(f" JEV-SLM won the match {scores[1]} to {scores[0]}!")

    # Record human match on the persistent leaderboard!
    lb = LeaderboardManager()
    lb.register_player("human_player", "You (Human)", "Human Mind", "-", 1500.0)
    lb.register_player("jev_champion", "JEV-Active (Latest)", "System1-SLM (4L SwiGLU Latest)", args.checkpoint, 1500.0)
    lb.record_match("human_player", "jev_champion", winner_is_a=(scores[0] >= target))
    lb.save()
    print("\nUpdated Leaderboard:")
    print(lb.format_table())


def run_leaderboard_cli(args: argparse.Namespace) -> None:
    device = get_device()
    tokenizer = TrucoTokenizer()
    lb = LeaderboardManager()
    if args.tournament_rounds > 0 or len(lb.players) == 0:
        lb = run_leaderboard_tournament(
            tokenizer=tokenizer,
            device=device,
            rounds_per_pair=max(1, args.tournament_rounds),
            reset=args.reset_leaderboard,
        )
    print("\n" + lb.format_table())


def run_challenge_leader(args: argparse.Namespace) -> None:
    """
    Finds the #1 ranked model on the Leaderboard (the Champion/Leader) and trains
    a Challenger model initialized from the Leader against the Top-3 Leaderboard Gauntlet
    using Advantage-Weighted Counterfactual Multi-Action Rollouts.
    """
    device = get_device()
    tokenizer = TrucoTokenizer()
    lb = LeaderboardManager()

    # Find top ranked neural models on the leaderboard
    ranked = [p for p in lb.get_ranked_players() if os.path.exists(p.checkpoint_path) and p.checkpoint_path.endswith(".pt")]
    if ranked:
        leader_stats = ranked[0]
        leader_ckpt_path = leader_stats.checkpoint_path
        leader_name = f"{leader_stats.display_name} ({leader_stats.elo:.0f} Elo)"
    else:
        leader_ckpt_path = BASE_CHECKPOINT if os.path.exists(BASE_CHECKPOINT) else args.checkpoint
        leader_name = "JEV-Base (Gen 0)"

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

    # Load runners-up (#2 and #3 on the Leaderboard) into the Gauntlet so the Challenger
    # beats the #1 Leader AND generalizes across all top styles!
    extra_opponents: list[System1TrucoSLM] = []
    for runner_up in ranked[1:4]:
        opp_m = build_model(tokenizer)
        r_ckpt = torch.load(runner_up.checkpoint_path, map_location=device, weights_only=False)
        opp_m.load_state_dict(r_ckpt["model_state_dict"])
        extra_opponents.append(opp_m)

    # Start Challenger from the current #1 Leader's weights so we strictly improve on the Champion
    challenger_model = build_model(tokenizer)
    challenger_model.load_state_dict(leader_model.state_dict())

    print("=" * 82)
    print(f" KING-OF-THE-HILL TRAINING: Challenger vs #1 Leader [{leader_name}] + Top Gauntlet")
    print("=" * 82)

    for g in range(1, args.generations + 1):
        print(f"\n--- Round {g}/{args.generations}: Training Challenger against Leader [{leader_name}] ---")
        h2h = train_against_leader(
            challenger_model=challenger_model,
            leader_model=leader_model,
            leader_name=leader_name,
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            extra_opponents=extra_opponents,
            rounds=args.hands_per_gen,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        # Save new generation checkpoint
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
        torch.save({"model_state_dict": challenger_model.state_dict(), "gen": next_gen}, args.checkpoint)
        print(f"        Saved Challenger checkpoint -> {gen_path} (and updated {args.checkpoint})")

        # If Challenger dethroned the Leader, promote Challenger to be the new Leader!
        if h2h.net_pts_per_hand > 0.02 or h2h.match_win_rate >= 0.52:
            extra_opponents.insert(0, build_model(tokenizer))
            extra_opponents[0].load_state_dict(leader_model.state_dict())
            leader_model.load_state_dict(challenger_model.state_dict())
            leader_name = f"JEV-League (Gen {next_gen}) [New King]"
            print(f"        👑 DETHRONED! Challenger promoted to new Leader: {leader_name}")


def run_championship_retrain_cli(args: argparse.Namespace) -> None:
    device = get_device()
    tokenizer = TrucoTokenizer()
    run_championship_and_retrain_all(
        tokenizer=tokenizer,
        device=device,
        championship_rounds=args.tournament_rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    lb = LeaderboardManager()
    print("\n" + lb.format_table())


def main() -> None:
    parser = argparse.ArgumentParser(description="JEV-Truco System 1 SLM Trainer & CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "league", "challenge-leader", "championship-retrain", "leaderboard", "play"],
        default="leaderboard",
        help="Train, run dual-instance league, challenge #1 leader, run championship + retrain all models, view/run leaderboard, or play",
    )
    parser.add_argument("--cfr-iters", type=int, default=1000, help="Number of MCCFR iterations")
    parser.add_argument("--distill-hands", type=int, default=1500, help="Number of hands to sample for distillation")
    parser.add_argument("--epochs", type=int, default=6, help="Distillation/training epochs")
    parser.add_argument("--self-play-episodes", type=int, default=800, help="Self-play RL episodes")
    parser.add_argument("--generations", type=int, default=3, help="Number of dual-instance or leader-challenge generations")
    parser.add_argument("--hands-per-gen", type=int, default=600, help="Hands played per generation")
    parser.add_argument("--tournament-rounds", type=int, default=3, help="Mirrored duplicate match rounds per pair for leaderboard/championship")
    parser.add_argument("--reset-leaderboard", action="store_true", help="Reset leaderboard stats before running tournament")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to save/load model checkpoint")
    parser.add_argument("--demo", action="store_true", default=True, help="Run a sample hand trace after training")
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "league":
        run_league(args)
    elif args.mode == "challenge-leader":
        run_challenge_leader(args)
    elif args.mode == "championship-retrain":
        run_championship_retrain_cli(args)
    elif args.mode == "leaderboard":
        run_leaderboard_cli(args)
    else:
        play_interactive(args)


if __name__ == "__main__":
    main()
