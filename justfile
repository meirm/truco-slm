# JEV-Truco: System 1 Small Language Model & Web Arena Justfile

python := ".venv/bin/python"

# List all available recipes
default:
    @just --list

# View the current Leaderboard and Head-to-Head Matchup Matrix
leaderboard:
    {{python}} main.py --mode leaderboard --tournament-rounds 0

# Run a fresh Round-Robin Tournament across all bots and print the Leaderboard
tournament rounds="8":
    {{python}} main.py --mode leaderboard --tournament-rounds {{rounds}} --reset-leaderboard

# Run a full Round-Robin Championship, harvest Full-Card Hindsight rollouts, retrain ALL neural models, and re-rank Leaderboard
championship-retrain rounds="2" epochs="5":
    {{python}} main.py --mode championship-retrain --tournament-rounds {{rounds}} --epochs {{epochs}}

# Train Challenger against the #1 Leaderboard Champion + Top-3 Gauntlet (King of the Hill)
challenge-leader generations="3" hands="1000" epochs="5":
    {{python}} main.py --mode challenge-leader --generations {{generations}} --hands-per-gen {{hands}} --epochs {{epochs}}

# Train Dual-Instance League (Instance 1 vs Instance 2) for 5,000 hands (5 gens x 1,000 hands)
league generations="5" hands="1000":
    {{python}} main.py --mode league --generations {{generations}} --hands-per-gen {{hands}}

# Train 5,000 rounds in King-of-the-Hill mode and then update the Leaderboard
train-5000:
    {{python}} main.py --mode challenge-leader --generations 5 --hands-per-gen 1000 --epochs 5
    {{python}} main.py --mode leaderboard --tournament-rounds 8 --reset-leaderboard

# Train initial Base Model from scratch (MCCFR + Distillation + Self-Play RL)
train-base cfr="2000" hands="2000" epochs="8" selfplay="1000":
    {{python}} main.py --mode train --cfr-iters {{cfr}} --distill-hands {{hands}} --epochs {{epochs}} --self-play-episodes {{selfplay}}

# Play an interactive 15-point Argentine Truco match in the terminal against the System 1 SLM
play:
    {{python}} main.py --mode play

# Launch the React Web UI & REST API Server (Leaderboard, King-of-the-Hill Trainer, and Video Game Replay Theater)
webui port="8000":
    {{python}} server.py --port {{port}}

# Run the unit test suite
test:
    {{python}} -m unittest discover -s tests -v
