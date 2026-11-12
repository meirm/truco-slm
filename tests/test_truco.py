"""
Unit tests verifying Argentine Truco rules (40-card power hierarchy, Envido calculation,
"El Envido está primero", Truco/Retruco/Vale Cuatro ownership, and Parda tie-breaker rules)
and System 1 SLM single-pass inference.
"""

from __future__ import annotations
import unittest
import torch

from truco.engine import (
    Action,
    Card,
    compute_envido,
    DECK,
    NUM_ACTIONS,
    Suit,
    TrucoHandState,
)
from truco.model import System1TrucoSLM
from truco.rlhf import InteractivePlaySession
from truco.tokenizer import compute_opponent_belief, TrucoTokenizer


def get_card(rank: int, suit: Suit) -> Card:
    for c in DECK:
        if c.rank == rank and c.suit == suit:
            return c
    raise ValueError(f"Card {rank} {suit} not found")


class TestArgentineTrucoEngine(unittest.TestCase):
    def test_deck_and_power_hierarchy(self) -> None:
        self.assertEqual(len(DECK), 40)
        ancho_espada = get_card(1, Suit.ESPADA)
        ancho_basto = get_card(1, Suit.BASTO)
        siete_espada = get_card(7, Suit.ESPADA)
        siete_oro = get_card(7, Suit.ORO)
        tres = get_card(3, Suit.COPA)
        dos = get_card(2, Suit.ORO)
        ancho_falso = get_card(1, Suit.COPA)
        rey = get_card(12, Suit.ESPADA)
        siete_falso = get_card(7, Suit.COPA)
        cuatro = get_card(4, Suit.BASTO)

        self.assertTrue(
            ancho_espada.truco_power
            > ancho_basto.truco_power
            > siete_espada.truco_power
            > siete_oro.truco_power
            > tres.truco_power
            > dos.truco_power
            > ancho_falso.truco_power
            > rey.truco_power
            > siete_falso.truco_power
            > cuatro.truco_power
        )

    def test_envido_calculation(self) -> None:
        # 7 and 6 of Espada -> 20 + 7 + 6 = 33
        h1 = [get_card(7, Suit.ESPADA), get_card(6, Suit.ESPADA), get_card(1, Suit.BASTO)]
        self.assertEqual(compute_envido(h1), 33)

        # Figure cards (10, 11, 12) count as 0 in Envido -> 12 and 10 of Oro = 20
        h2 = [get_card(12, Suit.ORO), get_card(10, Suit.ORO), get_card(4, Suit.COPA)]
        self.assertEqual(compute_envido(h2), 20)

        # Three different suits -> highest non-figure card
        h3 = [get_card(7, Suit.COPA), get_card(12, Suit.ESPADA), get_card(5, Suit.BASTO)]
        self.assertEqual(compute_envido(h3), 7)

    def test_envido_va_primero_when_truco_called(self) -> None:
        """If Player 0 calls Truco on Trick 0 before Player 1 plays, Player 1 can call Envido first."""
        h0 = [get_card(1, Suit.ESPADA), get_card(3, Suit.ORO), get_card(4, Suit.COPA)]
        h1 = [get_card(7, Suit.ESPADA), get_card(6, Suit.ESPADA), get_card(2, Suit.BASTO)]
        state = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))

        # Player 0 immediately calls Truco
        state.step(Action.CALL_TRUCO)
        self.assertEqual(state.current_player, 1)
        legal = state.legal_actions()
        self.assertIn(Action.CALL_ENVIDO, legal)

        # Player 1 responds with Envido ("Envido está primero"), overriding the unaccepted Truco
        state.step(Action.CALL_ENVIDO)
        self.assertEqual(state.current_player, 0)
        self.assertEqual(state.truco_level, 0)

        # Player 0 accepts Envido; Player 1 wins Envido (33 vs 4) -> +2 pts for Player 1
        state.step(Action.QUIERO)
        self.assertTrue(state.envido_resolved)
        self.assertEqual(state.points_won, [0, 2])
        # Turn returns to Player 0 (who was on turn before calling Truco)
        self.assertEqual(state.current_player, 0)

    def test_parda_tiebreaker_rules(self) -> None:
        """If Trick 1 is won by P0 and Trick 2 is Parda (tie), P0 immediately wins the hand."""
        h0 = [get_card(1, Suit.ESPADA), get_card(3, Suit.ORO), get_card(4, Suit.COPA)]
        h1 = [get_card(7, Suit.ORO), get_card(3, Suit.COPA), get_card(4, Suit.BASTO)]
        state = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))

        # Trick 1: P0 plays 1E (slot 0), P1 plays 7O (slot 0) -> P0 wins Trick 1
        state.step(Action.PLAY_CARD_0)
        state.step(Action.PLAY_CARD_0)
        self.assertFalse(state.is_terminal)
        self.assertEqual(state.trick_winners, [0])

        # Trick 2: P0 plays 3O (slot 1), P1 plays 3C (slot 1) -> Parda! P0 wins hand!
        state.step(Action.PLAY_CARD_1)
        state.step(Action.PLAY_CARD_1)
        self.assertTrue(state.is_terminal)
        self.assertEqual(state.truco_winner, 0)

    def test_system1_slm_forward_and_masking(self) -> None:
        tokenizer = TrucoTokenizer()
        model = System1TrucoSLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=2, n_layers=2, d_ff=128)
        device = torch.device("cpu")
        state = TrucoHandState.new_hand(mano=0)

        action, probs, val = model.act(state, tokenizer, device, temperature=0.0)
        self.assertIn(action, state.legal_actions())
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
        self.assertIsInstance(val, float)

    def test_bayesian_belief_and_hindsight_reveal(self) -> None:
        tokenizer = TrucoTokenizer()
        h0 = [get_card(1, Suit.ESPADA), get_card(7, Suit.ESPADA), get_card(6, Suit.ESPADA)]
        h1 = [get_card(4, Suit.COPA), get_card(5, Suit.ORO), get_card(6, Suit.BASTO)]
        state = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))

        # Player 0 holds 1E (strongest card in deck) and 33 Envido -> P(beat opp best)=1.0, P(win envido)=1.0
        b0 = compute_opponent_belief(state, 0)
        self.assertEqual(b0["p_beat_opp_best"], 1.0)
        self.assertEqual(b0["win_token"], "BELIEF_WIN_4")
        self.assertGreaterEqual(b0["p_win_envido"], 0.95)
        self.assertEqual(b0["env_token"], "BELIEF_ENV_4")

        # Verify belief tokens are injected into prompt before <ACT>
        prompt = tokenizer.format_prompt_text(state, 0)
        self.assertIn("BELIEF_WIN_4", prompt)
        self.assertIn("BELIEF_ENV_4", prompt)

    def test_tactical_personalities_and_trap_smelling(self) -> None:
        from truco.leaderboard import act_with_personality

        tokenizer = TrucoTokenizer()
        model = System1TrucoSLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=2, n_layers=2, d_ff=128)
        device = torch.device("cpu")

        # Trick 0: Player 0 has 1E, 7E, 4C; Player 1 has 3O, 2B, 4B
        h0 = [get_card(1, Suit.ESPADA), get_card(7, Suit.ESPADA), get_card(4, Suit.COPA)]
        h1 = [get_card(3, Suit.ORO), get_card(2, Suit.BASTO), get_card(4, Suit.BASTO)]
        state = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))

        for persona in ("standard", "bluffer", "phisher", "sheriff", "risk_taker"):
            act = act_with_personality(model, state, tokenizer, device, persona=persona, temperature=0.0)
            self.assertIn(act, state.legal_actions())

        # Simulate Player 0 playing weak card 4C on Trick 0 (PLAY_CARD_2) and Player 1 winning with 2B (PLAY_CARD_1)
        state.step(Action.PLAY_CARD_2)
        state.step(Action.PLAY_CARD_1)
        # Now in Trick 1, Player 1 observes that Player 0 threw a 4 on Trick 0 and stayed in -> trap suspect!
        b1 = compute_opponent_belief(state, 1)
        self.assertGreater(b1["p_trap_suspect"], 0.35)

    def test_hide_and_reveal_opponent_identity(self) -> None:
        tokenizer = TrucoTokenizer()
        session = InteractivePlaySession(torch.device("cpu"), tokenizer)
        data = session.start_match("mystery", hide_identity=True)

        self.assertTrue(data["hide_identity"])
        self.assertTrue(data["opponent"]["is_masked"])
        self.assertFalse(data["opponent"]["revealed"])
        self.assertEqual(data["opponent"]["name"], "🕵️ Mystery Opponent")
        self.assertEqual(data["opponent"]["elo"], "????")

        # Simulate match completion -> identity must be revealed!
        session.match_scores = [15, 11]
        session.match_over = True
        session.match_winner = 0
        end_data = session.serialize_session()
        self.assertFalse(end_data["opponent"]["is_masked"])
        self.assertTrue(end_data["opponent"]["revealed"])
        self.assertNotEqual(end_data["opponent"]["name"], "🕵️ Mystery Opponent")
        self.assertIsInstance(end_data["opponent"]["elo"], int)

    def test_son_buenas_and_envido_to_truco_card_deduction(self) -> None:
        # Mano (P0) has 7E, 6E, 4C (Envido = 33). Pie (P1) has 2O, 5O, 4B (Envido = 27).
        h0 = [get_card(7, Suit.ESPADA), get_card(6, Suit.ESPADA), get_card(4, Suit.COPA)]
        h1 = [get_card(2, Suit.ORO), get_card(5, Suit.ORO), get_card(4, Suit.BASTO)]
        state = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))

        # Mano calls Envido, Pie accepts (QUIERO) -> Mano declares 33 first; Pie says "Son buenas" (score hidden!)
        state.step(Action.CALL_ENVIDO)
        state.step(Action.QUIERO)

        self.assertEqual(state.envido_revealed_scores[0], 33)
        self.assertIsNone(state.envido_revealed_scores[1])
        self.assertTrue(state.envido_son_buenas)

        # Now Mano plays 6 de Espada (slot 1 in sorted h0: [7E, 6E, 4C]).
        # Because Mano declared 33 Envido and played 6E, 36/38 (94.7%) of valid hands contain 7E or 7O!
        state.step(Action.PLAY_CARD_1)
        b_pie = compute_opponent_belief(state, 1)
        self.assertEqual(b_pie["opp_declared_envido"], 33)
        self.assertGreaterEqual(b_pie["p_opp_monster"], 0.94)
        self.assertIn("7E", str(b_pie["envido_card_hint"]))

        # Pie wins Trick 0 with 2O (slot 0), then plays 5O (slot 1), and Mano plays 4C (slot 2).
        # Wait: before Trick 1 resolves, once Mano has played both 6E and 4C with declared Envido 33,
        # the ONLY card in the 40-card deck that gives 33 with (6E, 4C) is 7E (100% certainty)!
        state2 = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))
        state2.step(Action.CALL_ENVIDO)
        state2.step(Action.QUIERO)
        state2.step(Action.PLAY_CARD_2)  # Mano plays 4C on Trick 0
        state2.step(Action.PLAY_CARD_0)  # Pie wins Trick 0 with 2O
        state2.step(Action.PLAY_CARD_1)  # Pie leads Trick 1 with 5O
        state2.step(Action.PLAY_CARD_1)  # Mano plays 6E on Trick 1 -> Trick 2 starts!
        b_pie_t2 = compute_opponent_belief(state2, 1)
        self.assertEqual(b_pie_t2["p_opp_monster"], 1.0)
        self.assertIn("1 opción(es) (7E)", str(b_pie_t2["envido_card_hint"]))

    def test_envido_21_plus_played_12E_hypotheses(self) -> None:
        """
        Verify that when Mano declares 21 in Envido and plays 12 de Espada (12E),
        compute_opponent_belief decomposes the conjectures combining both the declared
        Envido (21) and the played card (12E):
          1. Liga con 12E: holds 1E + 1 free card (which can include 1B / another Monster)
          2. Par oculto de Basto (12E suelta): holds 1B + 10B/11B/12B
          3. Par oculto de Copa/Oro (12E suelta): holds 1O/1C + 10/11/12 (worth less in Truco)
        """
        by_short = {c.short_str(): c for c in DECK}
        # Mano holds 12E, 1B, 11B -> Envido = 21 (from 1B + 11B), plays 12E first
        h0 = (by_short["12E"], by_short["1B"], by_short["11B"])
        # Pie holds 3C, 6O, 4B -> Envido = 6
        h1 = (by_short["3C"], by_short["6O"], by_short["4B"])
        state = TrucoHandState.new_hand(mano=0, dealt_cards=(h0, h1))
        state.step(Action.CALL_ENVIDO)
        state.step(Action.QUIERO)
        self.assertEqual(state.envido_revealed_scores[0], 21)

        # Mano plays 12E on Trick 0 (sorted hand is [1B, 12E, 11B], so 12E is slot 1)
        state.step(Action.PLAY_CARD_1)
        belief = compute_opponent_belief(state, 1)
        hyps = belief["envido_hypotheses"]
        self.assertEqual(len(hyps), 3)
        self.assertIn("Liga con 12E", hyps[0])
        self.assertIn("1E", hyps[0])
        self.assertIn("Par oculto de Basto", hyps[1])
        self.assertIn("1B+10B, 1B+11B, 1B+12B", hyps[1])
        self.assertIn("Par oculto de Copa/Oro", hyps[2])
        self.assertIn("valen menos en el Truco", hyps[2])

        # Also verify the System 1 tokenizer includes BY_OPP ENV_21 alongside CARD_12E
        tok = TrucoTokenizer()
        prompt = tok.format_prompt_text(state, 1)
        self.assertIn("CARD_12E", prompt)
        self.assertIn("ENV_21", prompt)

        # Now if Pie wins Trick 0 with 3C and leads 6O on Trick 1, and Mano plays 11B:
        # Mano has played 12E and 11B with declared Envido 21 -> last card MUST be 1E or 1B!
        state.step(Action.PLAY_CARD_0)  # Pie wins Trick 0 with 3C
        state.step(Action.PLAY_CARD_1)  # Pie leads 6O on Trick 1
        state.step(Action.PLAY_CARD_2)  # Mano plays 11B on Trick 1
        belief_t2 = compute_opponent_belief(state, 1)
        self.assertEqual(belief_t2["p_opp_monster"], 1.0)
        self.assertIn("2 opción(es) (1E, 1B)", str(belief_t2["envido_card_hint"]))

    def test_leaderboard_pruning_and_restore(self) -> None:
        import os
        import tempfile
        from truco.leaderboard import LeaderboardManager, build_roster

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "lb.json")
            lb = LeaderboardManager(json_path)
            tok = TrucoTokenizer()
            dev = torch.device("cpu")
            roster = build_roster(lb, tok, dev)
            self.assertGreaterEqual(len(roster), 8)

            # Prune to Core 8
            lb.prune_to_core_8()
            roster_8 = build_roster(lb, tok, dev)
            self.assertEqual(len(roster_8), 8)
            self.assertEqual(len(lb.players), 8)

            # Prune to Top 5
            lb.prune_to_top_k(5)
            self.assertEqual(len(lb.players), 5)

            # Remove one specific player
            first_pid = lb.get_ranked_players()[0].player_id
            lb.remove_player(first_pid)
            self.assertEqual(len(lb.players), 4)

            # Restore all
            lb.restore_all_players()
            roster_restored = build_roster(lb, tok, dev)
            self.assertGreaterEqual(len(roster_restored), 8)

    def test_cannot_call_envido_after_truco_accepted(self) -> None:
        """
        Verify that if Player 0 calls Truco on Trick 0 and Player 1 accepts (QUIERO)
        without invoking 'El Envido está primero', the Envido window is permanently closed:
        neither Player 0 nor Player 1 can call Envido afterwards.
        """
        state = TrucoHandState.new_hand(mano=0)
        # Mano (0) calls Truco immediately
        state.step(Action.CALL_TRUCO)
        # Pie (1) CAN invoke 'El Envido está primero' right now while Truco is pending
        self.assertIn(Action.CALL_ENVIDO, state.legal_actions())
        # Instead, Pie accepts Truco directly with QUIERO
        state.step(Action.QUIERO)
        self.assertTrue(state.envido_resolved)
        # Now it's Mano's (0) turn to play their first card; Mano CANNOT call Envido after calling Truco!
        self.assertEqual(state.current_player, 0)
        self.assertNotIn(Action.CALL_ENVIDO, state.legal_actions())
        self.assertNotIn(Action.CALL_REAL_ENVIDO, state.legal_actions())
        self.assertNotIn(Action.CALL_FALTA_ENVIDO, state.legal_actions())
        # Mano plays a card; now it's Pie's (1) turn; Pie ALSO cannot call Envido!
        state.step(Action.PLAY_CARD_0)
        self.assertEqual(state.current_player, 1)
        self.assertNotIn(Action.CALL_ENVIDO, state.legal_actions())


if __name__ == "__main__":
    unittest.main()


