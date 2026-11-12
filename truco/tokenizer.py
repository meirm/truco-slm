"""
State serialization and custom vocabulary tokenizer for the System 1 Truco SLM.

Encodes a player's imperfect-information observation (Information Set) into a compact,
fixed-structure sequence of discrete tokens so a causal Transformer can emit an action
in a single forward pass (zero CoT / System 1 policy + value head).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from truco.engine import (
    Action,
    ACTION_NAMES,
    Card,
    compute_envido,
    DECK,
    NUM_ACTIONS,
    PendingState,
    Suit,
    SUIT_NAMES,
    TrucoHandState,
)


class TrucoTokenizer:
    """
    Compact domain-specific vocabulary for Argentine Truco information states.
    Total vocabulary size is < 160 tokens, enabling sub-millisecond embedding lookup.
    """

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special structural tokens
        special = [
            "<PAD>",
            "<BOS>",
            "<SEP>",
            "<ACT>",  # Final query token whose hidden state feeds policy & value heads
            "POS_MANO",
            "POS_PIE",
            "SLOT_PLAYED",
            "SLOT_AVAIL",
            "PEND_NONE",
            "PEND_ENVIDO",
            "PEND_TRUCO",
            "OWNER_NONE",
            "OWNER_SELF",
            "OWNER_OPP",
            "TRICK_0",
            "TRICK_1",
            "TRICK_2",
            "WIN_SELF",
            "WIN_OPP",
            "WIN_PARDA",
            "WIN_NONE",
            "BY_SELF",
            "BY_OPP",
        ]
        for tok in special:
            self._add(tok)

        # 40 Card tokens: e.g. CARD_1E, CARD_7O, etc.
        for card in DECK:
            self._add(f"CARD_{card.short_str()}")

        # Envido score tokens (0..33)
        for s in range(34):
            self._add(f"ENV_{s}")

        # Truco level tokens (0..3) and stakes (1..4)
        for lvl in range(4):
            self._add(f"TLVL_{lvl}")
        for stk in range(1, 5):
            self._add(f"TSTK_{stk}")

        # Match score bucket tokens (points remaining to target: 1..15+)
        for rem in range(1, 16):
            self._add(f"SELF_REM_{rem}")
            self._add(f"OPP_REM_{rem}")

        # Action history tokens
        for act in Action:
            self._add(f"HIST_ACT_{act.name}")

        # Bayesian Hidden-Card Belief Tokens (appended at end of vocabulary so older 147-token checkpoints upgrade seamlessly)
        # - BELIEF_WIN_0..4   : Probability AI's best unplayed card beats opponent's best hidden card (20% buckets)
        # - BELIEF_THREAT_0..4: Probability opponent holds at least one Monster card (3, 7O, 7E, 1B, 1E; power >= 10)
        # - BELIEF_ENV_0..4   : Probability AI wins the Envido showdown given unseen deck + opponent betting actions
        for b in range(5):
            self._add(f"BELIEF_WIN_{b}")
            self._add(f"BELIEF_THREAT_{b}")
            self._add(f"BELIEF_ENV_{b}")

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.act_id = self.token_to_id["<ACT>"]
        self.max_seq_len = 64

    def _add(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_observation(self, state: TrucoHandState, player: Optional[int] = None) -> List[int]:
        """
        Converts `state` from `player`'s perspective into a list of token IDs ending with `<ACT>`.
        Strictly uses only imperfect information visible to `player`, augmented with Bayesian
        hidden-card belief tokens computed from the unseen deck and opponent's public actions.
        """
        p = state.current_player if player is None else player
        opp = 1 - p

        tokens: List[str] = ["<BOS>"]

        # 1. Position (Mano or Pie) and score remaining
        tokens.append("POS_MANO" if state.mano == p else "POS_PIE")
        self_rem = max(1, min(15, state.target_score - state.match_scores[p]))
        opp_rem = max(1, min(15, state.target_score - state.match_scores[opp]))
        tokens.append(f"SELF_REM_{self_rem}")
        tokens.append(f"OPP_REM_{opp_rem}")

        # 2. Private hand (3 slots in order + whether each slot is still available) & own Envido score
        for slot in range(3):
            card = state.hands[p][slot]
            tokens.append(f"CARD_{card.short_str()}")
            tokens.append("SLOT_PLAYED" if state.played_mask[p][slot] else "SLOT_AVAIL")
        tokens.append(f"ENV_{state.envido_scores[p]}")
        tokens.append("<SEP>")

        # 3. Current betting state
        if state.pending == PendingState.NONE:
            tokens.append("PEND_NONE")
        elif state.pending == PendingState.ENVIDO:
            tokens.append("PEND_ENVIDO")
        else:
            tokens.append("PEND_TRUCO")

        tokens.append(f"TLVL_{state.truco_level}")
        tokens.append(f"TSTK_{state.truco_stakes}")
        if state.truco_owner == -1:
            tokens.append("OWNER_NONE")
        elif state.truco_owner == p:
            tokens.append("OWNER_SELF")
        else:
            tokens.append("OWNER_OPP")
        tokens.append("<SEP>")

        # 4. Trick outcomes so far
        for t_idx in range(3):
            tokens.append(f"TRICK_{t_idx}")
            if t_idx < len(state.trick_winners):
                w = state.trick_winners[t_idx]
                if w == p:
                    tokens.append("WIN_SELF")
                elif w == opp:
                    tokens.append("WIN_OPP")
                else:
                    tokens.append("WIN_PARDA")
            else:
                tokens.append("WIN_NONE")
        tokens.append("<SEP>")

        # 5. Public action & card history (truncated if needed to fit max_seq_len)
        opp_declared_env = state.envido_revealed_scores[opp]
        hist_slice = state.history[-13:] if opp_declared_env is not None else state.history[-14:]
        for actor, act, card in hist_slice:
            tokens.append("BY_SELF" if actor == p else "BY_OPP")
            tokens.append(f"HIST_ACT_{act.name}")
            if card is not None:
                tokens.append(f"CARD_{card.short_str()}")
        if opp_declared_env is not None:
            tokens.append("BY_OPP")
            tokens.append(f"ENV_{opp_declared_env}")

        # 6. Bayesian Hidden-Card Belief Tokens (evaluated in System 1 before <ACT>)
        belief = compute_opponent_belief(state, p)
        tokens.append(str(belief["win_token"]))
        tokens.append(str(belief["threat_token"]))
        tokens.append(str(belief["env_token"]))

        tokens.append("<ACT>")
        return [self.token_to_id[t] for t in tokens]

    def format_prompt_text(self, state: TrucoHandState, player: Optional[int] = None) -> str:
        """Human-readable representation of the exact System 1 observation prompt."""
        ids = self.encode_observation(state, player)
        return " ".join(self.id_to_token[i] for i in ids)


def compute_opponent_belief(state: TrucoHandState, observer: Optional[int] = None) -> Dict[str, object]:
    """
    Fast Bayesian Hidden-Card Belief Tracker for Imperfect-Information Argentine Truco.

    Uses strictly public information + `observer`'s own 3 cards to estimate:
    1. `p_beat_opp_best`: Probability `observer`'s strongest remaining card beats or ties
       the opponent's strongest remaining hidden card.
    2. `p_opp_monster`  : Probability the opponent holds at least one Monster card
       (`3`, `7 de Oro`, `7 de Espada`, `1 de Basto`, `1 de Espada` -> `truco_power >= 10`)
       among their remaining unplayed cards.
    3. `p_win_envido`   : Probability `observer`'s Envido score beats the opponent's Envido score,
       conditioned on known cards and whether the opponent called/escalated Envido.
    """
    p = state.current_player if observer is None else observer
    opp = 1 - p

    # 1. Identify all cards known to `p` (own 3 cards + opponent's publicly played cards)
    known_ids = {c.card_id for c in state.hands[p]}
    opp_played_cards: List[Card] = []
    for trick in state.tricks:
        for tc in trick:
            if tc.player == opp:
                known_ids.add(tc.card.card_id)
                opp_played_cards.append(tc.card)

    unseen_cards = [c for c in DECK if c.card_id not in known_ids]
    n_unseen = len(unseen_cards)  # 34..37
    opp_hidden_count = sum(1 for played in state.played_mask[opp] if not played)

    own_unplayed_powers = [
        state.hands[p][s].truco_power for s in range(3) if not state.played_mask[p][s]
    ]
    my_best_power = max(own_unplayed_powers) if own_unplayed_powers else 0

    p_trap_suspect = 0.0
    p_bluff_suspect = 0.0
    opp_declared_env: Optional[int] = state.envido_revealed_scores[opp]
    opp_said_son_buenas: bool = bool(state.envido_son_buenas and opp != state.mano)
    envido_card_hint: Optional[str] = None
    envido_hypotheses: List[str] = []

    if opp_hidden_count == 0 or n_unseen == 0:
        p_beat = 1.0
        p_monster = 0.0
    else:
        # Combinatorial base probabilities from the unseen deck
        cards_not_beating_me = sum(1 for c in unseen_cards if c.truco_power <= my_best_power)
        cards_below_monster = sum(1 for c in unseen_cards if c.truco_power < 10)

        # P(all k hidden opponent cards are <= my_best_power)
        prob_all_le = 1.0
        prob_no_monster = 1.0
        for i in range(opp_hidden_count):
            denom = max(1, n_unseen - i)
            prob_all_le *= max(0.0, float(cards_not_beating_me - i) / denom)
            prob_no_monster *= max(0.0, float(cards_below_monster - i) / denom)

        p_beat = prob_all_le
        p_monster = 1.0 - prob_no_monster

        # Exact Envido-to-Truco Card Filtering when Opponent Declared Envido (or said "Son buenas")!
        if opp_declared_env is not None or opp_said_son_buenas:
            mano_env = state.envido_scores[state.mano]
            valid_tuples: List[Tuple[Card, ...]] = []
            tuple_weights: List[float] = []

            if opp_hidden_count == 1 and len(opp_played_cards) == 2:
                c0, c1 = opp_played_cards[0], opp_played_cards[1]
                played_already_make_env = (
                    opp_declared_env is not None
                    and c0.suit == c1.suit
                    and 20 + c0.envido_value + c1.envido_value == opp_declared_env
                )
                linked_with_c0: List[Card] = []
                linked_with_c1: List[Card] = []
                free_third: List[Card] = []

                for c2 in unseen_cards:
                    env_v = compute_envido([c0, c1, c2])
                    if (opp_declared_env is not None and env_v == opp_declared_env) or (
                        opp_said_son_buenas and env_v <= mano_env
                    ):
                        valid_tuples.append((c2,))
                        tuple_weights.append(1.0)
                        if opp_declared_env is not None and opp_declared_env >= 20:
                            if played_already_make_env:
                                free_third.append(c2)
                            elif c2.suit == c0.suit and 20 + c0.envido_value + c2.envido_value == opp_declared_env:
                                linked_with_c0.append(c2)
                            elif c2.suit == c1.suit and 20 + c1.envido_value + c2.envido_value == opp_declared_env:
                                linked_with_c1.append(c2)

                if opp_declared_env is not None and valid_tuples:
                    if played_already_make_env:
                        monsters_left = [c.short_str() for c in free_third if c.truco_power >= 10]
                        envido_hypotheses.append(
                            f"Ya formó {opp_declared_env} con sus 2 jugadas ({c0.short_str()}+{c1.short_str()}): "
                            f"la 3ra carta es libre ({len(free_third)} posibles, incl. {len(monsters_left)} bravas)"
                        )
                    else:
                        if linked_with_c0:
                            s0 = ", ".join(c.short_str() for c in linked_with_c0)
                            envido_hypotheses.append(
                                f"Liga {opp_declared_env} con {c0.short_str()} jugada → tiene {s0}"
                            )
                        if linked_with_c1:
                            s1 = ", ".join(c.short_str() for c in linked_with_c1)
                            envido_hypotheses.append(
                                f"Liga {opp_declared_env} con {c1.short_str()} jugada → tiene {s1}"
                            )

            elif opp_hidden_count == 2 and len(opp_played_cards) == 1:
                c0 = opp_played_cards[0]
                if opp_declared_env is not None and opp_declared_env >= 20:
                    # Either (c0, c1) forms the Envido pair, or (c1, c2) forms the same-suit Envido pair
                    by_suit: Dict[int, List[Card]] = {0: [], 1: [], 2: [], 3: []}
                    for c in unseen_cards:
                        by_suit[int(c.suit)].append(c)

                    seen_pair_ids = set()
                    subcase_a_matching_c1: List[Card] = []
                    subcase_a_tuples: List[Tuple[Card, Card]] = []
                    subcase_a_weights: List[float] = []
                    subcase_b_by_suit: Dict[int, List[Tuple[Card, Card]]] = {0: [], 1: [], 2: [], 3: []}
                    subcase_b_weights_by_suit: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

                    # Subcase A: c1 shares suit with played card c0 and 20 + c0.env + c1.env == opp_declared_env
                    for c1 in by_suit[int(c0.suit)]:
                        if 20 + c0.envido_value + c1.envido_value == opp_declared_env:
                            subcase_a_matching_c1.append(c1)
                            for c2 in unseen_cards:
                                if c2.card_id == c1.card_id:
                                    continue
                                if compute_envido([c0, c1, c2]) == opp_declared_env:
                                    pair_key = (min(c1.card_id, c2.card_id), max(c1.card_id, c2.card_id))
                                    if pair_key not in seen_pair_ids:
                                        seen_pair_ids.add(pair_key)
                                        # Play-choice weight: if c2 is a high card (e.g. 1B/7E/7O/3) and c1 is also high,
                                        # playing c0 (e.g. 12E) first is a natural lead (weight 1.8 vs 1.0)
                                        w_a = 1.8 if c2.truco_power >= 10 else 1.0
                                        valid_tuples.append((c1, c2))
                                        tuple_weights.append(w_a)
                                        subcase_a_tuples.append((c1, c2))
                                        subcase_a_weights.append(w_a)

                    # Subcase B: (c1, c2) share a suit and 20 + c1.env + c2.env == opp_declared_env (c0 was the loose card)
                    for s_id, s_cards in by_suit.items():
                        ns = len(s_cards)
                        for i in range(ns):
                            for j in range(i + 1, ns):
                                c1, c2 = s_cards[i], s_cards[j]
                                if 20 + c1.envido_value + c2.envido_value == opp_declared_env:
                                    if compute_envido([c0, c1, c2]) == opp_declared_env:
                                        pair_key = (min(c1.card_id, c2.card_id), max(c1.card_id, c2.card_id))
                                        if pair_key not in seen_pair_ids:
                                            seen_pair_ids.add(pair_key)
                                            # Bayesian play-choice likelihood: when holding a 2-card Envido pair in another
                                            # suit plus a competitive loose card c0 (e.g. 12E, power >= 6), players lead c0
                                            # with higher conditional frequency (~2.5x) to fight Primera without breaking their pair.
                                            w_b = 2.5 if c0.truco_power >= 6 else 1.0
                                            valid_tuples.append((c1, c2))
                                            tuple_weights.append(w_b)
                                            subcase_b_by_suit[s_id].append((c1, c2))
                                            subcase_b_weights_by_suit[s_id] += w_b

                    total_w = sum(tuple_weights) or 1.0
                    # Build tactical hypotheses combining played card c0 + declared Envido
                    if subcase_a_tuples:
                        w_a_sum = sum(subcase_a_weights)
                        pct_a = round(w_a_sum / total_w * 100)
                        c1_names = "/".join(sorted({c.short_str() for c in subcase_a_matching_c1}))
                        # Check if 3rd free card c2 can also be a Monster (e.g., 1B, 7E, 7O, 3)
                        extra_monsters = sorted(
                            {
                                (c2.short_str() if c1 in subcase_a_matching_c1 else c1.short_str())
                                for c1, c2 in subcase_a_tuples
                                if (c2 if c1 in subcase_a_matching_c1 else c1).truco_power >= 10
                            }
                        )
                        extra_note = (
                            f" + 3ra libre (puede incluir otra brava como {'/'.join(extra_monsters[:3])})"
                            if extra_monsters
                            else " + 3ra carta libre"
                        )
                        envido_hypotheses.append(
                            f"Liga con {c0.short_str()} ({pct_a}%): tiene {c1_names}{extra_note}"
                        )

                    # High-value other suits (Espada / Basto, or pairs with 7O) vs lower-value suits (Oro / Copa without 7O)
                    for s_id in (Suit.ESPADA, Suit.BASTO):
                        pairs_s = subcase_b_by_suit[int(s_id)]
                        if pairs_s:
                            pct_s = round(subcase_b_weights_by_suit[int(s_id)] / total_w * 100)
                            sample_pairs = ", ".join(f"{a.short_str()}+{b.short_str()}" for a, b in pairs_s[:3])
                            envido_hypotheses.append(
                                f"Par oculto de {SUIT_NAMES[s_id]} ({pct_s}%, {c0.short_str()} suelta): "
                                f"{sample_pairs} (cartas fuertes para el Truco)"
                            )

                    low_suits_pairs: List[Tuple[Card, Card]] = []
                    low_suits_w = 0.0
                    high_oro_pairs: List[Tuple[Card, Card]] = []
                    high_oro_w = 0.0
                    w_b_unit = 2.5 if c0.truco_power >= 6 else 1.0
                    for s_id in (Suit.ORO, Suit.COPA):
                        for a, b in subcase_b_by_suit[int(s_id)]:
                            if max(a.truco_power, b.truco_power) >= 10:
                                high_oro_pairs.append((a, b))
                                high_oro_w += w_b_unit
                            else:
                                low_suits_pairs.append((a, b))
                                low_suits_w += w_b_unit

                    if high_oro_pairs:
                        pct_ho = round(high_oro_w / total_w * 100)
                        sample_ho = ", ".join(f"{a.short_str()}+{b.short_str()}" for a, b in high_oro_pairs[:3])
                        envido_hypotheses.append(
                            f"Par oculto bravo en Oro/Copa ({pct_ho}%, {c0.short_str()} suelta): {sample_ho}"
                        )
                    if low_suits_pairs:
                        pct_low = round(low_suits_w / total_w * 100)
                        sample_low = ", ".join(f"{a.short_str()}+{b.short_str()}" for a, b in low_suits_pairs[:3])
                        envido_hypotheses.append(
                            f"Par oculto de Copa/Oro ({pct_low}%, {c0.short_str()} suelta): "
                            f"{sample_low} (valen menos en el Truco)"
                        )
                else:
                    # Low Envido (<20) or "Son buenas" upper bound
                    nu = len(unseen_cards)
                    for i in range(nu):
                        for j in range(i + 1, nu):
                            c1, c2 = unseen_cards[i], unseen_cards[j]
                            env_v = compute_envido([c0, c1, c2])
                            if (opp_declared_env is not None and env_v == opp_declared_env) or (
                                opp_said_son_buenas and env_v <= mano_env
                            ):
                                valid_tuples.append((c1, c2))
                                tuple_weights.append(1.0)

            elif opp_hidden_count == 3 and opp_declared_env is not None:
                if opp_declared_env >= 20:
                    # 2 of the 3 hidden cards MUST share a suit and sum to opp_declared_env - 20
                    by_suit = {0: [], 1: [], 2: [], 3: []}
                    for c in unseen_cards:
                        by_suit[int(c.suit)].append(c)
                    cores_by_suit: Dict[int, List[Tuple[Card, Card]]] = {0: [], 1: [], 2: [], 3: []}
                    cores: List[Tuple[Card, Card]] = []
                    for s_id, s_cards in by_suit.items():
                        ns = len(s_cards)
                        for i in range(ns):
                            for j in range(i + 1, ns):
                                if 20 + s_cards[i].envido_value + s_cards[j].envido_value == opp_declared_env:
                                    cores.append((s_cards[i], s_cards[j]))
                                    cores_by_suit[s_id].append((s_cards[i], s_cards[j]))
                    for c1, c2 in cores:
                        for c3 in unseen_cards:
                            if c3.card_id == c1.card_id or c3.card_id == c2.card_id:
                                continue
                            if c3.suit == c1.suit and (
                                20 + c1.envido_value + c3.envido_value > opp_declared_env
                                or 20 + c2.envido_value + c3.envido_value > opp_declared_env
                            ):
                                continue
                            valid_tuples.append((c1, c2, c3))
                            tuple_weights.append(1.0)

                    total_cores = len(cores) or 1
                    for s_id in (Suit.ESPADA, Suit.BASTO):
                        cs = cores_by_suit[int(s_id)]
                        if cs:
                            pct_c = round(len(cs) / total_cores * 100)
                            sample_c = ", ".join(f"{a.short_str()}+{b.short_str()}" for a, b in cs[:3])
                            envido_hypotheses.append(
                                f"Envido {opp_declared_env} en {SUIT_NAMES[s_id]} ({pct_c}%): {sample_c} + 3ra libre"
                            )
                    oro_copa_cs = cores_by_suit[int(Suit.ORO)] + cores_by_suit[int(Suit.COPA)]
                    if oro_copa_cs:
                        pct_oc = round(len(oro_copa_cs) / total_cores * 100)
                        sample_oc = ", ".join(f"{a.short_str()}+{b.short_str()}" for a, b in oro_copa_cs[:3])
                        envido_hypotheses.append(
                            f"Envido {opp_declared_env} en Oro/Copa ({pct_oc}%): {sample_oc} + 3ra libre"
                        )
                else:
                    # opp_declared_env < 20: all 3 cards have different suits, max(envido_value) == opp_declared_env
                    eligible = [c for c in unseen_cards if c.envido_value <= opp_declared_env]
                    by_s = {0: [], 1: [], 2: [], 3: []}
                    for c in eligible:
                        by_s[int(c.suit)].append(c)
                    suits_list = [0, 1, 2, 3]
                    for a in range(4):
                        for b in range(a + 1, 4):
                            for d in range(b + 1, 4):
                                for c1 in by_s[suits_list[a]]:
                                    for c2 in by_s[suits_list[b]]:
                                        for c3 in by_s[suits_list[d]]:
                                            if max(c1.envido_value, c2.envido_value, c3.envido_value) == opp_declared_env:
                                                valid_tuples.append((c1, c2, c3))
                                                tuple_weights.append(1.0)

            if valid_tuples:
                n_val = len(valid_tuples)
                total_weight = sum(tuple_weights) if tuple_weights else float(n_val)
                beat_weight = sum(
                    w for tup, w in zip(valid_tuples, tuple_weights)
                    if max(c.truco_power for c in tup) <= my_best_power
                )
                monster_weight = sum(
                    w for tup, w in zip(valid_tuples, tuple_weights)
                    if max(c.truco_power for c in tup) >= 10
                )
                p_beat = beat_weight / total_weight
                p_monster = monster_weight / total_weight

                # Generate human-readable deduction hint from the Envido count + played cards
                if opp_declared_env is not None:
                    played_str = (
                        f" + jugó {', '.join(c.short_str() for c in opp_played_cards)}"
                        if opp_played_cards
                        else ""
                    )
                    if opp_hidden_count == 1:
                        sample_str = ", ".join(tup[0].short_str() for tup in valid_tuples[:5])
                        envido_card_hint = (
                            f"Cantó {opp_declared_env}{played_str} → última carta reducida a "
                            f"{n_val} opción(es) ({sample_str})"
                        )
                    else:
                        # Count weighted card likelihood across valid tuples
                        card_w_freq: Dict[str, float] = {}
                        for tup, w in zip(valid_tuples, tuple_weights):
                            for c in tup:
                                s = c.short_str()
                                card_w_freq[s] = card_w_freq.get(s, 0.0) + w
                        top_cards = sorted(card_w_freq.items(), key=lambda kv: kv[1], reverse=True)[:4]
                        top_str = ", ".join(f"{k} ({round(v / total_weight * 100)}%)" for k, v in top_cards)
                        envido_card_hint = (
                            f"Cantó {opp_declared_env}{played_str} → {n_val} combinaciones posibles "
                            f"(Más probables: {top_str})"
                        )
                elif opp_said_son_buenas:
                    envido_card_hint = (
                        f"Dijo 'Son buenas' (≤ {mano_env}) → {n_val} manos posibles restantes"
                    )

        # Bayesian Action Conditioning + Bluff / Phishing-Trap Smelling!
        opp_called_truco = any(
            actor == opp and act in (Action.CALL_TRUCO, Action.CALL_RETRUCO, Action.CALL_VALE_CUATRO)
            for actor, act, _ in state.history
        )
        opp_accepted_truco = any(
            actor == opp and act == Action.QUIERO and state.truco_level >= 1
            for actor, act, _ in state.history
        )

        # Detect "Achique" / Phishing Slow-Play Trap:
        # Opponent played a weak card (power <= 5: 4, 5, 6, 7C/7B, 10, 11) in Trick 0
        # without folding, keeping 1..2 hidden cards for Trick 1/2!
        opp_t0_weak = any(
            tc.player == opp and tc.card.truco_power <= 5
            for tc in (state.tricks[0] if state.tricks else [])
        )

        exact_pinpoint = (opp_hidden_count == 1 and opp_declared_env is not None)
        if opp_t0_weak and len(state.tricks) >= 2 and not state.is_terminal:
            # Classic phishing signature: threw garbage on Trick 1 but stayed alive in the hand
            p_trap_suspect = 0.65 if (opp_called_truco or opp_accepted_truco) else 0.42
            if not exact_pinpoint:
                p_monster = max(p_monster, min(0.95, p_monster + 0.30 * (1.0 - p_monster)))
                p_beat = max(0.05, p_beat * 0.75)
        elif opp_called_truco:
            # Opponent shouts TRUCO: is it a real Monster or a Bluff ("Mentira")?
            # If we hold high blocker cards (e.g., monsters ourselves) and opp hasn't shown strength yet,
            # there is a significant probability (~35-45%) they are bluffing!
            own_monsters = sum(1 for c in state.hands[p] if c.truco_power >= 10)
            p_bluff_suspect = min(0.60, 0.28 + 0.12 * own_monsters)
            if not exact_pinpoint:
                # Discount the intimidation effect by (1 - p_bluff_suspect) so the model doesn't over-fold to bluffers!
                credibility = 1.0 - 0.55 * p_bluff_suspect
                p_monster = max(p_monster, min(0.92, p_monster + (0.24 * credibility) * (1.0 - p_monster)))
                p_beat = max(0.08, p_beat * (1.0 - 0.18 * credibility))
        elif opp_accepted_truco and not exact_pinpoint:
            p_monster = max(p_monster, min(0.90, p_monster + 0.16 * (1.0 - p_monster)))
            p_beat = max(0.06, p_beat * 0.86)

    # 2. Bayesian Envido Win Probability Estimate
    my_env = state.envido_scores[p]
    # Empirical CDF of a random 3-card Argentine Truco Envido score (median ~23.5, 27+ is top ~28%)
    if my_env < 20:
        base_env_win = 0.05 + 0.22 * (my_env / 7.0)
    elif my_env <= 25:
        base_env_win = 0.30 + 0.35 * ((my_env - 20) / 5.0)
    elif my_env <= 29:
        base_env_win = 0.68 + 0.22 * ((my_env - 25) / 4.0)
    else:
        base_env_win = 0.91 + 0.08 * ((my_env - 29) / 4.0)

    if state.mano == p:
        base_env_win = min(0.99, base_env_win + 0.03)  # Mano wins Envido ties

    # Condition on whether Opponent called Envido / Real Envido / Falta Envido
    opp_env_calls = [
        act for actor, act, _ in state.history
        if actor == opp and act in (Action.CALL_ENVIDO, Action.CALL_REAL_ENVIDO, Action.CALL_FALTA_ENVIDO)
    ]
    if opp_env_calls:
        p_bluff_suspect = max(p_bluff_suspect, 0.25)
        # Given opponent called Envido, their distribution is truncated mostly to >= 25 (plus ~25% bluffs)
        if Action.CALL_REAL_ENVIDO in opp_env_calls or Action.CALL_FALTA_ENVIDO in opp_env_calls:
            # Opponent represents 28..33 (or bluff)
            if my_env < 26:
                p_win_env = 0.16
            elif my_env <= 29:
                p_win_env = 0.46
            else:
                p_win_env = 0.82
        else:
            # Standard Envido call by opponent (includes bluffers!)
            if my_env < 23:
                p_win_env = 0.20
            elif my_env <= 26:
                p_win_env = 0.45
            elif my_env <= 29:
                p_win_env = 0.72
            else:
                p_win_env = 0.92
    else:
        p_win_env = base_env_win

    win_bucket = min(4, max(0, int(p_beat * 5.0)))
    threat_bucket = min(4, max(0, int(p_monster * 5.0)))
    env_bucket = min(4, max(0, int(p_win_env * 5.0)))

    return {
        "p_beat_opp_best": round(float(p_beat), 3),
        "p_opp_monster": round(float(p_monster), 3),
        "p_win_envido": round(float(p_win_env), 3),
        "p_bluff_suspect": round(float(p_bluff_suspect), 3),
        "p_trap_suspect": round(float(p_trap_suspect), 3),
        "opp_hidden_cards": opp_hidden_count,
        "opp_declared_envido": opp_declared_env,
        "opp_son_buenas": opp_said_son_buenas,
        "envido_card_hint": envido_card_hint,
        "envido_hypotheses": envido_hypotheses,
        "win_token": f"BELIEF_WIN_{win_bucket}",
        "threat_token": f"BELIEF_THREAT_{threat_bucket}",
        "env_token": f"BELIEF_ENV_{env_bucket}",
    }


def infoset_key(state: TrucoHandState, player: Optional[int] = None) -> Tuple[int, ...]:
    """
    Canonical abstracted Information Set key for tabular/outcome-sampling MCCFR warm-start
    and exact state hashing. Abstracts cards into Truco power tiers (1..14) and Envido into buckets.
    """
    p = state.current_player if player is None else player
    opp = 1 - p

    is_mano = 1 if state.mano == p else 0
    # Available own card powers (sorted for canonical representation in CFR, mapped back to slots)
    hand_powers = tuple(
        state.hands[p][slot].truco_power if not state.played_mask[p][slot] else 0
        for slot in range(3)
    )
    # Envido bucket: <20->0, 20..23->1, 24..26->2, 27..29->3, 30..33->4
    env = state.envido_scores[p]
    if env < 20:
        env_b = 0
    elif env <= 23:
        env_b = 1
    elif env <= 26:
        env_b = 2
    elif env <= 29:
        env_b = 3
    else:
        env_b = 4

    # Publicly played cards in each trick (as power + who played)
    trick_hist: List[int] = []
    for t_idx, trick in enumerate(state.tricks):
        for tc in trick:
            trick_hist.append(1 if tc.player == p else 2)
            trick_hist.append(tc.card.truco_power)
        trick_hist.append(-1)

    # Betting sequence (non-card actions)
    bet_hist = tuple(
        (1 if actor == p else 2, int(act))
        for actor, act, _ in state.history
        if act not in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2)
    )

    return (
        is_mano,
        hand_powers[0],
        hand_powers[1],
        hand_powers[2],
        env_b,
        int(state.pending),
        state.truco_level,
        tuple(trick_hist),
        bet_hist,
    )
