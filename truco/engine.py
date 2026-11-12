"""
Argentine Truco (Sin Flor, 2-player) Game Engine.

Features complete rules for 40-card Spanish deck Argentine Truco:
- 40 cards (ranks 1-7, 10-12 across Espada, Basto, Oro, Copa)
- Exact Truco card hierarchy (14 tiers from 1 de Espada down to 4s)
- Envido calculation & betting chain (Envido, Real Envido, Falta Envido)
  including "Envido va primero" (calling Envido when Truco was just proposed)
- Truco betting chain (Truco -> Retruco -> Vale Cuatro) with ownership tracking
- Best-of-3 tricks (Manos) with full tie-breaker ("Parda") rules:
  - 1st parda -> winner of 2nd wins; if 2nd also parda -> winner of 3rd wins; if all 3 parda -> Mano wins
  - 2nd parda -> winner of 1st wins
  - 3rd parda -> winner of 1st wins
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import List, Optional, Tuple


class Suit(IntEnum):
    ESPADA = 0  # Spades / Swords
    BASTO = 1   # Clubs
    ORO = 2     # Golds / Coins
    COPA = 3    # Cups


SUIT_NAMES = {
    Suit.ESPADA: "Espada",
    Suit.BASTO: "Basto",
    Suit.ORO: "Oro",
    Suit.COPA: "Copa",
}

SUIT_SHORT = {
    Suit.ESPADA: "E",
    Suit.BASTO: "B",
    Suit.ORO: "O",
    Suit.COPA: "C",
}

RANKS = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12)


@dataclass(frozen=True)
class Card:
    rank: int
    suit: Suit
    card_id: int  # 0..39

    @property
    def truco_power(self) -> int:
        """Higher value wins the trick. Range: 1 (4s) to 14 (1 de Espada)."""
        if self.rank == 1 and self.suit == Suit.ESPADA:
            return 14
        if self.rank == 1 and self.suit == Suit.BASTO:
            return 13
        if self.rank == 7 and self.suit == Suit.ESPADA:
            return 12
        if self.rank == 7 and self.suit == Suit.ORO:
            return 11
        if self.rank == 3:
            return 10
        if self.rank == 2:
            return 9
        if self.rank == 1:  # 1 de Oro, 1 de Copa (Anchos falsos)
            return 8
        if self.rank == 12:
            return 7
        if self.rank == 11:
            return 6
        if self.rank == 10:
            return 5
        if self.rank == 7:  # 7 de Basto, 7 de Copa (Sietes falsos)
            return 4
        if self.rank == 6:
            return 3
        if self.rank == 5:
            return 2
        if self.rank == 4:
            return 1
        raise ValueError(f"Invalid card: {self}")

    @property
    def envido_value(self) -> int:
        """Value of single card for Envido (10, 11, 12 are worth 0)."""
        return 0 if self.rank >= 10 else self.rank

    def short_str(self) -> str:
        return f"{self.rank}{SUIT_SHORT[self.suit]}"

    def __str__(self) -> str:
        return f"{self.rank} de {SUIT_NAMES[self.suit]}"


# Build canonical 40-card deck
DECK: Tuple[Card, ...] = tuple(
    Card(rank=r, suit=Suit(s), card_id=s * 10 + i)
    for s in range(4)
    for i, r in enumerate(RANKS)
)


def compute_envido(cards: List[Card]) -> int:
    """Computes the maximum Envido score for a 3-card hand (range 0..33)."""
    best = max(c.envido_value for c in cards)
    n = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            if cards[i].suit == cards[j].suit:
                score = 20 + cards[i].envido_value + cards[j].envido_value
                if score > best:
                    best = score
    return best


class Action(IntEnum):
    PLAY_CARD_0 = 0
    PLAY_CARD_1 = 1
    PLAY_CARD_2 = 2
    CALL_ENVIDO = 3
    CALL_REAL_ENVIDO = 4
    CALL_FALTA_ENVIDO = 5
    CALL_TRUCO = 6
    CALL_RETRUCO = 7
    CALL_VALE_CUATRO = 8
    QUIERO = 9
    NO_QUIERO = 10
    IRSE_AL_MAZO = 11


ACTION_NAMES = {
    Action.PLAY_CARD_0: "Play Card 0",
    Action.PLAY_CARD_1: "Play Card 1",
    Action.PLAY_CARD_2: "Play Card 2",
    Action.CALL_ENVIDO: "Envido",
    Action.CALL_REAL_ENVIDO: "Real Envido",
    Action.CALL_FALTA_ENVIDO: "Falta Envido",
    Action.CALL_TRUCO: "Truco",
    Action.CALL_RETRUCO: "Retruco",
    Action.CALL_VALE_CUATRO: "Vale Cuatro",
    Action.QUIERO: "Quiero",
    Action.NO_QUIERO: "No Quiero",
    Action.IRSE_AL_MAZO: "Irse al Mazo",
}

NUM_ACTIONS = len(Action)


class PendingState(IntEnum):
    NONE = 0
    ENVIDO = 1          # Responding to Envido / Real Envido / Falta Envido
    TRUCO = 2           # Responding to Truco / Retruco / Vale Cuatro


@dataclass
class TrickCard:
    player: int
    card: Card
    hand_slot: int


@dataclass
class TrucoHandState:
    """Represents a single hand (round) of Argentine Truco."""
    mano: int = 0  # Player who is Mano (0 or 1), plays first in Trick 0 and wins ties
    match_scores: Tuple[int, int] = (0, 0)
    target_score: int = 15

    hands: List[List[Card]] = field(default_factory=lambda: [[], []])
    played_mask: List[List[bool]] = field(default_factory=lambda: [[False, False, False], [False, False, False]])
    envido_scores: Tuple[int, int] = (0, 0)

    # Tricks history: up to 3 tricks, each a list of TrickCard
    tricks: List[List[TrickCard]] = field(default_factory=lambda: [[]])
    trick_winners: List[int] = field(default_factory=list)  # 0, 1, or -1 for Parda (tie)

    current_player: int = 0
    turn_Before_bet: int = 0  # Tracks whose turn it is to play a card once bets resolve

    # Pending bet state
    pending: PendingState = PendingState.NONE
    pending_caller: int = -1

    # Envido state
    envido_resolved: bool = False
    envido_chain: List[Action] = field(default_factory=list)
    envido_points_accepted: int = 0
    envido_points_declined: int = 1
    envido_winner: int = -1
    envido_awarded_points: int = 0
    envido_revealed_scores: List[Optional[int]] = field(default_factory=lambda: [None, None])
    envido_son_buenas: bool = False
    truco_suspended_by_envido: bool = False  # "Envido va primero" when Truco was called in trick 0

    # Truco state
    truco_level: int = 0  # 0: uncalled (1pt), 1: Truco (2pt), 2: Retruco (3pt), 3: Vale Cuatro (4pt)
    truco_stakes: int = 1
    truco_owner: int = -1  # Who has "el quiero" (-1 = anyone can call Truco)

    # Terminal state
    is_terminal: bool = False
    truco_winner: int = -1
    points_won: List[int] = field(default_factory=lambda: [0, 0])
    history: List[Tuple[int, Action, Optional[Card]]] = field(default_factory=list)

    @classmethod
    def new_hand(
        cls,
        mano: int = 0,
        match_scores: Tuple[int, int] = (0, 0),
        target_score: int = 15,
        rng: Optional[random.Random] = None,
        dealt_cards: Optional[Tuple[List[Card], List[Card]]] = None,
    ) -> TrucoHandState:
        if dealt_cards is not None:
            h0 = sorted(dealt_cards[0], key=lambda c: (c.truco_power, c.card_id), reverse=True)
            h1 = sorted(dealt_cards[1], key=lambda c: (c.truco_power, c.card_id), reverse=True)
        else:
            r = rng or random
            sampled = r.sample(DECK, 6)
            h0 = sorted(sampled[:3], key=lambda c: (c.truco_power, c.card_id), reverse=True)
            h1 = sorted(sampled[3:], key=lambda c: (c.truco_power, c.card_id), reverse=True)

        state = cls(
            mano=mano,
            match_scores=match_scores,
            target_score=target_score,
            hands=[h0, h1],
            played_mask=[[False, False, False], [False, False, False]],
            envido_scores=(compute_envido(h0), compute_envido(h1)),
            tricks=[[]],
            trick_winners=[],
            current_player=mano,
            turn_Before_bet=mano,
        )
        return state

    def clone(self) -> TrucoHandState:
        return TrucoHandState(
            mano=self.mano,
            match_scores=self.match_scores,
            target_score=self.target_score,
            hands=[list(self.hands[0]), list(self.hands[1])],
            played_mask=[list(self.played_mask[0]), list(self.played_mask[1])],
            envido_scores=self.envido_scores,
            tricks=[list(t) for t in self.tricks],
            trick_winners=list(self.trick_winners),
            current_player=self.current_player,
            turn_Before_bet=self.turn_Before_bet,
            pending=self.pending,
            pending_caller=self.pending_caller,
            envido_resolved=self.envido_resolved,
            envido_chain=list(self.envido_chain),
            envido_points_accepted=self.envido_points_accepted,
            envido_points_declined=self.envido_points_declined,
            envido_winner=self.envido_winner,
            envido_awarded_points=self.envido_awarded_points,
            envido_revealed_scores=list(self.envido_revealed_scores),
            envido_son_buenas=self.envido_son_buenas,
            truco_suspended_by_envido=self.truco_suspended_by_envido,
            truco_level=self.truco_level,
            truco_stakes=self.truco_stakes,
            truco_owner=self.truco_owner,
            is_terminal=self.is_terminal,
            truco_winner=self.truco_winner,
            points_won=list(self.points_won),
            history=list(self.history),
        )

    def _compute_falta_envido_points(self) -> int:
        leader_score = max(self.match_scores[0], self.match_scores[1])
        return max(1, self.target_score - leader_score)

    def can_call_initial_envido(self, player: int) -> bool:
        """Envido can only be initiated in the first trick before `player` has played their first card."""
        if self.envido_resolved:
            return False
        if len(self.tricks) > 1:
            return False
        if any(self.played_mask[player]):
            return False
        return True

    def legal_actions(self) -> List[Action]:
        if self.is_terminal:
            return []

        p = self.current_player

        # 1. Responding to an Envido bet
        if self.pending == PendingState.ENVIDO:
            actions = [Action.QUIERO, Action.NO_QUIERO]
            last_call = self.envido_chain[-1]
            # Allowed escalations:
            # Envido -> Envido (if only 1 Envido called so far), Real Envido, Falta Envido
            # Real Envido -> Falta Envido
            # Falta Envido -> none
            if last_call == Action.CALL_ENVIDO:
                if len(self.envido_chain) == 1:
                    actions.append(Action.CALL_ENVIDO)
                actions.append(Action.CALL_REAL_ENVIDO)
                actions.append(Action.CALL_FALTA_ENVIDO)
            elif last_call == Action.CALL_REAL_ENVIDO:
                actions.append(Action.CALL_FALTA_ENVIDO)
            return sorted(actions)

        # 2. Responding to a Truco bet
        if self.pending == PendingState.TRUCO:
            actions = [Action.QUIERO, Action.NO_QUIERO]
            # Escalate Truco
            if self.truco_level == 1:
                actions.append(Action.CALL_RETRUCO)
            elif self.truco_level == 2:
                actions.append(Action.CALL_VALE_CUATRO)

            # "El Envido está primero": If opponent just called initial Truco in Trick 0
            # and the responder hasn't played a card yet and Envido hasn't been resolved,
            # ONLY the responder (p != self.pending_caller) can call Envido / Real Envido / Falta Envido
            # BEFORE accepting/declining Truco!
            if self.truco_level == 1 and p != self.pending_caller and self.can_call_initial_envido(p):
                actions.extend([
                    Action.CALL_ENVIDO,
                    Action.CALL_REAL_ENVIDO,
                    Action.CALL_FALTA_ENVIDO,
                ])
            return sorted(actions)

        # 3. Normal turn (PendingState.NONE)
        actions: List[Action] = []
        # Can play any unplayed card
        for slot in range(3):
            if not self.played_mask[p][slot]:
                actions.append(Action(slot))

        # Can call Envido ONLY if in Trick 0, haven't played a card yet, AND Truco has not already been called/accepted!
        if self.truco_level == 0 and self.can_call_initial_envido(p):
            actions.extend([
                Action.CALL_ENVIDO,
                Action.CALL_REAL_ENVIDO,
                Action.CALL_FALTA_ENVIDO,
            ])

        # Can call Truco / Retruco / Vale Cuatro if eligible
        if self.truco_level == 0:
            actions.append(Action.CALL_TRUCO)
        elif self.truco_level == 1 and self.truco_owner == p:
            actions.append(Action.CALL_RETRUCO)
        elif self.truco_level == 2 and self.truco_owner == p:
            actions.append(Action.CALL_VALE_CUATRO)

        # Can always fold (Irse al Mazo) on normal turn
        actions.append(Action.IRSE_AL_MAZO)
        return sorted(actions)

    def step(self, action: Action) -> TrucoHandState:
        if action not in self.legal_actions():
            raise ValueError(
                f"Illegal action {ACTION_NAMES.get(action, action)} for player {self.current_player}. "
                f"Legal: {[ACTION_NAMES[a] for a in self.legal_actions()]}"
            )

        p = self.current_player
        opp = 1 - p

        # Record in history (include card if playing a card)
        played_card: Optional[Card] = None
        if action in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2):
            played_card = self.hands[p][int(action)]
        self.history.append((p, action, played_card))

        # --- Case 1: Playing a card ---
        if action in (Action.PLAY_CARD_0, Action.PLAY_CARD_1, Action.PLAY_CARD_2):
            slot = int(action)
            self.played_mask[p][slot] = True
            card = self.hands[p][slot]
            current_trick = self.tricks[-1]
            current_trick.append(TrickCard(player=p, card=card, hand_slot=slot))

            if len(current_trick) == 1:
                # Opponent's turn to play second card in the trick
                self.current_player = opp
                self.turn_Before_bet = opp
            else:
                # Trick finished! Evaluate trick winner
                c0 = current_trick[0]
                c1 = current_trick[1]
                if c0.card.truco_power > c1.card.truco_power:
                    winner = c0.player
                elif c1.card.truco_power > c0.card.truco_power:
                    winner = c1.player
                else:
                    winner = -1  # Parda!
                self.trick_winners.append(winner)

                hand_winner = self._check_hand_winner()
                if hand_winner != -1:
                    self._finish_hand(hand_winner)
                else:
                    # Start next trick
                    self.tricks.append([])
                    # If parda, the player who led that trick leads the next trick (which is Mano if trick 0 was parda)
                    next_leader = c0.player if winner == -1 else winner
                    self.current_player = next_leader
                    self.turn_Before_bet = next_leader
            return self

        # --- Case 2: Folding (Irse al Mazo) ---
        if action == Action.IRSE_AL_MAZO:
            # If folding in trick 0 before envido was resolved and opponent hasn't played yet,
            # standard Argentine rule: if nobody played a card yet and Mano folds before Envido, it's 2 pts (1 envido + 1 truco),
            # otherwise current truco_stakes. Let's award 2 pts if folding on very first action of hand, else truco_stakes.
            if not self.envido_resolved and len(self.tricks) == 1 and len(self.tricks[0]) == 0 and self.truco_level == 0:
                self.points_won[opp] += 2
                self.truco_winner = opp
                self.is_terminal = True
            else:
                self._finish_hand(opp)
            return self

        # --- Case 3: Calling or Escalating Envido ---
        if action in (Action.CALL_ENVIDO, Action.CALL_REAL_ENVIDO, Action.CALL_FALTA_ENVIDO):
            if self.pending == PendingState.TRUCO:
                # "El Envido está primero": suspends the initial Truco call!
                self.truco_suspended_by_envido = True
                # Reset truco_level back to 0 because Envido overrides an unaccepted Truco call
                self.truco_level = 0

            self.pending = PendingState.ENVIDO
            self.pending_caller = p

            add_pts = (
                2 if action == Action.CALL_ENVIDO
                else (3 if action == Action.CALL_REAL_ENVIDO else self._compute_falta_envido_points())
            )
            if len(self.envido_chain) == 0:
                self.envido_points_declined = 1
                self.envido_points_accepted = add_pts
            else:
                self.envido_points_declined = self.envido_points_accepted
                if action == Action.CALL_FALTA_ENVIDO:
                    # In Argentine Truco, Falta Envido replaces or tops to target_score - max_score
                    self.envido_points_accepted = self._compute_falta_envido_points()
                else:
                    self.envido_points_accepted += add_pts

            self.envido_chain.append(action)
            self.current_player = opp
            return self

        # --- Case 4: Calling or Escalating Truco ---
        if action in (Action.CALL_TRUCO, Action.CALL_RETRUCO, Action.CALL_VALE_CUATRO):
            self.pending = PendingState.TRUCO
            self.pending_caller = p
            if action == Action.CALL_TRUCO:
                self.truco_level = 1
            elif action == Action.CALL_RETRUCO:
                self.truco_level = 2
                self.truco_stakes = 2  # Calling Retruco implicitly accepts Truco (2 pts if Retruco is declined)
                self.truco_owner = opp
                self.envido_resolved = True  # Once Truco is accepted/escalated, Envido window closes permanently
            elif action == Action.CALL_VALE_CUATRO:
                self.truco_level = 3
                self.truco_stakes = 3  # Calling Vale Cuatro implicitly accepts Retruco (3 pts if declined)
                self.truco_owner = opp
                self.envido_resolved = True

            self.current_player = opp
            return self

        # --- Case 5: Responding to Pending Bet (QUIERO / NO_QUIERO) ---
        if action in (Action.QUIERO, Action.NO_QUIERO):
            if self.pending == PendingState.ENVIDO:
                self.envido_resolved = True
                if action == Action.QUIERO:
                    # In Argentine Truco, Mano ALWAYS declares their Envido count first!
                    # Pie only reveals their count if it strictly beats Mano; otherwise Pie says "Son buenas"
                    # and keeps their own Envido score hidden!
                    mano_p = self.mano
                    pie_p = 1 - mano_p
                    self.envido_revealed_scores[mano_p] = self.envido_scores[mano_p]

                    if self.envido_scores[pie_p] > self.envido_scores[mano_p]:
                        ewinner = pie_p
                        self.envido_revealed_scores[pie_p] = self.envido_scores[pie_p]
                        self.envido_son_buenas = False
                    else:
                        ewinner = mano_p
                        self.envido_revealed_scores[pie_p] = None
                        self.envido_son_buenas = True

                    self.envido_winner = ewinner
                    self.envido_awarded_points = self.envido_points_accepted
                    self.points_won[ewinner] += self.envido_points_accepted
                else:
                    ewinner = self.pending_caller
                    self.envido_winner = ewinner
                    self.envido_awarded_points = self.envido_points_declined
                    self.points_won[ewinner] += self.envido_points_declined
                    self.envido_revealed_scores = [None, None]
                    self.envido_son_buenas = False

                self.pending = PendingState.NONE
                self.pending_caller = -1
                self.truco_suspended_by_envido = False
                self.current_player = self.turn_Before_bet
                return self

            elif self.pending == PendingState.TRUCO:
                if action == Action.QUIERO:
                    # Accepted! Stakes increase to truco_level + 1, and accepter gets "el quiero" (truco_owner = p).
                    # Once Truco is accepted with Quiero, the Envido window closes permanently!
                    self.envido_resolved = True
                    self.truco_stakes = self.truco_level + 1
                    self.truco_owner = p
                    self.pending = PendingState.NONE
                    self.pending_caller = -1
                    self.current_player = self.turn_Before_bet
                    return self
                else:
                    # Declined Truco/Retruco/Vale Cuatro -> hand ends immediately, caller wins current truco_stakes
                    caller = self.pending_caller
                    self.pending = PendingState.NONE
                    self.pending_caller = -1
                    self._finish_hand(caller)
                    return self

        raise RuntimeError(f"Unhandled action {action}")

    def _check_hand_winner(self) -> int:
        """
        Evaluates best-of-3 tricks with Argentine Parda rules.
        Returns 0, 1, or -1 if the hand is not yet decided.
        """
        w = self.trick_winners
        n = len(w)
        if n < 2:
            return -1

        # Count outright wins
        wins0 = sum(1 for x in w if x == 0)
        wins1 = sum(1 for x in w if x == 1)
        if wins0 == 2:
            return 0
        if wins1 == 2:
            return 1

        if n == 2:
            # Trick 1 was Parda, Trick 2 has a winner -> Trick 2 winner wins hand
            if w[0] == -1 and w[1] != -1:
                return w[1]
            # Trick 1 had a winner, Trick 2 was Parda -> Trick 1 winner wins hand
            if w[0] != -1 and w[1] == -1:
                return w[0]
            # Otherwise (e.g. [0, 1], [1, 0], or [-1, -1]) -> go to Trick 3
            return -1

        if n == 3:
            # If Trick 3 was Parda:
            if w[2] == -1:
                # If Trick 1 had a winner (i.e. [0, 1, -1] or [1, 0, -1]), Trick 1 winner wins!
                if w[0] != -1:
                    return w[0]
                # Triple Parda [-1, -1, -1] -> Mano wins!
                return self.mano
            else:
                # Trick 3 has a winner (either after [0, 1, w2] or [-1, -1, w2])
                return w[2]

        return -1

    def _finish_hand(self, winner: int) -> None:
        self.truco_winner = winner
        self.points_won[winner] += self.truco_stakes
        self.is_terminal = True

    def payoff(self, player: int) -> float:
        """Net points won by `player` in this hand."""
        if not self.is_terminal:
            raise RuntimeError("Hand is not terminal yet")
        return float(self.points_won[player] - self.points_won[1 - player])
