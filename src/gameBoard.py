class GameBoard():
    def __init__(self):
        self.cards_in_hand = []
        self.troops_in_arena = []
    def add_card_to_hand(self, card):
        self.cards_in_hand.append(card)

    def add_troop_to_arena(self, troop):
        self.troops_in_arena.append(troop)