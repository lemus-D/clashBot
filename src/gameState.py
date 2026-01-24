import time
import gameBoard


class GameState:
    """
    Tracks game state including match time and elixir count.
    Handles normal, double, and triple elixir phases automatically.
    """

    # Elixir generation rates (seconds per elixir)
    NORMAL_ELIXIR_RATE = 2.8
    DOUBLE_ELIXIR_RATE = 1.4
    TRIPLE_ELIXIR_RATE = 0.93

    # Match timing constants (in seconds)
    DOUBLE_ELIXIR_START = 120  # 2:00 mark
    REGULAR_TIME_END = 180  # 3:00 mark
    TRIPLE_ELIXIR_START = 240  # 4:00 mark
    MATCH_MAX_DURATION = 300  # 5:00 mark (end of overtime)

    # Elixir constants
    STARTING_ELIXIR = 5.0
    MAX_ELIXIR = 10.0

    def __init__(self):
        """Initialize game state tracker"""
        self.match_start_time = None
        self.current_elixir = self.STARTING_ELIXIR
        self.last_elixir_update = None
        self.is_match_active = False

    def start_match(self):
        """Start tracking a new match"""
        self.match_start_time = time.time()
        self.last_elixir_update = time.time()
        self.current_elixir = self.STARTING_ELIXIR
        self.is_match_active = True
        print("Match started!")

    def end_match(self):
        """End the current match"""
        self.is_match_active = False
        print("Match ended!")

    def get_current_match_time(self):
        """
        Get current match time in seconds (how much time has elapsed)

        Returns:
            float: Seconds elapsed since match start, or 0 if no active match
        """
        if not self.is_match_active or self.match_start_time is None:
            return 0

        elapsed = time.time() - self.match_start_time
        return min(elapsed, self.MATCH_MAX_DURATION)

    def get_time_remaining(self):
        """
        Get time remaining in current phase

        Returns:
            float: Seconds remaining in match
        """
        if not self.is_match_active:
            return 0.0

        elapsed = self.get_current_match_time()
        return max(0.0, self.MATCH_MAX_DURATION - elapsed)

    def get_match_phase(self):
        """
        Get current match phase

        Returns:
            str: 'normal', 'double', 'overtime_double', 'overtime_triple', or 'ended'
        """
        if not self.is_match_active:
            return 'ended'

        elapsed = self.get_current_match_time()

        if elapsed < self.DOUBLE_ELIXIR_START:
            return 'normal'
        elif elapsed < self.REGULAR_TIME_END:
            return 'double'
        elif elapsed < self.TRIPLE_ELIXIR_START:
            return 'overtime_double'
        else:
            return 'overtime_triple'

    def get_elixir_rate(self):
        """
        Get current elixir generation rate based on match time

        Returns:
            float: Seconds per elixir point
        """
        elapsed = self.get_current_match_time()

        if elapsed < self.DOUBLE_ELIXIR_START:  # 0:00 - 2:00
            return self.NORMAL_ELIXIR_RATE
        elif elapsed < self.TRIPLE_ELIXIR_START:  # 2:00 - 4:00
            return self.DOUBLE_ELIXIR_RATE
        else:  # 4:00 - 5:00
            return self.TRIPLE_ELIXIR_RATE

    def update_elixir(self):
        """
        Update elixir count based on time elapsed
        Should be called frequently (every frame) to maintain accuracy
        """
        if not self.is_match_active:
            return

        now = time.time()

        if self.last_elixir_update is None:
            self.last_elixir_update = now
            return

        elapsed = now - self.last_elixir_update
        elixir_rate = self.get_elixir_rate()
        elixir_gained = elapsed / elixir_rate

        self.current_elixir = min(self.MAX_ELIXIR, self.current_elixir + elixir_gained)
        self.last_elixir_update = now

    def get_current_elixir(self):
        """
        Get current elixir count

        Returns:
            float: Current elixir (0-10)
        """
        self.update_elixir()
        return self.current_elixir

    def spend_elixir(self, amount):
        """
        Spend elixir (when placing a card)

        Args:
            amount: Elixir cost of card placed

        Returns:
            bool: True if had enough elixir, False otherwise
        """
        if self.current_elixir >= amount:
            self.current_elixir -= amount
            print(f"Spent {amount} elixir. Remaining: {self.current_elixir:.1f}")
            return True
        else:
            print(f"Not enough elixir! Have {self.current_elixir:.1f}, need {amount}")
            return False

    def get_formatted_time(self):
        """
        Get formatted match time as MM:SS

        Returns:
            str: Time in format "M:SS" (e.g., "2:45")
        """
        elapsed = self.get_current_match_time()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}:{seconds:02d}"

    def get_formatted_time_remaining(self):
        """
        Get formatted time remaining as MM:SS

        Returns:
            str: Time remaining in format "M:SS"
        """
        remaining = self.get_time_remaining()
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}:{seconds:02d}"

    def is_overtime(self):
        """
        Check if match is in overtime

        Returns:
            bool: True if in overtime (after 3:00)
        """
        return self.get_current_match_time() >= self.REGULAR_TIME_END

    def get_status_string(self):
        """
        Get comprehensive status string for debugging

        Returns:
            str: Status string with time, elixir, and phase info
        """
        if not self.is_match_active:
            return "No active match"

        return (f"Time: {self.get_formatted_time()} | "
                f"Elixir: {self.current_elixir:.1f}/10 | "
                f"Phase: {self.get_match_phase()} | "
                f"Rate: {self.get_elixir_rate():.2f}s/elixir")


# Example usage
# if __name__ == "__main__":
#     # Create game state tracker
#     game = GameState()
#
#     # Start a match
#     game.start_match()
#
#     # Simulate game progression
#     for i in range(10):
#         time.sleep(0.5)  # Simulate frame delay
#
#         current_time = game.get_current_match_time()
#         elixir = game.get_current_elixir()
#         phase = game.get_match_phase()
#
#         print(f"[{game.get_formatted_time()}] Elixir: {elixir:.2f} | Phase: {phase}")
#
#         # Example: Place a card at 5 elixir
#         if elixir >= 5 and i == 5:
#             game.spend_elixir(5)
#
#     game.end_match()