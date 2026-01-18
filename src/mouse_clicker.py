import cv2
import numpy as np
import pywinctl as gw
import mss
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
import os
import pyautogui
import time

load_dotenv()

# --- CONFIGURATION ---
MODEL_ID = "troop-counter/7"
API_KEY = os.getenv('API_KEY')

# PyAutoGUI safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small delay between actions


class ClashRoyaleBot:
    def __init__(self, window_name):
        self.window_name = window_name
        self.window = None
        self.model = None
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Card hand positions (bottom of screen)
        self.card_slots = []  # Will be populated based on detections

        # Arena grid positions (8x18 tiles typically)
        self.arena_offset = {"x": 0, "y": 0}  # Relative to monitor region

    def initialize_window(self):
        """Find and activate the BlueStacks window"""
        try:
            windows = gw.getWindowsWithTitle(self.window_name)
            if not windows:
                print(f"No window found with title: {self.window_name}")
                return False

            self.window = windows[0]
            if self.window.isMinimized:
                self.window.restore()
            self.window.activate()
            time.sleep(0.5)  # Wait for window to activate
            return True
        except Exception as e:
            print(f"Error initializing window: {e}")
            return False

    def load_model(self):
        """Load the Roboflow model"""
        print("Loading model... this may take a moment first time.")
        self.model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        print("Model loaded successfully!")

    def get_monitor_region(self):
        """Get the current monitor region for the game window"""
        return {
            "top": self.window.top + 50,
            "left": self.window.left + 383,
            "width": self.window.width - 433,
            "height": self.window.height - 50
        }

    @staticmethod
    def screen_to_global(x, y, monitor):
        """
        Convert screen coordinates to global mouse coordinates

        Args:
            x: X coordinate relative to monitor region
            y: Y coordinate relative to monitor region
            monitor: The monitor dict with window position

        Returns:
            Tuple of (global_x, global_y)
        """
        global_x = monitor["left"] + x
        global_y = monitor["top"] + y
        return global_x, global_y

    def click_position(self, x, y, monitor, duration=0.1):
        """
        Click at a specific position relative to the game window

        Args:
            x: X coordinate relative to the captured monitor region
            y: Y coordinate relative to the captured monitor region
            monitor: The monitor dict with window position
            duration: Time to move mouse (0 for instant)
        """
        global_x, global_y = self.screen_to_global(x, y, monitor)

        # Move and click
        if duration > 0:
            pyautogui.moveTo(global_x, global_y, duration=duration)
        else:
            pyautogui.moveTo(global_x, global_y)

        pyautogui.click()
        print(f"Clicked at screen ({x}, {y}) -> global ({global_x}, {global_y})")

    def drag_card_to_position(self, card_x, card_y, target_x, target_y, monitor, duration=0.3):
        """
        Drag a card from hand to arena position

        Args:
            card_x: Starting X position (card in hand)
            card_y: Starting Y position (card in hand)
            target_x: Target X position in arena
            target_y: Target Y position in arena
            monitor: The monitor dict with window position
            duration: Drag duration
        """
        start_global_x, start_global_y = self.screen_to_global(card_x, card_y, monitor)
        end_global_x, end_global_y = self.screen_to_global(target_x, target_y, monitor)

        # Perform drag
        pyautogui.moveTo(start_global_x, start_global_y)
        time.sleep(0.05)
        pyautogui.drag(
            end_global_x - start_global_x,
            end_global_y - start_global_y,
            duration=duration,
            button='left'
        )

        print(f"Dragged card from ({card_x}, {card_y}) to ({target_x}, {target_y})")

    def place_card(self, card_index, arena_x, arena_y, monitor):
        """
        Place a card at a specific arena position

        Args:
            card_index: Index of card in hand (0-3 typically)
            arena_x: Arena X coordinate (pixels relative to monitor)
            arena_y: Arena Y coordinate (pixels relative to monitor)
            monitor: The monitor dict
        """
        # Calculate card position in hand
        # Adjust these values based on your actual card positions
        card_spacing = 100  # Horizontal spacing between cards
        card_y_pos = monitor["height"] - 100  # Cards near bottom
        card_x_pos = (monitor["width"] // 2) - 150 + (card_index * card_spacing)

        # Drag card to arena position
        self.drag_card_to_position(
            card_x_pos, card_y_pos,
            arena_x, arena_y,
            monitor,
            duration=0.2
        )

    @staticmethod
    def get_arena_tile_position(tile_x, tile_y, monitor):
        """
        Convert arena tile coordinates to pixel coordinates

        Args:
            tile_x: Tile X position (0-17 typically)
            tile_y: Tile Y position (0-31 typically)
            monitor: The monitor dict

        Returns:
            Tuple of pixel coordinates (x, y)
        """
        # These values need calibration based on your screen resolution
        tile_width = monitor["width"] / 18
        tile_height = monitor["height"] / 32

        pixel_x = int(tile_x * tile_width)
        pixel_y = int(tile_y * tile_height)

        return pixel_x, pixel_y

    def run_detection_loop(self, enable_mouse_control=False):
        """
        Main loop for card detection and optional mouse control

        Args:
            enable_mouse_control: If True, allows mouse control (use carefully!)
        """
        if not self.initialize_window():
            return

        with mss.mss() as sct:
            print("Bot running. Press 'q' to quit.")
            if enable_mouse_control:
                print("⚠️  MOUSE CONTROL ENABLED - Press 'p' to place test card")

            while True:
                # Update monitor region
                monitor = self.get_monitor_region()

                # Capture screen
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Run inference
                results = self.model.infer(frame)[0]
                detections = sv.Detections.from_inference(results)

                # Annotate frame
                annotated_frame = self.box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )

                # Display
                cv2.imshow("Clash Royale Bot", annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('p') and enable_mouse_control:
                    # Test: Place first card in center of arena
                    center_x = monitor["width"] // 2
                    center_y = monitor["height"] // 2
                    self.place_card(0, center_x, center_y, monitor)
                    time.sleep(1)  # Cooldown
                elif key == ord('c') and enable_mouse_control:
                    # Test: Click center of screen
                    self.click_position(
                        monitor["width"] // 2,
                        monitor["height"] // 2,
                        monitor
                    )

        cv2.destroyAllWindows()


def main():
    # Initialize bot
    bot = ClashRoyaleBot(window_name="BlueStacks")  # Adjust window name
    bot.load_model()

    # Run with mouse control disabled for safety
    # Set to True when ready to test mouse control
    bot.run_detection_loop(enable_mouse_control=False)


if __name__ == "__main__":
    main()