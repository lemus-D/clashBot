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

# Suppress unnecessary model warnings
os.environ['CORE_MODEL_SAM_ENABLED'] = 'False'
os.environ['CORE_MODEL_SAM3_ENABLED'] = 'False'
os.environ['CORE_MODEL_GAZE_ENABLED'] = 'False'
os.environ['CORE_MODEL_YOLO_WORLD_ENABLED'] = 'False'

# --- CONFIGURATION ---
MODEL_ID = "troop-counter/7"
API_KEY = os.getenv('API_KEY')

# PyAutoGUI safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small delay between actions


# --- MOUSE CONTROL FUNCTIONS ---

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


def click_position(x, y, monitor, duration=0.1):
    """
    Click at a specific position relative to the game window

    Args:
        x: X coordinate relative to the captured monitor region
        y: Y coordinate relative to the captured monitor region
        monitor: The monitor dict with window position
        duration: Time to move mouse (0 for instant)
    """
    global_x, global_y = screen_to_global(x, y, monitor)

    # Move and click
    if duration > 0:
        pyautogui.moveTo(global_x, global_y, duration=duration)
    else:
        pyautogui.moveTo(global_x, global_y)

    pyautogui.click()
    print(f"Clicked at screen ({x}, {y}) -> global ({global_x}, {global_y})")


def drag_card_to_position(card_x, card_y, target_x, target_y, monitor, duration=0.3):
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
    start_global_x, start_global_y = screen_to_global(card_x, card_y, monitor)
    end_global_x, end_global_y = screen_to_global(target_x, target_y, monitor)

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


def place_card(card_index, arena_x, arena_y, monitor):
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
    drag_card_to_position(
        card_x_pos, card_y_pos,
        arena_x, arena_y,
        monitor,
        duration=0.2
    )


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


# --- MAIN WINDOW CAPTURE FUNCTION ---

def start_window_cap(window_name, enable_mouse_control=False):
    """
    Start window capture with card detection and optional mouse control

    Args:
        window_name: Name of the window to capture
        enable_mouse_control: If True, enables mouse control features
    """
    # 1. LOAD THE MODEL (Downloads weights once, then runs locally)
    print("Loading model... this may take a moment first time.")
    print(f"Model ID: {MODEL_ID}")
    print("Downloading/loading weights...")

    try:
        model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. SETUP ANNOTATORS (For drawing boxes on the screen)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 3. FIND AND ACTIVATE WINDOW
    try:
        window = gw.getWindowsWithTitle(window_name)[0]
        if window.isMinimized:
            window.restore()
        window.activate()
        time.sleep(0.5)  # Wait for window to activate
    except IndexError:
        print(f'No window found with title: {window_name}')
        return

    with mss.mss() as sct:
        print("Press 'q' to quit.")
        if enable_mouse_control:
            print("⚠️  MOUSE CONTROL ENABLED")
            print("Press 'p' to place test card at center")
            print("Press 'c' to click center of screen")

        while True:
            # Update monitor region in case window moves
            monitor = {
                "top": window.top + 50,
                "left": window.left + 383,
                "width": window.width - 433,
                "height": window.height - 50
            }

            # Capture screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            # Convert BGRA (mss) -> BGR (OpenCV)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # --- INFERENCE STEP ---
            # Run the model on the current frame locally
            results = model.infer(frame)[0]

            # Convert results to Supervision format
            detections = sv.Detections.from_inference(results)

            # Draw boxes and labels on the frame
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            # Print card counts to console (Optional debugging)
            # print(f"Cards detected: {len(detections)}")

            # Display the result
            cv2.imshow("Window Capture", annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p') and enable_mouse_control:
                # Test: Place first card in center of arena
                center_x = monitor["width"] // 2
                center_y = monitor["height"] // 2
                place_card(0, center_x, center_y, monitor)
                time.sleep(1)  # Cooldown
            elif key == ord('c') and enable_mouse_control:
                # Test: Click center of screen
                click_position(
                    monitor["width"] // 2,
                    monitor["height"] // 2,
                    monitor
                )

    cv2.destroyAllWindows()