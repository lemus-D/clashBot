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
from gameBoard import GameBoard  # Import your GameBoard class

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
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


def start_window_cap(window_name, enable_mouse_control=False, save_training_data=False):
    """
    Start window capture with card detection and optional mouse control

    Args:
        window_name: Name of the window to capture
        enable_mouse_control: If True, enables mouse control features
        save_training_data: If True, saves board states for AI training
    """
    # 1. LOAD THE MODEL
    print("Loading model... this may take a moment first time.")
    print(f"Model ID: {MODEL_ID}")

    try:
        model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. SETUP ANNOTATORS
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 3. FIND AND ACTIVATE WINDOW
    try:
        window = gw.getWindowsWithTitle(window_name)[0]
        if window.isMinimized:
            window.restore()
        window.activate()
        time.sleep(0.5)
    except IndexError:
        print(f'No window found with title: {window_name}')
        return

    # 4. INITIALIZE GAME BOARD
    # We'll update this with actual monitor dimensions once captured
    game_board = None

    # For saving training data
    training_data = []
    frame_count = 0

    with mss.mss() as sct:
        print("Press 'q' to quit.")
        print("Press 's' to save current board state for training")
        print("Press 'd' to display current board state")
        if enable_mouse_control:
            print("⚠️  MOUSE CONTROL ENABLED")

        while True:
            # Update monitor region
            monitor = {
                "top": window.top + 50,
                "left": window.left + 383,
                "width": window.width - 433,
                "height": window.height - 50
            }

            # Initialize GameBoard with monitor dimensions on first frame
            if game_board is None:
                game_board = GameBoard(monitor["width"], monitor["height"])
                print(f"GameBoard initialized: {monitor['width']}x{monitor['height']} pixels")
                print(f"Tile size: {game_board.tile_width:.1f}x{game_board.tile_height:.1f} pixels")

            # Capture screen
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # --- INFERENCE STEP ---
            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)

            # --- PROCESS DETECTIONS AND UPDATE GAME BOARD ---
            game_board.clear_arena()  # Clear previous frame's troops
            game_board.clear_hand()  # Clear previous frame's cards

            detection_summary = game_board.process_detections(detections)

            # Draw boxes and labels
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            # Overlay tile grid (optional - helpful for debugging)
            annotated_frame = draw_tile_grid(annotated_frame, game_board)

            # Display detection summary on frame
            summary_text = [
                f"Cards in hand: {len(detection_summary['cards_in_hand'])}",
                f"Cards filtered: {len(detection_summary.get('cards_filtered', []))}",
                f"Troops on board: {len(detection_summary['troops_on_board'])}"
            ]

            y_offset = 30
            for text in summary_text:
                cv2.putText(annotated_frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Display cards in hand with positions
            if detection_summary['cards_in_hand']:
                y_offset += 10
                cv2.putText(annotated_frame, "Cards in hand (L->R):", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25

                for position, card in enumerate(detection_summary['cards_in_hand']):
                    text = f"[{position}] {card['name']}"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # Draw position number on the card itself
                    card_x = int(card['center_x'])
                    card_y = int(card['center_y'])
                    cv2.putText(annotated_frame, f"[{position}]", (card_x - 20, card_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.circle(annotated_frame, (card_x, card_y), 5, (0, 255, 0), -1)

                    y_offset += 20

            # Display filtered cards (up next card)
            if detection_summary.get('cards_filtered'):
                y_offset += 10
                cv2.putText(annotated_frame, "Filtered (up next):", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_offset += 25

                for card in detection_summary['cards_filtered']:
                    text = f"{card['name']} (size: {int(card['area'])})"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Mark filtered card with red X
                    card_x = int(card['center_x'])
                    card_y = int(card['center_y'])
                    cv2.putText(annotated_frame, "X", (card_x - 10, card_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.circle(annotated_frame, (card_x, card_y), 5, (0, 0, 255), -1)

                    y_offset += 20

            # Display troops with tile coordinates
            if detection_summary['troops_on_board']:
                y_offset += 10
                cv2.putText(annotated_frame, "Troops:", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25

                for troop in detection_summary['troops_on_board'][:5]:  # Show first 5
                    tile_x, tile_y = troop['tile_coords']
                    text = f"{troop['color']} {troop['name']}: ({tile_x},{tile_y})"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 20

            cv2.imshow("Window Capture", annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current board state for training
                board_state = game_board.get_board_state()
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                training_entry = {
                    'timestamp': timestamp,
                    'frame': frame_count,
                    'board_state': board_state,
                    'detection_summary': detection_summary
                }
                training_data.append(training_entry)

                # Save to file
                with open(f'training_data_{timestamp}.txt', 'w') as f:
                    f.write(board_state)
                    f.write("\n\n=== RAW DETECTION DATA ===\n")
                    f.write(str(detection_summary))

                print(f"Saved board state to training_data_{timestamp}.txt")

            elif key == ord('d'):
                # Display current board state in console
                print("\n" + "=" * 50)
                print(game_board.get_board_state())
                print("=" * 50 + "\n")

            frame_count += 1

    cv2.destroyAllWindows()

    # Save all training data if enabled
    if save_training_data and training_data:
        with open('all_training_data.txt', 'w') as f:
            for entry in training_data:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Frame: {entry['frame']} | Time: {entry['timestamp']}\n")
                f.write(f"{'=' * 60}\n")
                f.write(entry['board_state'])
                f.write("\n")


def draw_tile_grid(frame, game_board):
    """
    Draw tile grid overlay on frame for debugging

    Args:
        frame: OpenCV frame to draw on
        game_board: GameBoard instance with tile dimensions

    Returns:
        Frame with grid overlay
    """
    color = (0, 255, 255)  # Yellow grid
    thickness = 1

    # Draw vertical lines
    for x in range(10):  # 0 to 9 (9 columns means 10 lines)
        x_pos = int(x * game_board.tile_width)
        cv2.line(frame, (x_pos, 0), (x_pos, frame.shape[0]), color, thickness)

    # Draw horizontal lines
    for y in range(17):  # 0 to 16 (16 rows means 17 lines)
        y_pos = int(y * game_board.tile_height)
        cv2.line(frame, (0, y_pos), (frame.shape[1], y_pos), color, thickness)

    # Add tile coordinate labels
    for x in range(9):
        for y in range(16):
            label = f"{x},{y}"
            x_pos = int((x + 0.5) * game_board.tile_width)
            y_pos = int((y + 0.5) * game_board.tile_height)

            # Only show labels for every other tile to reduce clutter
            if (x + y) % 2 == 0:
                cv2.putText(frame, label, (x_pos - 20, y_pos + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame

