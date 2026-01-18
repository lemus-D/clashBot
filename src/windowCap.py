import cv2
import numpy as np
import pywinctl as gw
import mss
import supervision as sv
from inference import get_model
from dotenv import load_dotenv
import os

load_dotenv()

# --- CONFIGURATION ---
# Replace with your actual Project ID and Version Number (e.g., "troop-counter/2")
MODEL_ID = "troop-counter/7"
API_KEY = os.getenv('API_KEY')  # Find this in your Roboflow Settings


def start_window_cap(window_name):
    # 1. LOAD THE MODEL (Downloads weights once, then runs locally)
    print("Loading model... this may take a moment first time.")
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)

    # 2. SETUP ANNOTATORS (For drawing boxes on the screen)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    try:
        window = gw.getWindowsWithTitle(window_name)[0]
        if window.isMinimized:
            window.restore()
        window.activate()
    except IndexError:
        print('No window found')
        exit(0)

    with mss.mss() as sct:
        print("Press 'q' to quit.")

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


