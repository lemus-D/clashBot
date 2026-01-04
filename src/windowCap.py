import cv2
import numpy as np
import pywinctl as gw
import mss

def start_window_cap(window_name):
    try:
        # actually gets the window
        window = gw.getWindowsWithTitle(window_name)[0]

        # checks if the window is minimized and if it is makes it opens the window back up
        if window.isMinimized:
            window.restore()

        # brings the window to the top of the stack of windows
        window.activate()

    except IndexError:
        print('No window found')
        exit(0)

    # starts to capture the feed
    with mss.mss() as sct:
        print("Press 'q' to quit.")

        while True:
            # Define the capture region using the window's current coordinates
            # update this every frame in case the window moves around
            monitor = {
                "top": window.top + 50,
                "left": window.left + 383,
                "width": window.width -433,
                "height": window.height -50
            }

            # Grab the screenshot
            screenshot = sct.grab(monitor)

            # Convert to an OpenCV/Numpy compatible format
            # mss returns BGRA, OpenCV uses BGR
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Display the result
            cv2.imshow("Window Capture", img)

            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
