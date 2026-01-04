import cv2
import numpy as np
import pywinctl as gw
import mss
import windowCap
from src.windowCap import start_window_cap

#Defining the capture window
WINDOW_TITLE = 'BlueStacks App Player 1'

def main():
    start_window_cap(WINDOW_TITLE)

if __name__ == '__main__':
    main()