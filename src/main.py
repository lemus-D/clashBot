from src.windowCap import start_window_cap


#Defining the capture window
WINDOW_TITLE = 'BlueStacks App Player 1'

def main():


    # windowCap does its thing (detects cards, places cards)
    # game_state does its thing (tracks time/elixir)
    # ***IMPORTANT*** They don't talk to each other yet ***IMPORTANT***
    start_window_cap(WINDOW_TITLE, enable_mouse_control=True)


if __name__ == '__main__':
    main()

