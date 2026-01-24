from cardClasses import BlankSpace, Card, Troop


class GameBoard:
    def __init__(self, monitor_width, monitor_height):
        """
        Initialize game board with monitor dimensions for coordinate conversion

        Args:
            monitor_width: Width of the captured window region in pixels
            monitor_height: Height of the captured window region in pixels
        """
        self.cards_in_hand = [BlankSpace(), BlankSpace(), BlankSpace(), BlankSpace()]
        self.troops_in_arena = [[BlankSpace() for _ in range(9)] for _ in range(16)]

        # Store monitor dimensions for coordinate conversion
        self.monitor_width = monitor_width
        self.monitor_height = monitor_height

        # Calculate tile dimensions
        self.tile_width = monitor_width / 9
        self.tile_height = monitor_height / 16

    def add_card_to_hand(self, card, position):
        """Add a card to hand at specified position (0-3)"""
        if 0 <= position < 4:
            self.cards_in_hand[position] = card
        else:
            print(f"Invalid hand position: {position}. Must be 0-3.")

    def add_troop_to_arena(self, troop, x_cord, y_cord):
        """Add a troop to arena at tile coordinates"""
        if 0 <= x_cord < 9 and 0 <= y_cord < 16:
            self.troops_in_arena[y_cord][x_cord] = troop
        else:
            print(f"Invalid arena position: ({x_cord}, {y_cord})")

    def convert_image_cord_to_tile(self, x_image_cord, y_image_cord):
        """
        Convert pixel coordinates from visual detection to game board tile coordinates

        Args:
            x_image_cord: X pixel coordinate from detection (0 to monitor_width)
            y_image_cord: Y pixel coordinate from detection (0 to monitor_height)

        Returns:
            Tuple of (tile_x, tile_y) where:
                - tile_x is column index (0-8)
                - tile_y is row index (0-15)
            Returns None if coordinates are out of bounds
        """
        # Validate input coordinates
        if x_image_cord < 0 or x_image_cord > self.monitor_width:
            print(f"Warning: X coordinate {x_image_cord} out of bounds [0, {self.monitor_width}]")
            return None
        if y_image_cord < 0 or y_image_cord > self.monitor_height:
            print(f"Warning: Y coordinate {y_image_cord} out of bounds [0, {self.monitor_height}]")
            return None

        # Convert pixel coordinates to tile indices
        tile_x = int(x_image_cord / self.tile_width)
        tile_y = int(y_image_cord / self.tile_height)

        # Clamp to valid tile range (safety check)
        tile_x = min(tile_x, 8)  # Max column index is 8 (0-8 = 9 columns)
        tile_y = min(tile_y, 15)  # Max row index is 15 (0-15 = 16 rows)

        return (tile_x, tile_y)

    def convert_tile_to_image_cord(self, tile_x, tile_y):
        """
        Convert game board tile coordinates to pixel coordinates (center of tile)
        Useful for placing cards at specific tiles

        Args:
            tile_x: Column index (0-8)
            tile_y: Row index (0-15)

        Returns:
            Tuple of (pixel_x, pixel_y) representing the center of the tile
            Returns None if tile coordinates are invalid
        """
        if tile_x < 0 or tile_x >= 9:
            print(f"Invalid tile_x: {tile_x}. Must be 0-8.")
            return None
        if tile_y < 0 or tile_y >= 16:
            print(f"Invalid tile_y: {tile_y}. Must be 0-15.")
            return None

        # Calculate center of tile
        pixel_x = int((tile_x + 0.5) * self.tile_width)
        pixel_y = int((tile_y + 0.5) * self.tile_height)

        return (pixel_x, pixel_y)

    def process_detections(self, detections):
        """
        Process Roboflow/Supervision detections and update game board

        Args:
            detections: Supervision Detections object from model inference

        Returns:
            Dictionary containing processed detection info:
                - 'cards_in_hand': List of detected cards in hand (ordered left to right)
                - 'cards_filtered': List of cards filtered out (up next card)
                - 'troops_on_board': List of troops with their tile positions
        """
        cards_detected = []
        cards_filtered = []
        troops_detected = []

        # Iterate through all detections
        for i in range(len(detections)):
            class_name = detections.data['class_name'][i]

            # Get bounding box coordinates
            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Calculate bounding box width and height
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_area = bbox_width * bbox_height

            # Check if it's a card in hand (starts with "card")
            if class_name.startswith('card'):
                card_name = class_name[4:]  # Remove "card" prefix
                cards_detected.append({
                    'name': card_name,
                    'pixel_coords': (center_x, center_y),
                    'bbox': bbox,
                    'width': bbox_width,
                    'height': bbox_height,
                    'area': bbox_area,
                    'center_x': center_x,
                    'center_y': center_y
                })

            # Check if it's a troop (starts with "blue" or "red")
            elif class_name.startswith('blue') or class_name.startswith('red'):
                # Extract color and troop name
                if class_name.startswith('blue'):
                    color = 'blue'
                    troop_name = class_name[4:]  # Remove "blue" prefix
                else:
                    color = 'red'
                    troop_name = class_name[3:]  # Remove "red" prefix

                # Convert to tile coordinates
                tile_coords = self.convert_image_cord_to_tile(center_x, center_y)

                if tile_coords:
                    troops_detected.append({
                        'name': troop_name,
                        'color': color,
                        'pixel_coords': (center_x, center_y),
                        'tile_coords': tile_coords
                    })

                    # Update the game board
                    troop = Troop(troop_name, color)
                    self.add_troop_to_arena(troop, tile_coords[0], tile_coords[1])

        # --- FILTER AND ORDER CARDS IN HAND ---
        if cards_detected:
            # Filter out the "up next" card based on size and position
            cards_in_hand = self.filter_cards_in_hand(cards_detected)

            # Sort remaining cards from left to right
            cards_in_hand.sort(key=lambda card: card['center_x'])

            # Update the game board with ordered cards
            for position, card_info in enumerate(cards_in_hand):
                if position < 4:  # Only first 4 cards
                    card = Card(card_info['name'], cost=0)  # Cost unknown from visual detection
                    self.add_card_to_hand(card, position)

            # Identify filtered cards
            cards_filtered = [c for c in cards_detected if c not in cards_in_hand]
        else:
            cards_in_hand = []

        return {
            'cards_in_hand': cards_in_hand,
            'cards_filtered': cards_filtered,
            'troops_on_board': troops_detected
        }

    def filter_cards_in_hand(self, cards_detected):
        """
        Filter out the "up next" card which is smaller and in lower left corner

        Args:
            cards_detected: List of detected card dictionaries

        Returns:
            List of cards that are actually in hand (max 4 cards)
        """
        if len(cards_detected) <= 4:
            return cards_detected

        # Calculate median card size to identify outliers
        areas = [card['area'] for card in cards_detected]
        median_area = sorted(areas)[len(areas) // 2]

        # Calculate median Y position (vertical)
        y_positions = [card['center_y'] for card in cards_detected]
        median_y = sorted(y_positions)[len(y_positions) // 2]

        # Filter cards based on size and position
        valid_cards = []

        for card in cards_detected:
            # Card should be:
            # 1. Similar size to median (not significantly smaller)
            # 2. At similar height (Y position close to median)
            # 3. Not in the far left corner (up next card position)

            size_threshold = 0.6  # Card must be at least 60% of median size
            y_threshold = self.monitor_height * 0.1  # Within 10% of screen height
            left_edge_threshold = self.monitor_width * 0.15  # Not in leftmost 15% of screen

            is_normal_size = card['area'] >= (median_area * size_threshold)
            is_similar_height = abs(card['center_y'] - median_y) < y_threshold
            not_far_left = card['center_x'] > left_edge_threshold

            if is_normal_size and is_similar_height and not_far_left:
                valid_cards.append(card)

        # If we still have more than 4 cards, keep the 4 largest
        if len(valid_cards) > 4:
            valid_cards.sort(key=lambda card: card['area'], reverse=True)
            valid_cards = valid_cards[:4]

        return valid_cards

    def get_board_state(self):
        """
        Get current state of the game board as a string representation
        Useful for debugging and AI training data

        Returns:
            String representation of the board
        """
        state = "=== CARDS IN HAND (Position 0-3, Left to Right) ===\n"
        for i, card in enumerate(self.cards_in_hand):
            if isinstance(card, Card):
                state += f"Position {i}: {card.name} (Cost: {card.cost})\n"
            else:
                state += f"Position {i}: Empty\n"

        state += "\n=== TROOPS IN ARENA (9x16 grid) ===\n"
        state += "   " + "".join(f"{i:3}" for i in range(9)) + "\n"

        for y in range(16):
            state += f"{y:2} "
            for x in range(9):
                troop = self.troops_in_arena[y][x]
                if isinstance(troop, Troop):
                    # Show first letter of color + first letter of name
                    state += f" {troop.color[0].upper()}{troop.name[0]} "
                else:
                    state += " . "
            state += "\n"

        return state

    def clear_arena(self):
        """Clear all troops from arena"""
        self.troops_in_arena = [[BlankSpace() for _ in range(9)] for _ in range(16)]

    def clear_hand(self):
        """Clear all cards from hand"""
        self.cards_in_hand = [BlankSpace(), BlankSpace(), BlankSpace(), BlankSpace()]