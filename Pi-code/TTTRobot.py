import time
import numpy as np
import cv2
import requests
import json
import os
from adafruit_servokit import ServoKit
from picamera import PiCamera
from io import BytesIO
from PIL import Image

# Constants for servo positions
SERVO_BASE = 0       # Base rotation (clockwise/counterclockwise)
SERVO_SHOULDER = 1   # Shoulder joint (up/down)
SERVO_ELBOW = 2      # Elbow joint (up/down)
SERVO_WRIST_PITCH = 3  # Wrist pitch (up/down)
SERVO_WRIST_ROLL = 4   # Wrist roll (rotation)
SERVO_CLAW = 5       # Claw (open/close)

# Initialize the servo kit
kit = ServoKit(channels=16)

# Game board (3x3)
# 0 = empty, 1 = player (blue), 2 = robot (red)
board = np.zeros((3, 3), dtype=int)

# Configure OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set")

# Initialize camera
try:
    camera = PiCamera()
    camera.resolution = (1024, 768)
    camera.start_preview()
    # Allow camera to warm up
    time.sleep(2)
    print("Camera initialized successfully")
except Exception as e:
    print(f"Error initializing camera: {e}")
    print("Camera simulation mode will be used")
    camera = None

# Physical coordinates for each position on the board (x, y, z)
# These would be calibrated based on actual arm mechanics - ADJUST THESE VALUES
board_positions = {
    (0, 0): {"base": 30, "shoulder": 90, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (0, 1): {"base": 30, "shoulder": 100, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (0, 2): {"base": 30, "shoulder": 110, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (1, 0): {"base": 60, "shoulder": 90, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (1, 1): {"base": 60, "shoulder": 100, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (1, 2): {"base": 60, "shoulder": 110, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (2, 0): {"base": 90, "shoulder": 90, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (2, 1): {"base": 90, "shoulder": 100, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    (2, 2): {"base": 90, "shoulder": 110, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
}

# Storage positions for blue and red blocks - ADJUST THESE VALUES
# Positions should be spaced appropriately for your physical setup
blue_storage = {
    1: {"base": 150, "shoulder": 80, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    2: {"base": 150, "shoulder": 90, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    3: {"base": 150, "shoulder": 100, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    4: {"base": 150, "shoulder": 110, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    5: {"base": 150, "shoulder": 120, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
}

red_storage = {
    1: {"base": 180, "shoulder": 80, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    2: {"base": 180, "shoulder": 90, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    3: {"base": 180, "shoulder": 100, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    4: {"base": 180, "shoulder": 110, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
    5: {"base": 180, "shoulder": 120, "elbow": 45, "wrist_pitch": 0, "wrist_roll": 90},
}

# Track which blocks have been used - now with 5 blocks each
blue_blocks_used = [False, False, False, False, False]  # Index 0 = block 1, etc.
red_blocks_used = [False, False, False, False, False]   # Index 0 = block 1, etc.

# Home position
home_position = {"base": 120, "shoulder": 45, "elbow": 90, "wrist_pitch": 0, "wrist_roll": 90}

def move_servo_smoothly(servo_num, target_angle, current_angle=None, steps=10, delay=0.05):
    """Move a servo smoothly from current position to target position."""
    if current_angle is None:
        # Try to get current angle, or use target as starting point if not available
        try:
            current_angle = kit.servo[servo_num].angle
        except:
            current_angle = target_angle
    
    # Calculate step size
    step_size = (target_angle - current_angle) / steps
    
    # Move in steps
    for step in range(steps):
        angle = current_angle + step_size * (step + 1)
        kit.servo[servo_num].angle = angle
        time.sleep(delay)

def move_arm_to_position(position, smooth=True):
    """Move the arm to a specified position."""
    print(f"Moving arm to position: {position}")
    
    if smooth:
        # Move each servo smoothly to target position
        move_servo_smoothly(SERVO_BASE, position["base"])
        move_servo_smoothly(SERVO_SHOULDER, position["shoulder"])
        move_servo_smoothly(SERVO_ELBOW, position["elbow"])
        move_servo_smoothly(SERVO_WRIST_PITCH, position["wrist_pitch"])
        move_servo_smoothly(SERVO_WRIST_ROLL, position["wrist_roll"])
    else:
        # Set servo angles directly
        kit.servo[SERVO_BASE].angle = position["base"]
        kit.servo[SERVO_SHOULDER].angle = position["shoulder"]
        kit.servo[SERVO_ELBOW].angle = position["elbow"]
        kit.servo[SERVO_WRIST_PITCH].angle = position["wrist_pitch"]
        kit.servo[SERVO_WRIST_ROLL].angle = position["wrist_roll"]
        time.sleep(1)
    
    print("Arm in position")

def open_claw():
    """Open the claw."""
    print("Opening claw")
    move_servo_smoothly(SERVO_CLAW, 0)

def close_claw():
    """Close the claw."""
    print("Closing claw")
    move_servo_smoothly(SERVO_CLAW, 90)

def get_next_available_block(is_robot_block):
    """Get the next available block index."""
    blocks_used = red_blocks_used if is_robot_block else blue_blocks_used
    
    for i, used in enumerate(blocks_used):
        if not used:
            blocks_used[i] = True
            return i + 1  # Return 1-based index
    
    print("Warning: No more blocks available!")
    return None

def pick_block(is_robot_block):
    """Pick up a block from storage."""
    # Get next available block
    block_num = get_next_available_block(is_robot_block)
    
    if block_num is None:
        print("Error: No blocks available!")
        return False
    
    # Move to appropriate storage location
    if is_robot_block:
        print(f"Picking up red block #{block_num}")
        move_arm_to_position(red_storage[block_num])
    else:
        print(f"Picking up blue block #{block_num}")
        move_arm_to_position(blue_storage[block_num])
    
    # Open claw, lower to block, close claw, raise
    open_claw()
    time.sleep(0.5)
    
    # Lower wrist slightly to grab block
    current_wrist = kit.servo[SERVO_WRIST_PITCH].angle
    move_servo_smoothly(SERVO_WRIST_PITCH, current_wrist + 30)
    time.sleep(0.5)
    
    close_claw()
    time.sleep(0.5)
    
    # Raise wrist back up
    move_servo_smoothly(SERVO_WRIST_PITCH, current_wrist)
    time.sleep(0.5)
    
    return True

def place_block(row, col):
    """Place a block at the specified board position."""
    print(f"Placing block at position ({row}, {col})")
    
    # Move to the board position
    move_arm_to_position(board_positions[(row, col)])
    
    # Get current wrist position
    current_wrist = kit.servo[SERVO_WRIST_PITCH].angle
    
    # Lower wrist, open claw, raise wrist
    move_servo_smoothly(SERVO_WRIST_PITCH, current_wrist + 30)
    time.sleep(0.5)
    
    open_claw()
    time.sleep(0.5)
    
    move_servo_smoothly(SERVO_WRIST_PITCH, current_wrist)
    time.sleep(0.5)

def move_to_home():
    """Move the arm to home position."""
    print("Moving to home position")
    move_arm_to_position(home_position)

def capture_image():
    """Capture an image using the Pi camera."""
    print("Capturing image of the board...")
    
    if camera is None:
        print("Camera not available, returning test image")
        # Return a placeholder/test image path in simulation mode
        return "test_board_image.jpg"
    
    try:
        # Create a buffer to store the image
        image_stream = BytesIO()
        camera.capture(image_stream, format='jpeg')
        image_stream.seek(0)
        
        # Save the image
        image_path = f"board_state_{int(time.time())}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_stream.getvalue())
        
        print(f"Image captured and saved as {image_path}")
        return image_path
    
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def analyze_board_with_vision_api(image_path):
    """
    Use the OpenAI Vision API to analyze the board state from an image.
    Returns a 3x3 numpy array representing the board state.
    """
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key not set")
        return None
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return None
    
    try:
        # Prepare the API request
        url = "https://api.openai.com/v1/chat/completions"
        
        # Read the image and encode as base64
        with open(image_path, "rb") as image_file:
            import base64
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "This is a tic-tac-toe board with red and blue blocks. Red blocks are the robot's pieces, blue blocks are the player's pieces. Please analyze the image and return the exact positions of all blocks on the board in JSON format. The format should be a 3x3 grid where 0 = empty, 1 = blue (player), 2 = red (robot). For example: [[0,1,0],[0,2,0],[1,0,0]] would mean blue blocks at top-middle and bottom-left, and a red block in the middle center."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        print("Sending image to OpenAI Vision API...")
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response_data = response.json()
        
        # Extract the response content
        if 'choices' in response_data and len(response_data['choices']) > 0:
            content = response_data['choices'][0]['message']['content']
            
            # Extract the JSON data from the response
            import re
            json_match = re.search(r'\[\s*\[.*?\]\s*\]', content, re.DOTALL)
            
            if json_match:
                board_str = json_match.group(0)
                print(f"Detected board state: {board_str}")
                try:
                    # Parse the JSON board state
                    board_data = json.loads(board_str)
                    return np.array(board_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON board state: {e}")
            else:
                print("Could not extract board state from API response")
                print(f"API response: {content}")
        else:
            print(f"Unexpected API response format: {response_data}")
        
    except Exception as e:
        print(f"Error analyzing board with Vision API: {e}")
    
    return None

def analyze_board_state():
    """
    Analyze the board state by capturing an image and using the Vision API.
    """
    # Capture an image of the board
    image_path = capture_image()
    
    if image_path:
        # Analyze the image with the Vision API
        detected_board = analyze_board_with_vision_api(image_path)
        
        if detected_board is not None:
            global board
            board = detected_board
            return True
    
    print("Failed to analyze board state")
    return False

def detect_player_move(prev_board):
    """
    Detect the player's move by comparing the previous board state with the current one.
    Returns (row, col) of the move, or None if no valid move was detected.
    """
    # Analyze current board state
    if not analyze_board_state():
        print("Failed to detect player's move")
        return None
    
    # Find the difference between previous and current board
    for row in range(3):
        for col in range(3):
            if prev_board[row, col] == 0 and board[row, col] == 1:
                print(f"Detected player move at ({row}, {col})")
                return row, col
    
    print("No valid player move detected")
    return None

def wait_for_player_move(prev_board, timeout=60, check_interval=5):
    """
    Wait for the player to make a move within the timeout period.
    Returns (row, col) of the move, or None if timeout occurs.
    """
    print("\nWaiting for player to make a move...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Check if player has made a move
        move = detect_player_move(prev_board.copy())
        if move:
            return move
        
        # Wait before checking again
        print(f"No move detected yet. Waiting {check_interval} seconds...")
        time.sleep(check_interval)
    
    print("Timeout waiting for player move")
    return None

def check_win(board, player):
    """Check if the specified player has won."""
    # Check rows
    for row in range(3):
        if np.all(board[row, :] == player):
            return True
    
    # Check columns
    for col in range(3):
        if np.all(board[:, col] == player):
            return True
    
    # Check diagonals
    if board[0, 0] == player and board[1, 1] == player and board[2, 2] == player:
        return True
    if board[0, 2] == player and board[1, 1] == player and board[2, 0] == player:
        return True
    
    return False

def check_draw(board):
    """Check if the game is a draw."""
    return 0 not in board

def get_robot_move():
    """Determine the robot's next move."""
    # Check if robot can win
    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                board[row, col] = 2  # Try placing a robot piece
                if check_win(board, 2):
                    board[row, col] = 0  # Reset the cell
                    return row, col
                board[row, col] = 0  # Reset the cell
    
    # Check if player can win and block
    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                board[row, col] = 1  # Try placing a player piece
                if check_win(board, 1):
                    board[row, col] = 0  # Reset the cell
                    return row, col
                board[row, col] = 0  # Reset the cell
    
    # Try to take center
    if board[1, 1] == 0:
        return 1, 1
    
    # Try to take corners
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    import random
    random.shuffle(corners)
    for row, col in corners:
        if board[row, col] == 0:
            return row, col
    
    # Take any available edge
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    random.shuffle(edges)
    for row, col in edges:
        if board[row, col] == 0:
            return row, col

def display_board():
    """Display the current board state."""
    symbols = {0: " ", 1: "X", 2: "O"}
    print("\nCurrent Board:")
    print("-------------")
    for row in range(3):
        print("|", end=" ")
        for col in range(3):
            print(f"{symbols[board[row, col]]}", end=" | ")
        print("\n-------------")

def reset_game():
    """Reset the game state."""
    global board, blue_blocks_used, red_blocks_used
    board = np.zeros((3, 3), dtype=int)
    blue_blocks_used = [False, False, False, False, False]  # Reset to 5 blocks
    red_blocks_used = [False, False, False, False, False]   # Reset to 5 blocks

def calibrate_servos():
    """Run a simple calibration routine to test all servos."""
    print("Starting servo calibration...")
    
    # Test base servo
    print("Testing base servo...")
    move_servo_smoothly(SERVO_BASE, 0)
    time.sleep(1)
    move_servo_smoothly(SERVO_BASE, 180)
    time.sleep(1)
    move_servo_smoothly(SERVO_BASE, 90)
    time.sleep(1)
    
    # Test shoulder servo
    print("Testing shoulder servo...")
    move_servo_smoothly(SERVO_SHOULDER, 45)
    time.sleep(1)
    move_servo_smoothly(SERVO_SHOULDER, 90)
    time.sleep(1)
    
    # Test elbow servo
    print("Testing elbow servo...")
    move_servo_smoothly(SERVO_ELBOW, 45)
    time.sleep(1)
    move_servo_smoothly(SERVO_ELBOW, 90)
    time.sleep(1)
    
    # Test wrist pitch
    print("Testing wrist pitch servo...")
    move_servo_smoothly(SERVO_WRIST_PITCH, 0)
    time.sleep(1)
    move_servo_smoothly(SERVO_WRIST_PITCH, 45)
    time.sleep(1)
    
    # Test wrist roll
    print("Testing wrist roll servo...")
    move_servo_smoothly(SERVO_WRIST_ROLL, 0)
    time.sleep(1)
    move_servo_smoothly(SERVO_WRIST_ROLL, 90)
    time.sleep(1)
    
    # Test claw
    print("Testing claw servo...")
    open_claw()
    time.sleep(1)
    close_claw()
    time.sleep(1)
    
    print("Servo calibration complete")
    move_to_home()

def test_block_positions():
    """Test moving to each block storage position."""
    print("Testing block storage positions...")
    
    # Test blue block positions
    for i in range(1, 6):
        print(f"Moving to blue block {i} position...")
        move_arm_to_position(blue_storage[i])
        time.sleep(1)
    
    # Return home
    move_to_home()
    time.sleep(1)
    
    # Test red block positions
    for i in range(1, 6):
        print(f"Moving to red block {i} position...")
        move_arm_to_position(red_storage[i])
        time.sleep(1)
    
    # Return home
    move_to_home()
    print("Block position test complete")

def run_game():
    """Run the actual tic-tac-toe game with robot arm and vision detection."""
    global board
    reset_game()
    game_over = False
    
    print("Starting new tic-tac-toe game")
    move_to_home()
    
    # Initial board scan to make sure game is empty
    if not analyze_board_state():
        print("Failed to analyze initial board state")
        return False
    
    # Ensure the board is empty (or close to it) before starting
    if np.sum(board) > 0:
        print("Warning: Board does not appear to be empty")
        display_board()
        clear = input("Continue anyway? (y/n): ")
        if clear.lower() != 'y':
            return False
    
    print("Waiting for player to make first move...")
    
    while not game_over:
        # Store current board state for comparison
        prev_board = board.copy()
        
        # Wait for player's move
        player_move = wait_for_player_move(prev_board)
        
        if player_move is None:
            print("No player move detected within timeout. Game aborted.")
            return False
        
        # Update display
        display_board()
        
        # Check if player won or game is a draw
        if check_win(board, 1):
            print("Player wins!")
            game_over = True
        elif check_draw(board):
            print("It's a draw!")
            game_over = True
        else:
            # Robot's turn
            print("\nRobot's turn...")
            
            # Get robot's move
            robot_row, robot_col = get_robot_move()
            print(f"Robot decides to place at position ({robot_row}, {robot_col})")
            
            # Execute the move with the robotic arm
            if pick_block(True):  # Pick up a red block
                place_block(robot_row, robot_col)
                move_to_home()
                
                # Update the board state
                board[robot_row, robot_col] = 2
                display_board()
                
                # Check if robot won or game is a draw
                if check_win(board, 2):
                    print("Robot wins!")
                    game_over = True
                elif check_draw(board):
                    print("It's a draw!")
                    game_over = True
            else:
                print("Robot couldn't complete move - out of blocks")
                game_over = True
    
    print("Game over!")
    return True

def main():
    """Main function to run the tic-tac-toe robot."""
    try:
        print("Tic-Tac-Toe Robot Arm starting up...")
        
        # Check if API key is set
        if not OPENAI_API_KEY:
            api_key = input("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Display menu
        while True:
            print("\n=== Tic-Tac-Toe Robot Menu ===")
            print("1. Run servo calibration")
            print("2. Test block positions")
            print("3. Play game")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                calibrate_servos()
            elif choice == '2':
                test_block_positions()
            elif choice == '3':
                # Run the game
                while True:
                    run_game()
                    
                    # Ask if user wants to play again
                    play_again = input("Play again? (y/n): ")
                    if play_again.lower() != 'y':
                        break
                    
                    # Reset for new game
                    reset_game()
            elif choice == '4':
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Return to home position before exiting
        try:
            move_to_home()
            print("Servos returned to home position")
            
            # Clean up camera if used
            if camera:
                camera.close()
                print("Camera resources released")
        except:
            pass

if __name__ == "__main__":
    main()
