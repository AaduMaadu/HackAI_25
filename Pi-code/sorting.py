# Two-Color Smart Sorting System with Raspberry Pi Camera Module 3
# For AI Hackathon with Robotic Arm - Black and Aqua Cubes
# Adjusted for inverted wrist servo

import time
import cv2
import numpy as np
from adafruit_servokit import ServoKit
import threading
import os
import picamera2  # Using picamera2 library for Pi Camera Module 3

# Initialize the servo kit
kit = ServoKit(channels=16)

# Define servo channels
BASE = 0
SHOULDER1 = 1
ELBOW = 8
WRIST = 9  # Inverted wrist (180 is up)
CLAW = 10

# Define positions - adjusted for inverted wrist
HOME_POSITION = {
    BASE: 15,
    SHOULDER1: 0,
    ELBOW: 0,
    WRIST: 180,  # Inverted wrist position
    CLAW: 40  # Open position
}

# Sorting positions
BLACK_POSITION = {BASE: 0, SHOULDER1: 15, ELBOW: 5, WRIST: 125}  # Left position
AQUA_POSITION = {BASE: 45, SHOULDER1: 15, ELBOW: 5, WRIST: 125}  # Right position
PICKUP_POSITION = {BASE: 15, SHOULDER1: 15, ELBOW: 5, WRIST: 125}  # Center position

# Initialize Raspberry Pi Camera
try:
    picam2 = picamera2.Picamera2()
    camera_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # Allow camera to initialize
    print("Camera initialized successfully")
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)

# Color ranges in HSV - will be calibrated
# Starting with reasonable defaults
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 100, 100])

# Aqua (blue-green) range
lower_aqua = np.array([80, 50, 50])
upper_aqua = np.array([110, 255, 255])

# Counters for sorting statistics
black_count = 0
aqua_count = 0
total_sorted = 0

def move_servo_smoothly(channel, target_angle, steps=10, delay=0.02):
    """Move servo smoothly from current position to target position"""
    current_angle = kit.servo[channel].angle
    if current_angle is None:
        current_angle = 0
    
    step_size = (target_angle - current_angle) / steps
    
    for i in range(steps):
        next_angle = current_angle + step_size
        kit.servo[channel].angle = next_angle
        current_angle = next_angle
        time.sleep(delay)
    
    # Ensure we reach the exact target
    kit.servo[channel].angle = target_angle

def move_to_position(position, delay=0.5):
    """Move the arm to a defined position"""
    # Move base first
    if BASE in position:
        move_servo_smoothly(BASE, position[BASE])
        time.sleep(delay)
    
    # Move shoulder and elbow together
    if SHOULDER1 in position and ELBOW in position:
        threading.Thread(target=move_servo_smoothly, args=(SHOULDER1, position[SHOULDER1])).start()
        threading.Thread(target=move_servo_smoothly, args=(ELBOW, position[ELBOW])).start()
        time.sleep(delay)
    elif SHOULDER1 in position:
        move_servo_smoothly(SHOULDER1, position[SHOULDER1])
        time.sleep(delay)
    elif ELBOW in position:
        move_servo_smoothly(ELBOW, position[ELBOW])
        time.sleep(delay)
    
    # Move wrist
    if WRIST in position:
        move_servo_smoothly(WRIST, position[WRIST])
        time.sleep(delay)
    
    # Operate claw last
    if CLAW in position:
        move_servo_smoothly(CLAW, position[CLAW])
        time.sleep(delay)

def move_home():
    """Move the arm to home position"""
    move_to_position(HOME_POSITION)

def grab_object():
    """Close the claw to grab an object"""
    move_servo_smoothly(CLAW, 7)  # Close position
    time.sleep(0.5)

def release_object():
    """Open the claw to release an object"""
    move_servo_smoothly(CLAW, 40)  # Open position
    time.sleep(0.5)

def capture_image():
    """Capture an image from the Pi Camera"""
    try:
        # Capture frame from Pi Camera 2
        frame = picam2.capture_array()
        # Convert to BGR format (from default RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame
    except Exception as e:
        print(f"Error capturing image: {e}")
        return False, None

def detect_cubes():
    """Detect black and aqua cubes in camera view"""
    ret, frame = capture_image()
    if not ret:
        print("Failed to capture image")
        return None, None
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for each color
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    aqua_mask = cv2.inRange(hsv, lower_aqua, upper_aqua)
    
    # Optional: Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    
    aqua_mask = cv2.morphologyEx(aqua_mask, cv2.MORPH_OPEN, kernel)
    aqua_mask = cv2.morphologyEx(aqua_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_aqua, _ = cv2.findContours(aqua_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cubes = []
    
    # Process black cubes
    for contour in contours_black:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cubes.append(('black', center_x, center_y, w * h))
    
    # Process aqua cubes
    for contour in contours_aqua:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cubes.append(('aqua', center_x, center_y, w * h))
    
    # Create a copy of the frame for visualization
    display_frame = frame.copy()
    
    # Draw results on frame for visualization
    for cube in cubes:
        color, x, y, area = cube
        color_bgr = (0, 0, 0) if color == 'black' else (255, 190, 0)  # black or aqua color
        cv2.circle(display_frame, (x, y), 10, color_bgr, -1)
        cv2.putText(display_frame, color, (x-25, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    
    # Add sorting statistics
    cv2.putText(display_frame, f"Black: {black_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(display_frame, f"Aqua: {aqua_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 190, 0), 2)
    cv2.putText(display_frame, f"Total: {total_sorted}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add masks as smaller images in the corners
    h, w = display_frame.shape[:2]
    small_h, small_w = h // 4, w // 4
    
    # Resize masks for display
    small_black_mask = cv2.resize(cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR), (small_w, small_h))
    small_aqua_mask = cv2.resize(cv2.cvtColor(aqua_mask, cv2.COLOR_GRAY2BGR), (small_w, small_h))
    
    # Place masks in corners
    display_frame[0:small_h, 0:small_w] = small_black_mask
    display_frame[0:small_h, w-small_w:w] = small_aqua_mask
    
    # Display the frame
    cv2.imshow("Cube Detection", display_frame)
    key = cv2.waitKey(1)
    
    # Allow calibration during detection
    if key == ord('c'):
        calibrate_colors()
    
    if cubes:
        # Find the largest cube (by area)
        largest_cube = max(cubes, key=lambda obj: obj[3])
        return largest_cube[0], (largest_cube[1], largest_cube[2])
    
    return None, None

def calibrate_colors():
    """Interactive calibration for the cube colors"""
    global lower_black, upper_black, lower_aqua, upper_aqua
    
    print("\n=== COLOR CALIBRATION ===")
    print("Place a BLACK cube in the center of the camera view")
    print("Press 'b' to calibrate BLACK")
    print("Place an AQUA cube in the center of the camera view")
    print("Press 'a' to calibrate AQUA")
    print("Press 'q' to finish calibration")
    
    calibrating = True
    
    while calibrating:
        ret, frame = capture_image()
        if not ret:
            print("Failed to capture image")
            continue
        
        # Draw a target rectangle in the center
        h, w = frame.shape[:2]
        roi_size = min(h, w) // 3
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place cube in box", (w//2-100, h//2-roi_size//2-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display current calibration values
        cv2.putText(frame, f"BLACK: {lower_black} - {upper_black}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f"AQUA: {lower_aqua} - {upper_aqua}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 0), 1)
        
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = hsv[y1:y2, x1:x2]
        
        if key == ord('b'):  # Calibrate black
            if roi.size > 0:
                # Calculate the color range based on the ROI
                hsv_values = roi.reshape(-1, 3)
                mean_val = np.mean(hsv_values, axis=0)
                std_val = np.std(hsv_values, axis=0)
                
                # Black is usually low value with any hue
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 100, 100])  # Adjusted for black
                
                print(f"Black calibrated: Lower {lower_black}, Upper {upper_black}")
            else:
                print("ROI is empty, calibration failed")
            
        elif key == ord('a'):  # Calibrate aqua
            if roi.size > 0:
                # Calculate the color range based on the ROI
                hsv_values = roi.reshape(-1, 3)
                mean_val = np.mean(hsv_values, axis=0)
                std_val = np.std(hsv_values, axis=0)
                
                # Aqua is usually in the blue-green hue range
                lower_aqua = np.array([max(0, mean_val[0] - 20), max(0, mean_val[1] - 50), max(0, mean_val[2] - 50)])
                upper_aqua = np.array([min(180, mean_val[0] + 20), min(255, mean_val[1] + 50), min(255, mean_val[2] + 50)])
                
                print(f"Aqua calibrated: Lower {lower_aqua}, Upper {upper_aqua}")
            else:
                print("ROI is empty, calibration failed")
                
        elif key == ord('q'):
            calibrating = False
            print("Calibration complete")
    
    cv2.destroyWindow("Calibration")
    
    # Save calibration values to file
    with open('color_calibration.txt', 'w') as f:
        f.write(f"BLACK_LOWER: {lower_black[0]},{lower_black[1]},{lower_black[2]}\n")
        f.write(f"BLACK_UPPER: {upper_black[0]},{upper_black[1]},{upper_black[2]}\n")
        f.write(f"AQUA_LOWER: {lower_aqua[0]},{lower_aqua[1]},{lower_aqua[2]}\n")
        f.write(f"AQUA_UPPER: {upper_aqua[0]},{upper_aqua[1]},{upper_aqua[2]}\n")

def load_calibration():
    """Load color calibration from file if exists"""
    global lower_black, upper_black, lower_aqua, upper_aqua
    
    if os.path.exists('color_calibration.txt'):
        try:
            with open('color_calibration.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('BLACK_LOWER:'):
                        values = line.split(':')[1].strip().split(',')
                        lower_black = np.array([int(values[0]), int(values[1]), int(values[2])])
                    elif line.startswith('BLACK_UPPER:'):
                        values = line.split(':')[1].strip().split(',')
                        upper_black = np.array([int(values[0]), int(values[1]), int(values[2])])
                    elif line.startswith('AQUA_LOWER:'):
                        values = line.split(':')[1].strip().split(',')
                        lower_aqua = np.array([int(values[0]), int(values[1]), int(values[2])])
                    elif line.startswith('AQUA_UPPER:'):
                        values = line.split(':')[1].strip().split(',')
                        upper_aqua = np.array([int(values[0]), int(values[1]), int(values[2])])
            print("Color calibration loaded from file")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    return False

def pick_and_sort():
    """Main function to detect, pick and sort cubes"""
    global black_count, aqua_count, total_sorted
    
    try:
        print("Starting cube sorting system")
        print("Press 'c' during detection to calibrate colors")
        print("Press 'q' to quit")
        
        # Load calibration if available
        if not load_calibration():
            print("No calibration file found. Starting with default values.")
            print("It's recommended to calibrate colors first.")
            calibrate_colors()
        
        move_home()
        time.sleep(1)
        
        while True:
            print("\nLooking for cubes...")
            
            # Move to pickup position for scanning
            move_to_position(PICKUP_POSITION)
            time.sleep(1)
            
            # Detect cubes
            cube_color, position = detect_cubes()
            
            # Check if user wants to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Program stopped by user")
                break
            elif key == ord('c'):
                calibrate_colors()
                continue
            
            if not cube_color:
                print("No cubes detected")
                time.sleep(1)
                continue
            
            print(f"Detected {cube_color} cube")
            
            # Pickup the cube
            print("Picking up cube...")
            # Approach cube
            move_to_position(PICKUP_POSITION)
            time.sleep(0.5)
            # Lower arm to grab
            move_servo_smoothly(SHOULDER1, 25)
            time.sleep(0.5)
            # Close gripper
            grab_object()
            time.sleep(0.5)
            # Lift cube
            move_servo_smoothly(SHOULDER1, 15)
            time.sleep(0.5)
            
            # Move to sorting position based on color
            print(f"Moving to {cube_color} position...")
            if cube_color == 'black':
                move_to_position(BLACK_POSITION)
                black_count += 1
            elif cube_color == 'aqua':
                move_to_position(AQUA_POSITION)
                aqua_count += 1
            
            total_sorted += 1
            
            # Release the cube
            move_servo_smoothly(SHOULDER1, 25)
            time.sleep(0.5)
            release_object()
            time.sleep(0.5)
            
            # Return to slightly raised position
            move_servo_smoothly(SHOULDER1, 15)
            time.sleep(0.5)
            
            # Move back to home position
            move_home()
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Program stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        print("Cleaning up...")
        move_home()
        cv2.destroyAllWindows()
        print(f"Final stats: Black: {black_count}, Aqua: {aqua_count}, Total: {total_sorted}")

if __name__ == "__main__":
    print("===== AI CUBE SORTING SYSTEM =====")
    print("1. Start sorting")
    print("2. Calibrate colors")
    print("3. Test arm movements")
    print("4. Exit")
    
    choice = input("Select an option: ")
    
    if choice == '1':
        pick_and_sort()
    elif choice == '2':
        calibrate_colors()
    elif choice == '3':
        print("Testing arm movements...")
        move_home()
        time.sleep(1)
        print("Moving to BLACK position")
        move_to_position(BLACK_POSITION)
        time.sleep(2)
        print("Moving to AQUA position")
        move_to_position(AQUA_POSITION)
        time.sleep(2)
        print("Moving to PICKUP position")
        move_to_position(PICKUP_POSITION)
        time.sleep(2)
        print("Testing grab and release")
        grab_object()
        time.sleep(1)
        release_object()
        time.sleep(1)
        print("Returning home")
        move_home()
    else:
        print("Exiting program")
