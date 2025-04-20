import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image
import io
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

class TicTacToeDetector:
    def __init__(self):
        logger.info("Initializing Tic-Tac-Toe detector...")
        
    def detect(self, image):
        """
        Detect tic-tac-toe board and symbols in the image using basic computer vision
        """
        logger.info("Detecting tic-tac-toe board in image...")
        
        # Save the original image for debugging
        debug_image = image.copy()
        visualization = image.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement to make faint lines more visible
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Apply adaptive thresholding with different parameters for light drawings
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Apply morphological operations to enhance the grid lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Try to detect the grid using Hough lines
        edges = cv2.Canny(eroded, 50, 150, apertureSize=3)
        
        # Use HoughLinesP to detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=50, 
            minLineLength=min(image.shape[0], image.shape[1]) // 6, 
            maxLineGap=20
        )
        
        # Draw detected lines on visualization image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on visualization image
        cv2.drawContours(visualization, contours, -1, (255, 0, 0), 2)
        
        # If no significant contours found, try direct cell analysis
        if not contours or max(cv2.contourArea(c) for c in contours) < 1000:
            logger.info("No significant contours found, trying direct cell analysis")
            board, cell_visualizations, confidences = self._analyze_cells_directly(eroded, visualization)
            
            # Convert visualization to base64 for frontend
            _, buffer = cv2.imencode('.jpg', visualization)
            visualization_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return board, visualization_base64, cell_visualizations, confidences
        
        # Find the largest contour which should be the board
        board_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(board_contour)
        
        # Draw the bounding rectangle on visualization
        cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # If the aspect ratio is too far from 1:1, it's probably not a tic-tac-toe board
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            logger.info(f"Aspect ratio {aspect_ratio} is not close to 1:1, trying direct cell analysis")
            board, cell_visualizations, confidences = self._analyze_cells_directly(eroded, visualization)
            
            # Convert visualization to base64 for frontend
            _, buffer = cv2.imencode('.jpg', visualization)
            visualization_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return board, visualization_base64, cell_visualizations, confidences
        
        # Create a square region of interest
        size = max(w, h)
        roi = eroded[y:y+size, x:x+size]
        
        # Resize to a standard size for easier processing
        standard_size = 300
        roi_resized = cv2.resize(roi, (standard_size, standard_size))
        
        # Divide the board into 9 cells
        cell_size = standard_size // 3
        board = []
        cell_visualizations = []
        confidences = []
        
        for row in range(3):
            for col in range(3):
                # Extract the cell
                cell_y = row * cell_size
                cell_x = col * cell_size
                cell = roi_resized[cell_y:cell_y+cell_size, cell_x:cell_x+cell_size]
                
                # Create a visualization of this cell
                cell_vis = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                cell_vis[:, :, 0] = cell  # Blue channel
                cell_vis[:, :, 1] = cell  # Green channel
                cell_vis[:, :, 2] = cell  # Red channel
                
                # Determine if the cell contains X, O, or is empty
                symbol, confidence = self._classify_cell(cell)
                
                # Add text to cell visualization
                text = f"{symbol} ({confidence:.2f})"
                cv2.putText(cell_vis, text, (5, cell_size-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Convert cell visualization to base64
                _, buffer = cv2.imencode('.jpg', cell_vis)
                cell_vis_base64 = base64.b64encode(buffer).decode('utf-8')
                cell_visualizations.append(cell_vis_base64)
                
                board.append(symbol)
                confidences.append(float(confidence))
        
        # Draw the grid on the visualization
        for i in range(1, 3):
            # Horizontal lines
            cv2.line(visualization, 
                    (x, y + i * size // 3), 
                    (x + size, y + i * size // 3), 
                    (255, 255, 0), 2)
            # Vertical lines
            cv2.line(visualization, 
                    (x + i * size // 3, y), 
                    (x + i * size // 3, y + size), 
                    (255, 255, 0), 2)
        
        # Convert visualization to base64 for frontend
        _, buffer = cv2.imencode('.jpg', visualization)
        visualization_base64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.info(f"Detected board state: {board}")
        return board, visualization_base64, cell_visualizations, confidences
    
    def _analyze_cells_directly(self, binary_image, visualization):
        """Analyze the image by dividing it into a 3x3 grid directly"""
        height, width = binary_image.shape
        cell_height = height // 3
        cell_width = width // 3
        
        board = []
        cell_visualizations = []
        confidences = []
        
        # Draw the grid on the visualization
        for i in range(1, 3):
            # Horizontal lines
            cv2.line(visualization, 
                    (0, i * height // 3), 
                    (width, i * height // 3), 
                    (255, 255, 0), 2)
            # Vertical lines
            cv2.line(visualization, 
                    (i * width // 3, 0), 
                    (i * width // 3, height), 
                    (255, 255, 0), 2)
        
        for row in range(3):
            for col in range(3):
                # Extract the cell
                cell_y = row * cell_height
                cell_x = col * cell_width
                cell = binary_image[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width]
                
                # Create a visualization of this cell
                cell_vis = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
                cell_vis[:, :, 0] = cell  # Blue channel
                cell_vis[:, :, 1] = cell  # Green channel
                cell_vis[:, :, 2] = cell  # Red channel
                
                # Determine if the cell contains X, O, or is empty
                symbol, confidence = self._classify_cell(cell)
                
                # Add text to cell visualization
                text = f"{symbol} ({confidence:.2f})"
                cv2.putText(cell_vis, text, (5, cell_height-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Convert cell visualization to base64
                _, buffer = cv2.imencode('.jpg', cell_vis)
                cell_vis_base64 = base64.b64encode(buffer).decode('utf-8')
                cell_visualizations.append(cell_vis_base64)
                
                board.append(symbol)
                confidences.append(float(confidence))
        
        logger.info(f"Direct cell analysis board state: {board}")
        return board, cell_visualizations, confidences
    
    def _classify_cell(self, cell):
        """Classify a cell as X, O, or empty and return confidence score"""
        # Resize for consistent processing
        cell_resized = cv2.resize(cell, (100, 100))
        
        # Count white pixels (foreground)
        white_pixels = cv2.countNonZero(cell_resized)
        
        # Calculate the percentage of white pixels
        white_percentage = white_pixels / (100 * 100)
        
        # MUCH more aggressive empty space detection - if less than 8% white pixels, consider it empty
        if white_percentage < 0.08:
            return '', 1.0 - white_percentage
        
        # Create templates for X and O
        x_template = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(x_template, (20, 20), (80, 80), 255, 5)
        cv2.line(x_template, (20, 80), (80, 20), 255, 5)
        
        o_template = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(o_template, (50, 50), 30, 255, 5)
        
        # Match the cell against the templates
        x_match = cv2.matchTemplate(cell_resized, x_template, cv2.TM_CCOEFF_NORMED)
        o_match = cv2.matchTemplate(cell_resized, o_template, cv2.TM_CCOEFF_NORMED)
        
        x_score = np.max(x_match)
        o_score = np.max(o_match)
        
        # Alternative approach: check for diagonal lines (X) vs circular patterns (O)
        # Hough lines for X detection
        edges = cv2.Canny(cell_resized, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
        
        # Hough circles for O detection
        circles = cv2.HoughCircles(
            cell_resized, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=10, minRadius=20, maxRadius=40
        )
        
        # Count diagonal lines for X
        diagonal_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Check if line is diagonal (around 45 or 135 degrees)
                if (angle > 30 and angle < 60) or (angle > 120 and angle < 150):
                    diagonal_count += 1
        
        # Check if circles were found for O
        has_circle = circles is not None
        
        # Calculate center of mass for distribution analysis
        moments = cv2.moments(cell_resized)
        center_dist = 100  # Default high value
        
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center_dist = np.sqrt((cx - 50)**2 + (cy - 50)**2)
        
        # Decision logic combining all methods
        # Improved empty space detection - check white percentage at multiple thresholds
        if white_percentage < 0.08:
            return '', 0.95
        
        # Require higher confidence for symbol detection
        x_combined_score = x_score * 0.4 + (diagonal_count / 4) * 0.4 + (center_dist / 50) * 0.2
        o_combined_score = o_score * 0.4 + (1 if has_circle else 0) * 0.4 + (1 - center_dist / 50) * 0.2
        
        # If both scores are low, it might be empty
        if x_combined_score < 0.3 and o_combined_score < 0.3:
            return '', 0.7
        
        # Otherwise, decide between X and O with higher threshold
        if x_combined_score > o_combined_score and x_combined_score > 0.3:
            return 'X', x_combined_score
        elif o_combined_score > 0.3:
            return 'O', o_combined_score
        else:
            return '', 0.6  # Default to empty if confidence is low

# Initialize detector
detector = TicTacToeDetector()

@app.route('/')
def index():
    """Serve the main HTML page"""
    with open('index.html', 'r') as f:
        return f.read()

@app.route('/process_board', methods=['POST'])
def process_board():
    """Process the captured image and detect the tic-tac-toe board"""
    try:
        # Get image data from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove the data:image/jpeg;base64, prefix
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image with detector
        board, visualization_base64, cell_visualizations, confidences = detector.detect(opencv_image)
        
        # Return detected board state and visualization
        return jsonify({
            'success': True,
            'board': board,
            'confidences': confidences,
            'visualization': f"data:image/jpeg;base64,{visualization_base64}",
            'cellVisualizations': [f"data:image/jpeg;base64,{vis}" for vis in cell_visualizations]
        })
    
    except Exception as e:
        logger.error(f"Error processing board: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    logger.info("Starting Tic-Tac-Toe with Camera server...")
    app.run(debug=True)
