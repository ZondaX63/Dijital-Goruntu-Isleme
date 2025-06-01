import cv2
import numpy as np

def detect_lane_lines(image, rho=1, theta=np.pi/180, threshold=50, min_line_length=100, max_line_gap=10):
    """
    Detect lane lines using Hough Transform
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    # Create output image
    result = image.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return result

def detect_eyes(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect eyes using Haar Cascade Classifier
    """
    # Load eye cascade classifier
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    
    # Create output image
    result = image.copy()
    
    # Draw rectangles around eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return result

def detect_circles(image, dp=1.2, min_dist=100, param1=50, param2=30, min_radius=0, max_radius=0):
    """
    Detect circles using Hough Circle Transform
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Create output image
    result = image.copy()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return result 