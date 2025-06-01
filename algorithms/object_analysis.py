import cv2
import numpy as np
from scipy import stats
import pandas as pd

def detect_dark_green_regions(image, hsv_lower=(35, 50, 50), hsv_upper=(85, 255, 255)):
    """
    Detect dark green regions in hyperspectral image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for dark green regions
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, mask

def calculate_region_features(image, contour):
    """
    Calculate features for a detected region
    """
    # Create mask for the region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Get region pixels
    region_pixels = image[mask > 0]
    
    # Calculate features
    features = {
        'center': tuple(map(int, cv2.minEnclosingCircle(contour)[0])),
        'length': cv2.arcLength(contour, True),
        'width': cv2.minAreaRect(contour)[1][0],
        'diagonal': np.sqrt(cv2.arcLength(contour, True) ** 2 + cv2.minAreaRect(contour)[1][0] ** 2),
        'energy': np.sum(region_pixels ** 2) / len(region_pixels),
        'entropy': stats.entropy(region_pixels.flatten()),
        'mean': np.mean(region_pixels),
        'median': np.median(region_pixels)
    }
    
    return features

def analyze_hyperspectral_image(image, hsv_lower=(35, 50, 50), hsv_upper=(85, 255, 255)):
    """
    Analyze hyperspectral image and extract features
    """
    # Detect dark green regions
    contours, mask = detect_dark_green_regions(image, hsv_lower, hsv_upper)
    
    # Calculate features for each region
    features_list = []
    for i, contour in enumerate(contours):
        features = calculate_region_features(image, contour)
        features['No'] = i + 1
        features_list.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Reorder columns
    columns = ['No', 'center', 'length', 'width', 'diagonal', 
              'energy', 'entropy', 'mean', 'median']
    df = df[columns]
    
    return df, mask

def export_to_excel(df, filename='analysis_results.xlsx'):
    """
    Export analysis results to Excel file
    """
    df.to_excel(filename, index=False)
    return filename

def visualize_regions(image, contours, mask):
    """
    Visualize detected regions
    """
    # Create visualization image
    vis_image = image.copy()
    
    # Draw contours
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
    
    # Create mask visualization
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return vis_image, mask_vis 