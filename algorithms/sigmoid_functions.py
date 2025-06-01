import numpy as np
import cv2

def standard_sigmoid(image, alpha=1.0):
    """
    Apply standard sigmoid function for contrast enhancement
    f(x) = 1 / (1 + e^(-alpha * x))
    """
    # Normalize image to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Apply sigmoid function
    enhanced = 1 / (1 + np.exp(-alpha * (normalized - 0.5)))
    
    # Scale back to [0, 255]
    return (enhanced * 255).astype(np.uint8)

def shifted_sigmoid(image, alpha=1.0, shift=0.0):
    """
    Apply horizontally shifted sigmoid function
    f(x) = 1 / (1 + e^(-alpha * (x - shift)))
    """
    normalized = image.astype(np.float32) / 255.0
    enhanced = 1 / (1 + np.exp(-alpha * (normalized - shift)))
    return (enhanced * 255).astype(np.uint8)

def sloped_sigmoid(image, alpha=1.0, beta=1.0):
    """
    Apply sloped sigmoid function
    f(x) = beta * (1 / (1 + e^(-alpha * x)))
    """
    normalized = image.astype(np.float32) / 255.0
    enhanced = beta * (1 / (1 + np.exp(-alpha * (normalized - 0.5))))
    # Clip values to [0, 1] range
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def custom_sigmoid(image, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Custom sigmoid function with additional parameters
    f(x) = beta * (1 / (1 + e^(-alpha * (x^gamma - 0.5))))
    """
    normalized = image.astype(np.float32) / 255.0
    enhanced = beta * (1 / (1 + np.exp(-alpha * (np.power(normalized, gamma) - 0.5))))
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8) 