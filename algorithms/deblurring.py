import cv2
import numpy as np
from scipy import signal

def estimate_motion_blur_kernel(image, angle=0, length=10):
    """
    Estimate motion blur kernel
    """
    # Create motion blur kernel
    kernel = np.zeros((length, length))
    center = length // 2
    
    # Set kernel values
    for i in range(length):
        kernel[center, i] = 1.0
    
    # Rotate kernel
    kernel = cv2.warpAffine(
        kernel,
        cv2.getRotationMatrix2D((center, center), angle, 1.0),
        (length, length)
    )
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    return kernel

def wiener_deblur(image, kernel, K=0.01):
    """
    Apply Wiener deconvolution
    """
    # Convert to float32
    image = image.astype(np.float32)
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Pad kernel to match image size
    kernel_padded = np.zeros((h, w))
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    
    # Compute FFT
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_padded)
    
    # Wiener deconvolution
    kernel_fft_conj = np.conj(kernel_fft)
    kernel_fft_abs_sq = np.abs(kernel_fft) ** 2
    
    # Apply Wiener filter
    deblurred_fft = (kernel_fft_conj * image_fft) / (kernel_fft_abs_sq + K)
    
    # Inverse FFT
    deblurred = np.real(np.fft.ifft2(deblurred_fft))
    
    # Normalize and convert to uint8
    deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
    
    return deblurred

def richardson_lucy_deblur(image, kernel, iterations=30):
    """
    Apply Richardson-Lucy deconvolution
    """
    # Convert to float32
    image = image.astype(np.float32)
    
    # Initialize estimate
    estimate = image.copy()
    
    # Create flipped kernel
    kernel_flipped = np.flip(kernel)
    
    for _ in range(iterations):
        # Forward projection
        forward = signal.convolve2d(estimate, kernel, mode='same')
        
        # Compute ratio
        ratio = image / (forward + 1e-10)
        
        # Backward projection
        backward = signal.convolve2d(ratio, kernel_flipped, mode='same')
        
        # Update estimate
        estimate = estimate * backward
    
    # Normalize and convert to uint8
    estimate = np.clip(estimate, 0, 255).astype(np.uint8)
    
    return estimate

def deblur_image(image, method='wiener', angle=0, length=10, K=0.01, iterations=30):
    """
    Main deblurring function
    """
    # Estimate motion blur kernel
    kernel = estimate_motion_blur_kernel(image, angle, length)
    
    # Apply selected deblurring method
    if method == 'wiener':
        return wiener_deblur(image, kernel, K)
    elif method == 'richardson_lucy':
        return richardson_lucy_deblur(image, kernel, iterations)
    else:
        raise ValueError("Invalid deblurring method") 