
import cv2
import numpy as np

class EnhancementPipeline:
    """
    Preprocessing pipeline designed for Acute Ischemic Stroke Detection.
    Based on ICIP 2015 paper: "Dark Image Enhancement Based on Pairwise Target Contrast"
    
    Updated: Tamed version with Alpha Blending to preserve Original density.
    
     techniques:
    1. Optimal Windowing (Stroke Window)
    2. CLAHE (Local Contrast Enhancement)
    3. Unsharp Masking (Detail Boosting)
    4. Alpha Blending (Original vs Enhanced)
    """
    def __init__(self, 
                 stroke_center=40,   # Back to standard-ish but still optimized
                 stroke_width=40,
                 clahe_clip_limit=1.0, # Reduced from 2.0
                 clahe_tile_grid_size=(8,8),
                 boost_strength=0.5,   # Reduced from 1.5
                 alpha=0.3):           # Blend factor: 30% Enhanced, 70% Original
        
        self.stroke_center = stroke_center
        self.stroke_width = stroke_width
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.boost_strength = boost_strength
        self.alpha = alpha
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, 
            tileGridSize=self.clahe_tile_grid_size
        )

    def apply_window(self, image, center, width):
        """Apply CT windowing"""
        lower = center - (width / 2)
        upper = center + (width / 2)
        
        img = np.clip(image, lower, upper)
        img = (img - lower) / (upper - lower)
        return img

    def apply_clahe(self, image):
        """Apply CLAHE to enhance local contrast"""
        # Convert to 8-bit for CLAHE
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE
        enhanced = self.clahe.apply(img_uint8)
        
        # Convert back to float [0,1]
        return enhanced.astype(np.float32) / 255.0

    def apply_unsharp_mask(self, image):
        """Boost details using Unsharp Masking"""
        # Gaussian Blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Unsharp Mask formulation: Original + Strength * (Original - Blurred)
        enhanced = cv2.addWeighted(image, 1.0 + self.boost_strength, gaussian, -self.boost_strength, 0)
        
        # Clip to valid range
        return np.clip(enhanced, 0, 1)

    def process_blended(self, image_hu):
        """
        Run full pipeline but BLEND with original window to preserve density.
        """
        # 1. Base Image (Standard Stroke Window)
        # Using the same window as enhancement for consistent blending basis
        base = self.apply_window(image_hu, self.stroke_center, self.stroke_width)
        
        # 2. Enhancement Chain
        enhanced = self.apply_clahe(base)
        enhanced = self.apply_unsharp_mask(enhanced)
        
        # 3. Alpha Blending
        # Final = (1 - alpha) * Base + alpha * Enhanced
        final_image = cv2.addWeighted(base, 1.0 - self.alpha, enhanced, self.alpha, 0)
        
        return final_image
    
    def process(self, image_hu):
        """Alias for process_blended for backward compatibility"""
        return self.process_blended(image_hu)
    
    def get_3_channels(self, image_hu):
        """
        Return 3-channel image as requested:
        Ch1: Standard Window (Anchor)
        Ch2: Context Window (Standard Brain 40/80)
        Ch3: Enhanced (Detail Boost)
        """
        # Channel 1: Optimal Stroke Window (Base)
        ch1 = self.apply_window(image_hu, self.stroke_center, self.stroke_width)
        
        # Channel 2: Context Window (Standard 40/80)
        ch2 = self.apply_window(image_hu, 40, 80)
        
        # Channel 3: Aggressive Enhancement (Higher contrast for features)
        # We can use slightly higher params here since it's a separate channel
        temp_clahe = self.apply_clahe(ch1)
        ch3 = self.apply_unsharp_mask(temp_clahe)
        
        return np.stack([ch1, ch2, ch3], axis=0) # (3, H, W)

# Singleton instance for easy import
default_pipeline = EnhancementPipeline()
