import cv2
import numpy as np

class Preprocessor:
    """Preprocess video frames into normalized representations."""

    def __init__(self, output_size=(224, 224), blur_kernel=(5, 5)):
        # Fixed-size output keeps distance comparisons consistent.
        self.output_size = output_size
        # Light blur suppresses camera/compression noise before comparison.
        self.blur_kernel = blur_kernel

    def preprocess(self, frame):
        """Convert a single BGR frame into normalized grayscale representation."""
        # Convert to grayscale (simpler + robust)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Small blur helps suppress sensor noise and compression artifacts.
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Resize to fixed size
        resized = cv2.resize(blurred, self.output_size)

        # Normalize pixel intensities to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def preprocess_window(self, frames):
        """Preprocess all frames in a window and return shape (T, H, W)."""
        if frames is None or len(frames) == 0:
            raise ValueError("frames must contain at least one frame")

        processed_frames = [self.preprocess(frame) for frame in frames]
        return np.stack(processed_frames, axis=0)

    def window_signature(self, frames):
        # Use the per-pixel temporal mean as a compact signature for a window.
        processed_window = self.preprocess_window(frames)
        return np.mean(processed_window, axis=0)
