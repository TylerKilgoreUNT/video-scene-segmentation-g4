import numpy as np


class CoreProcessor:
    """
    Handles feature extraction and window-to-window comparison.
    Uses Preprocessor outputs to compute scene change signals.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def compute_window_signature(self, frames):
        """
        Convert a window of frames into a compact representation.
        Uses mean grayscale image from Preprocessor.
        """
        if frames is None or len(frames) == 0:
            raise ValueError("Empty frame window received")

        return self.preprocessor.window_signature(frames)

    def compute_difference(self, sigA, sigB):
        """
        Compute difference between two window signatures.
        Using L1 distance (robust + simple).
        """
        return np.sum(np.abs(sigA - sigB))

    def process_windows(self, window_iterator):
        """
        Main pipeline:
        - Takes sliding windows from VideoReader
        - Computes differences between consecutive windows

        Returns:
        - diff_values: numpy array of difference scores
        - indices: numpy array of frame indices (center of window)
        """

        prev_signature = None

        diff_values = []
        indices = []

        for window in window_iterator:
            frames = window.get("frames", None)

            # Skip invalid windows
            if frames is None or len(frames) == 0:
                continue

            # Compute signature for this window
            curr_signature = self.compute_window_signature(frames)

            if prev_signature is not None:
                diff = self.compute_difference(prev_signature, curr_signature)

                diff_values.append(diff)

                # Use center frame as representative index
                start = window["start_index"]
                end = window["end_index"]
                center_index = (start + end) // 2

                indices.append(center_index)

            prev_signature = curr_signature

        return np.array(diff_values), np.array(indices)