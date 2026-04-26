from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
import scenedetect.scene_manager
import cv2
import numpy as np
from collections import deque
from core import CoreProcessor
from preprocessing import Preprocessor
from videoReader import VideoReaderModule

# Override PySceneDetect's progress bar string to remove the redundant `Detected: X` text
class SilentProgress(str):
    def __mod__(self, other):
        return "Detection Progress"

scenedetect.scene_manager.PROGRESS_BAR_DESCRIPTION = SilentProgress()

class AdaptiveContentDetector(ContentDetector):
    """
    Dynamically adjusts the detection threshold based on a rolling window 
    of the video's average brightness.
    """
    def __init__(self, base_threshold=27.0, window_size=30, **kwargs):
        super().__init__(threshold=base_threshold, **kwargs)
        self.base_threshold = base_threshold
        self.brightness_history = deque(maxlen=window_size)
        
    def process_frame(self, frame_num, frame_img):
        # Calculate brightness of the current frame
        hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        self.brightness_history.append(brightness)
        
        # Calculate moving average brightness
        avg_brightness = sum(self.brightness_history) / len(self.brightness_history)
        
        # Dynamically adjust threshold for this frame
        self.threshold = calculate_adaptive_threshold(avg_brightness, self.base_threshold)
        
        # Print live output every 15 frames (~2 times a second for 30fps)
        if frame_num % 15 == 0:
            print(f"[Live] Frame {frame_num:<5} | Local Brightness: {avg_brightness:<6.2f} | Threshold: {self.threshold:<5.2f}")
            
        # Call the parent class's frame processing logic with the newly adjusted threshold
        return super().process_frame(frame_num, frame_img)

def calculate_adaptive_threshold(average_brightness, base_threshold=27.0):
    """
    Adjusts the PySceneDetect ContentDetector threshold based on average video brightness.
    Brightness is assumed to be on a scale of 0 (black) to 255 (white).
    
    Uses a continuous sliding scale:
    - Neutral brightness (127.5) results in the base_threshold.
    - Darker videos (< 127.5) proportionally lower the threshold (up to -3.0).
    - Brighter videos (> 127.5) proportionally raise the threshold (up to +3.0).
    """
    # Max adjustment amount for pure black (0) or pure white (255)
    max_adjustment = 3.0
    
    # Calculate difference from neutral (127.5)
    brightness_diff = average_brightness - 127.5
    
    # Calculate proportional adjustment
    adjustment = (brightness_diff / 127.5) * max_adjustment
    
    # Apply adjustment and round to 2 decimal places for clean output
    adjusted_threshold = round(base_threshold + adjustment, 2)
    
    return adjusted_threshold

class SceneDetectorModule:
    """
    Wraps the PySceneDetect SceneManager API to detect scene boundaries using HSV Histogram Analysis.
    """
    def __init__(self, threshold=27.0, adaptive_threshold=True, enable_validation=True, validation_threshold=1500.0):
        self.base_threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.enable_validation = enable_validation
        self.validation_threshold = validation_threshold

    def detect(self, video_path):
        """
        Runs the PySceneDetect analysis against the provided video file.
        Returns a list of tuples containing (start_time, end_time) FrameTimecode objects for each scene.
        """
        self.used_threshold = self.base_threshold
        if self.adaptive_threshold:
            self.used_threshold = f"Adaptive (Base: {self.base_threshold})"
            print(f"Using Adaptive thresholding (Base: {self.base_threshold}, Window: 30 frames)")
            detector_instance = AdaptiveContentDetector(base_threshold=self.base_threshold)
        else:
            detector_instance = ContentDetector(threshold=self.base_threshold)

        # Open video using the v0.6+ PySceneDetect API
        video = open_video(video_path)
        
        # Instantiate SceneManager (Controller)
        scene_manager = SceneManager()
        
        # Configure ContentDetector (adjust sensitivity)
        scene_manager.add_detector(detector_instance)
        
        # Process the mathematical analysis loop
        # We disable show_progress if adaptive is true because we output our own live threshold stats
        scene_manager.detect_scenes(video, show_progress=not self.adaptive_threshold)
        
        # Add a newline after the live output finishes
        if self.adaptive_threshold:
            print()
        
        # Get the detected scenes, defaulting to 1 continuous scene if no cuts are found
        scene_list = scene_manager.get_scene_list(start_in_scene=True)
        
        # Secondary Validation
        if self.enable_validation:
            scene_list = self.validate_scene_cuts(scene_list, video_path, threshold=self.validation_threshold)
            
        return scene_list

    def validate_scene_cuts(self, scene_list, video_path, window_size=5, threshold=1500.0):
        print("\n--- Validating Cuts via CoreProcessor ---")
        if len(scene_list) <= 1:
            print("No candidate cuts detected to validate.")
            return scene_list
            
        video_reader = VideoReaderModule(video_path)
        preprocessor = Preprocessor()
        core_processor = CoreProcessor(preprocessor)
        
        valid_scenes = []
        current_scene_start = scene_list[0][0]
        
        for i in range(len(scene_list) - 1):
            scene_end = scene_list[i][1]
            next_scene_start = scene_list[i+1][0]
            cut_frame = scene_end.get_frames()
            
            # Extract window before and after cut
            pre_cut_frames = video_reader.get_frames_at(max(0, cut_frame - window_size), window_size)
            post_cut_frames = video_reader.get_frames_at(cut_frame, window_size)
            
            if not pre_cut_frames or not post_cut_frames:
                valid_scenes.append((current_scene_start, scene_end))
                current_scene_start = next_scene_start
                continue
                
            sig_pre = core_processor.compute_window_signature(pre_cut_frames)
            sig_post = core_processor.compute_window_signature(post_cut_frames)
            diff = core_processor.compute_difference(sig_pre, sig_post)
            
            status = "VALID" if diff >= threshold else "FALSE POSITIVE (Merged)"
            print(f"Cut at frame {cut_frame:<5} | L1 Diff: {diff:<8.2f} | Status: {status}")
            
            if diff >= threshold:
                valid_scenes.append((current_scene_start, scene_end))
                current_scene_start = next_scene_start
                
        # Append the very last scene segment
        valid_scenes.append((current_scene_start, scene_list[-1][1]))
        
        return valid_scenes
