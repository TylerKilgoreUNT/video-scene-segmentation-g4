from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
import scenedetect.scene_manager
import cv2
import numpy as np

# Override PySceneDetect's progress bar string to remove the redundant `Detected: X` text
class SilentProgress(str):
    def __mod__(self, other):
        return "Detection Progress"

scenedetect.scene_manager.PROGRESS_BAR_DESCRIPTION = SilentProgress()

def get_average_brightness(video_path, sample_rate=30):
    """
    Calculates the average brightness of a video by sampling frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 127 # Fallback to mid-gray if video cannot be opened
        
    total_brightness = 0
    frame_count = 0
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % sample_rate == 0:
            # Convert frame to HSV and get the Value channel (brightness)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            total_brightness += brightness
            frame_count += 1
            
        count += 1
        
    cap.release()
    
    if frame_count == 0:
        return 127
        
    return total_brightness / frame_count

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
    def __init__(self, threshold=27.0, adaptive_threshold=True):
        self.base_threshold = threshold
        self.adaptive_threshold = adaptive_threshold

    def detect(self, video_path):
        """
        Runs the PySceneDetect analysis against the provided video file.
        Returns a list of tuples containing (start_time, end_time) FrameTimecode objects for each scene.
        """
        self.used_threshold = self.base_threshold
        if self.adaptive_threshold:
            avg_brightness = get_average_brightness(video_path)
            self.used_threshold = calculate_adaptive_threshold(avg_brightness, self.base_threshold)
            print(f"Adaptive thresholding: Avg Brightness = {avg_brightness:.2f}, Adjusted Threshold = {self.used_threshold}")

        # Open video using the v0.6+ PySceneDetect API
        video = open_video(video_path)
        
        # Instantiate SceneManager (Controller)
        scene_manager = SceneManager()
        
        # Configure ContentDetector (adjust sensitivity)
        scene_manager.add_detector(ContentDetector(threshold=self.used_threshold))
        
        # Process the mathematical analysis loop and show progress in terminal
        scene_manager.detect_scenes(video, show_progress=True)
        
        # Get the detected scenes, defaulting to 1 continuous scene if no cuts are found
        scene_list = scene_manager.get_scene_list(start_in_scene=True)
        
        return scene_list
