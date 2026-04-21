from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
import scenedetect.scene_manager

# Override PySceneDetect's progress bar string to remove the redundant `Detected: X` text
class SilentProgress(str):
    def __mod__(self, other):
        return "Detection Progress"

scenedetect.scene_manager.PROGRESS_BAR_DESCRIPTION = SilentProgress()

class SceneDetectorModule:
    """
    Wraps the PySceneDetect SceneManager API to detect scene boundaries using HSV Histogram Analysis.
    """
    def __init__(self, threshold=40.0):
        self.threshold = threshold

    def detect(self, video_path):
        """
        Runs the PySceneDetect analysis against the provided video file.
        Returns a list of tuples containing (start_time, end_time) FrameTimecode objects for each scene.
        """
        # Open video using the v0.6+ PySceneDetect API
        video = open_video(video_path)
        
        # Instantiate SceneManager (Controller)
        scene_manager = SceneManager()
        
        # Configure ContentDetector (adjust sensitivity)
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        
        # Process the mathematical analysis loop and show progress in terminal
        scene_manager.detect_scenes(video, show_progress=True)
        
        # Get the detected scenes, defaulting to 1 continuous scene if no cuts are found
        scene_list = scene_manager.get_scene_list(start_in_scene=True)
        
        return scene_list
