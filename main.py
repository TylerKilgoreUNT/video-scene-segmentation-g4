import argparse
import sys
import cv2

from preprocessing import Preprocessor
from sceneDetection import SceneDetectorModule
from output import format_and_print_results
import scenedetect


def build_parser():
    """Define CLI arguments for video playback and scene detection."""
    parser = argparse.ArgumentParser(description="Video Scene Segmentation (using PySceneDetect + OpenCV preview)")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--threshold", type=float, default=27.0, 
                        help="Base threshold for ContentDetector (default: 27.0)")
    parser.add_argument("--disable-adaptive", action="store_true", 
                        help="Disable adaptive thresholding based on video brightness")
    parser.add_argument("--validation-threshold", type=float, default=1500.0, 
                        help="L1 threshold for secondary core.py validation (default: 1500.0)")
    parser.add_argument("--disable-validation", action="store_true", 
                        help="Disable secondary validation layer")
    return parser


def play_video(video_path):
    """Play a video file. Press q to quit."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video for playback.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    print("\n--- Playing raw video (Press 'q' to quit) ---")
    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow("Video Player", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def play_preprocessed_video(video_path, preprocessor):
    """Display preprocessed frames for debugging/testing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video for preprocessing preview.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))

    print("\n--- Playing preprocessed video (Press 'q' to quit) ---")
    while True:
        success, frame = cap.read()
        if not success:
            break

        processed = preprocessor.preprocess(frame)
        display_frame = (processed * 255).astype("uint8")

        cv2.imshow("Preprocessed Video (Testing)", display_frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Parse CLI args and launch detection + playback."""
    parser = build_parser()
    
    # If no args are passed, fallback to showing help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    # Enable native PySceneDetect logging so you see the console output
    scenedetect.init_logger()

    # 1. SCENE DETECTION (Using PySceneDetect API)
    print(f"Initializing SceneDetector with threshold={args.threshold}...")
    detector = SceneDetectorModule(
        threshold=args.threshold, 
        adaptive_threshold=not args.disable_adaptive,
        enable_validation=not args.disable_validation,
        validation_threshold=args.validation_threshold
    )
    
    print(f"Processing video: {args.video_path}")
    scene_list = detector.detect(args.video_path)
    
    # Validate and print output
    format_and_print_results(scene_list, threshold=getattr(detector, 'used_threshold', args.threshold))


    # 2. RAW PLAYBACK (Original behavior)
    play_video(args.video_path)

    # 3. PREPROCESSED PLAYBACK (Original testing behavior)
    preprocessor = Preprocessor()
    play_preprocessed_video(args.video_path, preprocessor)


if __name__ == "__main__":
    main()
