import argparse

import cv2

from preprocessing import Preprocessor


def build_parser():
	"""Define CLI arguments for simple video playback."""
	parser = argparse.ArgumentParser(description="Simple video player")
	parser.add_argument("video_path", help="Path to input video file")
	return parser


def play_video(video_path):
	"""Play a video file. Press q to quit."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("Error: could not open video for playback.")
		return

	fps = cap.get(cv2.CAP_PROP_FPS) or 30
	delay = max(1, int(1000 / fps))

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
	"""Parse CLI args and launch playback."""
	args = build_parser().parse_args()
	play_video(args.video_path)

	# TESTING ONLY:
	# Preprocessor preview to verify frame normalization + grayscale pipeline.
	preprocessor = Preprocessor()
	play_preprocessed_video(args.video_path, preprocessor)


if __name__ == "__main__":
	main()
