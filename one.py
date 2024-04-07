import torch
import cv2
import argparse

# Import yolov5 module
from yolov5 import detect

def detect_on_video(video_path, weights_path, output_path):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = detect(frame, model, device='cpu')

        # Draw bounding boxes on the frame
        output_frame = results.render()[0]

        # Write the frame with bounding boxes to the output video
        out.write(output_frame)

        # Display progress
        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}")

    # Release everything when finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object detection on video using YOLOv5')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv5 weights file')
    parser.add_argument('--output', type=str, required=True, help='Path to output video file')
    args = parser.parse_args()

    detect_on_video(args.video, args.weights, args.output)
