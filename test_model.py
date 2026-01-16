import os
import cv2
import argparse
import sys
import json

from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file

def hex_to_bgr(hex_color):
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    # Convert hex to RGB, then to BGR for OpenCV
    rgb = tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
    return (rgb[2], rgb[1], rgb[0])

# Load labels and colors from label.json
try:
    with open('label.json', 'r') as f:
        labels_data = json.load(f)
    VOC_CLASSES = [item['name'] for item in labels_data]
    COLORS = [hex_to_bgr(item['color']) for item in labels_data]
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Could not load labels from label.json: {e}. Using default labels.")
    VOC_CLASSES = [str(i) for i in range(15)] # Default of 15 classes if json fails
    # Generate some default colors
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)] * 3

FONT = cv2.FONT_HERSHEY_SIMPLEX

def test_model(config_file, image_path, output_path):
    # 1. Load the configure file
    # This will also set cfg.RESUME_CHECKPOINT based on the config file
    cfg_from_file(config_file)

    # NOTE: The model checkpoint specified in the config file
    # (e.g., './experiments/models/mobilenet_v2_ssd_lite_voc.pth')
    # appears to be an empty file (0 bytes).
    # Please ensure you have a valid, pre-trained model checkpoint
    # at the path specified by RESUME_CHECKPOINT in your .yml config file
    # before running this script.

    # 2. Load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 4. Detect
    _labels, _scores, _coords = object_detector.predict(image)

    # 5. Draw bounding box on the image
    for label, score, coords in zip(_labels, _scores, _coords):
        # Ensure coords are integers for cv2 functions
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        color = COLORS[label % len(COLORS)] # Cycle through predefined colors
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{VOC_CLASSES[label]}: {score:.3f}'
        cv2.putText(image, text, (x1, y1 - 10), FONT, 0.5, color, 2)
    
    # 6. Write result
    cv2.imwrite(output_path, image)
    print(f"Detection result saved to {output_path}")

def camera_test(config_file):
    # 1. Load the configure file
    cfg_from_file(config_file)

    # 2. Load detector
    object_detector = ObjectDetector()

    # 3. Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # 4. Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # 5. Detect objects
        _labels, _scores, _coords = object_detector.predict(frame)

        # 6. Draw bounding boxes
        for label, score, coords in zip(_labels, _scores, _coords):
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            color = COLORS[label % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f'{VOC_CLASSES[label]}: {score:.3f}'
            cv2.putText(frame, text, (x1, y1 - 10), FONT, 0.5, color, 2)

        # 7. Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # 8. Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a SSDR network with an image or camera feed')
    parser.add_argument('--cfg', dest='config_file',
                        help='Path to the configuration file (e.g., config/ssd_lite_mobilenetv2_train_voc.yml)',
                        default='config/ssd_lite_mobilenetv2_train_voc.yml', type=str)
    parser.add_argument('--input_image', dest='input_image',
                        help='Path to the input image (e.g., test/test.jpg)',
                        default='test/test.jpg', type=str)
    parser.add_argument('--output_image', dest='output_image',
                        help='Path to save the output image (e.g., test/test_result.jpg)',
                        default='test/test_result.jpg', type=str)
    parser.add_argument('--use_camera', action='store_true',
                        help='Use the camera for live object detection')
    
    args = parser.parse_args()

    if args.use_camera:
        camera_test(args.config_file)
    else:
        # Make sure the output directory exists
        output_dir = os.path.dirname(args.output_image)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        test_model(args.config_file, args.input_image, args.output_image)