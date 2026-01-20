import os
import cv2
import argparse
import sys
import json
import time

import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, shape_inference
import torch
import numpy as np
# from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file
from lib.utils.config_parse import cfg
from lib.modeling.model_builder import create_model
from lib.utils.data_augment import preproc
from lib.layers import Detect

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class CalibrationDataReader:
    def __init__(self, cfg, preprocessor):
        self.preprocessor = preprocessor
        self.h = 360
        self.w = 640
        self.input_name = 'input'
        self.call_count = 0
        self.num_samples = 10

    def get_next(self):
        if self.call_count >= self.num_samples:
            return None

        # Generate random image
        img = (np.random.rand(self.h, self.w,3) * 255).astype(np.uint8)

        # Preprocess the image
        x_preprocessed = self.preprocessor(img)[0].unsqueeze(0).cpu().numpy()

        self.call_count += 1
        return {self.input_name: x_preprocessed}

class ONNXObjectDetector:
    def __init__(self, viz_arch=False):
        self.cfg = cfg

        # --- PyTorch Model Loading for Conversion ---
        print('===> Building PyTorch model for ONNX conversion')
        self.model, self.priorbox = create_model(cfg.MODEL)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model.to(self.device)
            self.priors.to(self.device)

        print('=> loading checkpoint {:s}'.format(cfg.RESUME_CHECKPOINT))
        checkpoint = torch.load(cfg.RESUME_CHECKPOINT, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # --- ONNX Conversion ---
        self.onnx_path = os.path.splitext(cfg.RESUME_CHECKPOINT)[0] + ".onnx"
        if not os.path.exists(self.onnx_path):
            print(f"ONNX model not found at {self.onnx_path}, converting from .pth...")
            self.export_to_onnx()
        else:
            print(f"Found existing ONNX model at {self.onnx_path}")

        # --- Post-processor setup needs to be before quantization for calibration ---
        self.preprocessor = preproc(cfg.MODEL.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, -2)
        
        # --- ONNX Quantization ---
        self.quantize_to_int8()

        # --- ONNX Runtime Session Loading ---
        print("===> Loading ONNX model for inference")
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path,providers=['CPUExecutionProvider']	)

        # --- Detector setup ---
        self.detector = Detect(cfg.POST_PROCESS, self.priors)


    def export_to_onnx(self):
        img_size = self.cfg.MODEL.IMAGE_SIZE
        dummy_input = torch.randn(1, 3, img_size[1], img_size[0], device=self.device)
        
        output_names = ['loc', 'conf']
        
        torch.onnx.export(self.model,
                          dummy_input,
                          self.onnx_path,
                          export_params=True,
                          opset_version=13,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=output_names,
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'loc' : {0 : 'batch_size'},
                                        'conf' : {0 : 'batch_size'}})
        print(f"ONNX model exported to {self.onnx_path}")


    def quantize_to_int8(self):
        print("===> Quantizing ONNX model to INT8")
        onnx_int8_path = os.path.splitext(self.onnx_path)[0] + "_int8.onnx"
        onnx_preprocessed_path = os.path.splitext(self.onnx_path)[0] + "_preprocessed.onnx"

        if not os.path.exists(onnx_int8_path):
            print("Preprocessing model for quantization...")
            # Step 1: Pre-process (Required to avoid ConvInteger errors)
            shape_inference.quant_pre_process(self.onnx_path, onnx_preprocessed_path)
            
            print("Performing static quantization...")
            calibration_data_reader = self.create_calibration_data_reader()
            
            # Step 2: Quantize the pre-processed model
            quantize_static(
                model_input=onnx_preprocessed_path,
                model_output=onnx_int8_path,
                calibration_data_reader=calibration_data_reader,
                activation_type=QuantType.QUInt8, # Explicitly set activation type
                weight_type=QuantType.QInt8
            )
            print(f"Quantized model saved to {onnx_int8_path}")
        
        self.onnx_path = onnx_int8_path

    def create_calibration_data_reader(self):
        return CalibrationDataReader(self.cfg, self.preprocessor)


    def predict(self, img, threshold=0.6):
        assert img.shape[2] == 3
        height, width, _ = img.shape
        scale = torch.Tensor([width, height, width, height]).to(self.device)

        x_preprocessed = self.preprocessor(img)[0].unsqueeze(0)
        x = x_preprocessed.to(self.device)

        # ONNX forward pass
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        out = [torch.from_numpy(o).to(self.device) for o in ort_outs]

        with torch.no_grad():
            detections = self.detector.forward(out)

        labels, scores, coords = [list() for _ in range(3)]
        batch = 0
        for classes in range(1, detections.size(1)): # Skip background class 0
            num = 0
            while num < detections.size(2) and detections[batch, classes, num, 0] >= threshold:
                score = detections[batch, classes, num, 0]
                current_coords = detections[batch, classes, num, 1:] * scale

                scores.append(score)
                labels.append(classes - 1)
                coords.append(current_coords)
                num += 1
        return labels, scores, coords


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

    object_detector = ONNXObjectDetector()

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
        text = '{cls}: {score:.3f}'.format(cls=VOC_CLASSES[label], score=score)
        cv2.putText(image, text, (x1, y1 - 10), FONT, 0.5, color, 2)
    
    # 6. Write result
    cv2.imwrite(output_path, image)
    print(f"Detection result saved to {output_path}")

def camera_test(config_file):
    # 1. Load the configure file
    cfg_from_file(config_file)

    # 2. Load detector
    object_detector = ONNXObjectDetector()

    # 3. Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # To calculate FPS
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # 4. Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Start timer after getting the frame
        new_frame_time = time.time()

        # 5. Detect objects
        _labels, _scores, _coords = object_detector.predict(frame)

        # 6. Draw bounding boxes and log detections
        if len(_labels) > 0:
            print("--- Frame Detections ---")
        for label, score, coords in zip(_labels, _scores, _coords):
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            color = COLORS[label % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            class_name = VOC_CLASSES[label]
            text = '{cls}: {score:.3f}'.format(cls=class_name, score=score)
            cv2.putText(frame, text, (x1, y1 - 10), FONT, 0.5, color, 2)
            
            # Log detected object to console
            print('Detected: {cls} | Score: {score:.3f}'.format(cls=class_name, score=score))

        # Calculate and display FPS
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), FONT, 1, (0, 255, 0), 2)
        
        prev_frame_time = new_frame_time


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