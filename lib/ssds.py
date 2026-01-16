from __future__ import print_function
import numpy as np

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.utils.config_parse import cfg


class ObjectDetector:
    def __init__(self, viz_arch=False):
        self.cfg = cfg

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        # Print the model architecture and parameters
        if viz_arch is True:
            print('Model architectures:\n{}\n'.format(self.model))

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.half = False
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            self.model.to(self.device)
            self.priors.to(self.device)
            cudnn.benchmark = True
            # self.model = torch.nn.DataParallel(self.model).module
            # Utilize half precision
            self.half = cfg.MODEL.HALF_PRECISION
            if self.half:
                self.model = self.model.half()
                self.priors = self.priors.half()

        # Build preprocessor and detector
        self.preprocessor = preproc(cfg.MODEL.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS, -2)
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Load weight:
        if cfg.RESUME_CHECKPOINT == '':
            AssertionError('RESUME_CHECKPOINT can not be empty')
        print('=> loading checkpoint {:s}'.format(cfg.RESUME_CHECKPOINT))
        # checkpoint = torch.load(cfg.RESUME_CHECKPOINT)
        checkpoint = torch.load(cfg.RESUME_CHECKPOINT, map_location='cuda' if self.use_gpu else 'cpu')
        self.model.load_state_dict(checkpoint)
        # test only
        self.model.eval()

    def predict(self, img, threshold=0.6):
        assert img.shape[2] == 3
        height, width, _ = img.shape
        scale = torch.Tensor([width, height, width, height]).to(self.device)

        x = self.preprocessor(img)[0].unsqueeze(0).to(self.device)

        # forward
        if self.half:
            x = x.half()
        
        with torch.no_grad():
            out = self.model(x)  # forward pass
            detections = self.detector.forward(out)

        # output
        labels, scores, coords = [list() for _ in range(3)]
        
        batch = 0
        for classes in range(detections.size(1)):
            num = 0
            # Check if the number of detections is within bounds
            while num < detections.size(2) and detections[batch, classes, num, 0] >= threshold:
                scores.append(detections[batch, classes, num, 0])
                labels.append(classes - 1)
                coords.append(detections[batch, classes, num, 1:] * scale)
                num += 1
        return labels, scores, coords
