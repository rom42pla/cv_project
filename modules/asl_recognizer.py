from os.path import join

import time
from datetime import datetime
from threading import Thread

from . import utils

import torch
import torch.nn.functional as F
import torchvision

import cv2 as cv

from .camera import Camera
import numpy as np
from modules.features_extraction import FeatureExtractor
from modules.utils import read_json
from model import ASLRecognizerModel


class ASLRecognizer:
    def __init__(self, camera: Camera, assets_path: str,
                 predictions_delta: float = 2):
        # setups the faster device available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # binds the camera to the recognizer
        assert isinstance(camera, Camera)
        self.camera = camera

        # loads the model
        assert isinstance(assets_path, str)
        self.model_path = join(assets_path, "model")
        self.parameters = read_json(filepath=join(self.model_path, "parameters.json"))
        self.vocab = utils.read_json(join(self.model_path, "vocab.json"))
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}
        self.model = ASLRecognizerModel(n_classes=len(self.vocab),
                                        frames_per_video=self.parameters["training"]["frames_per_video"],
                                        lstm_num_layers=self.parameters["training"]["lstm_num_layers"],
                                        lstm_bidirectional=self.parameters["training"]["lstm_bidirectional"],
                                        lstm_dropout=self.parameters["training"]["lstm_dropout"])
        self.model.load_state_dict(torch.load(join(self.model_path, "ASLRecognizer_weights.pth")))
        self.model.eval()

        # setups the thread
        self.is_running, self.thread = False, None

        # setups the predictions
        assert isinstance(predictions_delta, int) or isinstance(predictions_delta, float)
        assert predictions_delta > 1
        self.predictions_delta = predictions_delta

    def recognize(self):
        while self.is_running and self.camera.is_running:
            # waits for some seconds
            time.sleep(self.predictions_delta)
            current_time = datetime.now()
            predicted_letter = self.predict_letter(self.camera.get_saved_frames(
                frames_per_video=self.parameters["training"]["frames_per_video"]))
            print(f"{current_time.hour}:{current_time.minute}:{current_time.second}\t{predicted_letter}")

    def start(self):
        # starts the thread
        self.is_running = True
        self.thread = Thread(target=self.recognize)
        self.thread.start()

    def stop(self):
        # stops the thread
        self.is_running = False

    def predict_letter(self, X):
        X = X.to(self.model.device)
        with torch.no_grad():
            prediction = self.model(X)
        prediction = torch.argmax(prediction, dim=-1).item()

        return self.vocab_reversed[prediction]

    # def get_images(self):
    #     # gets the images from the camera object
    #     images = self.camera.q_frames
    #
    #     keep = np.ceil(np.linspace(0, 30 - 1, 8, endpoint=True)).astype(np.int32)
    #
    #     images = images[keep]
    #
    #     sample = np.empty((images.shape[0], 128, 128, 3), dtype=np.uint8)
    #
    #     for i in range(sample.shape[0]):
    #         sample[i, :, :, 0] = cv.resize(images[i, :, :, 2], (128, 128), interpolation=cv.INTER_AREA)
    #         sample[i, :, :, 1] = cv.resize(images[i, :, :, 1], (128, 128), interpolation=cv.INTER_AREA)
    #         sample[i, :, :, 2] = cv.resize(images[i, :, :, 0], (128, 128), interpolation=cv.INTER_AREA)
    #
    #     sample = [ sample.astype(np.float32) / 255.0, *FeatureExtractor.process(sample) ]
    #     sample = [ reshape_swap(x, 1, 3) for x in sample ]
    #     return sample

    # def predict_letter(self, X_rgb, X_canny, X_lk):
    #
    #     X_rgb = torch.FloatTensor(X_rgb).to(self.device).unsqueeze(0)
    #     X_canny = torch.FloatTensor(X_canny).to(self.device).unsqueeze(0)
    #     X_lk = torch.FloatTensor(X_lk).to(self.device).unsqueeze(0)
    #
    #     with torch.no_grad():
    #         prediction = self.model(X_rgb, X_canny, X_lk)
    #
    #     prediction = sorted(enumerate(prediction.flatten().tolist()), key=lambda x: x[1], reverse=True)
    #
    #     print("=" * 32)
    #     for p in prediction[:5]:
    #         print(self.vocab[p[0]], p[1])
    #
    #     return self.vocab[prediction[0][0]]
