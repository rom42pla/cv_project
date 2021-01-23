from os.path import join

import time
from datetime import datetime
from threading import Thread

from . import utils

import torch

from .camera import Camera

from modules.utils import read_json
from model import ASLRecognizerModel, ASLRecognizerModelFigo


class ASLRecognizer:
    def __init__(self, camera: Camera, assets_path: str,
                 predictions_delta: float = 2):
        # setups the faster device available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # binds the camera to the recognizer
        assert isinstance(camera, Camera)
        self.camera = camera
        self.camera.asl_recognizer = self

        # loads the model
        assert isinstance(assets_path, str)
        self.model_path = join(assets_path, "model")
        self.parameters = read_json(filepath=join(self.model_path, "parameters.json"))
        self.vocab = utils.read_json(join(self.model_path, "vocab.json"))
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}
        self.model = ASLRecognizerModelFigo(n_classes=len(self.vocab),
                                            frames_per_video=self.parameters["training"]["frames_per_video"],
                                            weights_path=join(self.model_path, "ASLRecognizer_weights.pth"))
        self.model.eval()

        # setups the thread
        self.is_running, self.thread = False, None
        self.waiting_since = None
        self.predicted_letters = []

        # setups the predictions
        assert isinstance(predictions_delta, int) or isinstance(predictions_delta, float)
        assert predictions_delta > 1
        self.predictions_delta = predictions_delta

    def recognize(self):
        while self.is_running and self.camera.is_running:
            # waits for some seconds
            self.waiting_since = time.time()
            time.sleep(self.predictions_delta)
            current_time = datetime.now()
            prediction_thread = Thread(target=self.predict_letter,
                                       args=[self.camera.get_saved_frames(
                                           frames_per_video=self.parameters["training"]["frames_per_video"])])
            prediction_thread.start()
            # print(f"{current_time.hour}:{current_time.minute}:{current_time.second}\t{predicted_letter}")

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
        letter = self.vocab_reversed[prediction]
        self.predicted_letters += [letter]
        return letter
