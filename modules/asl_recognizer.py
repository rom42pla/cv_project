from os.path import join
import time
from datetime import datetime
from threading import Thread

import pandas as pd

import torch

from .camera import Camera
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
        self.camera.asl_recognizer = self

        # paths
        assert isinstance(assets_path, str)
        model_path = join(assets_path, "model")
        parameters_path, vocab_path, model_weights_path = join(model_path, "parameters.json"), \
                                                          join(model_path, "vocab.json"), \
                                                          join(model_path, "ASLRecognizer_weights.pth")
        # parameters
        self.parameters, self.vocab = read_json(filepath=parameters_path), \
                                      read_json(filepath=vocab_path)
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}
        # defines the model
        self.model = ASLRecognizerModel(n_classes=len(self.vocab),
                                        pretrained_resnet=False)
        self.model.load_weights(weights_path=model_weights_path)
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

    def predict_letter(self, X, threshold=0.12):
        X, prediction_time = X.to(self.model.device_str), datetime.now()
        with torch.no_grad():
            prediction = self.model(X)
        # orders the predicted letters
        prediction_indexes = torch.argsort(prediction, descending=True)
        letter = self.vocab_reversed[prediction_indexes[0].item()]
        print(f"Prediction of {prediction_time.hour}:{prediction_time.minute}:{prediction_time.second}\n",
              pd.DataFrame(index=[pd.to_datetime(pd.Timestamp.now(), format='%H:%M')],
                           data={
                               "highest value": [torch.max(prediction).item()],
                               "mean": [torch.mean(prediction).item()],
                               "variance": [torch.var(prediction).item()],
                               "std": [torch.std(prediction).item()],
                               "threshold std": [threshold],
                               "is a letter?": [True if torch.std(prediction).item() >= threshold else False]
                           }).to_string(index=True), "\n",
              pd.DataFrame(data={
                  "letter": [self.vocab_reversed[p.item()] for p in prediction_indexes][:5],
                  "confidence": [prediction[p].item() for p in prediction_indexes][:5]
              }).to_string(index=False))
        # checks if the predicted letter is above the threshold
        if torch.std(prediction) >= threshold:
            self.predicted_letters += [letter]
        return letter
