from os.path import exists, join

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score

import pytorch_lightning as pl

from modules.utils import show_img, resize_videos, validation_data_augmentation, training_data_augmentation


class ASLRecognizerModel(pl.LightningModule):
    def __init__(self, n_classes: int,
                 pretrained_resnet: bool = True,
                 lr_features_extractor: float = 1e-5, lr_classification: float = 1e-4,
                 training_checkpoint_path: str = None, device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device_str = device if device in {"cpu", "cuda"} else "cuda" if torch.cuda.is_available() else "cpu"

        super(ASLRecognizerModel, self).__init__()

        assert isinstance(n_classes, int) and n_classes >= 2
        self.n_classes = n_classes

        # gets the feature extractor from a pretrained CNN
        resnet = models.video.r2plus1d_18(pretrained=pretrained_resnet)
        self.img_embeddings_size = list(resnet.children())[-1].weight.shape[-1]
        self.features_extractor = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
        )
        self.classification = nn.Sequential(
            nn.Linear(in_features=self.img_embeddings_size,
                      out_features=self.n_classes)
        )

        # weights path
        assert not training_checkpoint_path or isinstance(training_checkpoint_path, str)
        self.training_checkpoint_path = training_checkpoint_path

        # learning rates
        self.lr_features_extractor, self.lr_classification_layer = lr_features_extractor, lr_classification

        # stats
        self._last_train_epoch_stats, self._last_val_epoch_stats = [], []
        self.training_stats = pd.DataFrame(columns=["train_loss", "val_loss", "train_f1", "val_f1"])
        self.to(self.device_str)

    def forward(self, X: torch.FloatTensor):
        in_dim = len(X.shape)
        if in_dim == 4:
            X = X.unsqueeze(0).to(self.device_str)
        # eventually resizes the videos
        if X.shape[-2] != 112 or X.shape[-1] != 112:
            X = resize_videos(X, 112)
        # normalizes the input for ResNet
        X = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                 std=[0.22803, 0.22145, 0.216989])(X).permute(0, 2, 1, 3, 4)
        # predicts the labels
        features = self.features_extractor(X)
        predictions = self.classification(features)

        # softmax is automatically applied by the CrossEntropy loss during training
        if not self.training:
            predictions = F.softmax(predictions, dim=-1)

        if in_dim == 4:
            predictions = predictions.squeeze().to(self.device_str)

        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.features_extractor.parameters(), "lr": self.lr_features_extractor},
            {"params": self.classification.parameters(), "lr": self.lr_classification_layer},
        ], lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # sets the model in training mode
        self.train()
        # retrieves X and y from the batch
        X, y = batch[0], batch[1]
        # applies data augmentation
        X = training_data_augmentation(X)
        # predicts the labels
        y_pred = self.forward(X)
        # updates the stats
        loss = F.cross_entropy(y_pred, y)
        self._last_train_epoch_stats += [
            {
                "loss": loss,
                "y": y,
                "y_pred": torch.argmax(y_pred, dim=-1)
            }
        ]
        return loss

    def validation_step(self, batch, batch_idx):
        # sets the model in eval mode
        self.eval()
        # retrieves X and y from the batch
        X, y = batch[0], batch[1]
        # applies data augmentation
        X = validation_data_augmentation(X)
        # predicts the labels
        y_pred = self.forward(X)
        # updates the stats
        loss = F.cross_entropy(y_pred, y)
        self._last_val_epoch_stats += [
            {
                "loss": loss,
                "y": y,
                "y_pred": torch.argmax(y_pred, dim=-1)
            }
        ]

    def on_epoch_end(self):
        if self._last_train_epoch_stats == [] or self._last_val_epoch_stats == []:
            return
        train_loss, val_loss = np.mean([batch_output["loss"].item() for batch_output in self._last_train_epoch_stats]), \
                               np.mean([batch_output["loss"].item() for batch_output in self._last_val_epoch_stats])
        train_y, train_y_pred = [value.item() for batch_output in self._last_train_epoch_stats
                                 for value in batch_output["y"]], \
                                [value.item() for batch_output in self._last_train_epoch_stats
                                 for value in batch_output["y_pred"]]
        val_y, val_y_pred = [value.item() for batch_output in self._last_val_epoch_stats
                             for value in batch_output["y"]], \
                            [value.item() for batch_output in self._last_val_epoch_stats
                             for value in batch_output["y_pred"]]
        train_f1, val_f1 = f1_score(train_y, train_y_pred, average="macro"), \
                           f1_score(val_y, val_y_pred, average="macro")
        # eventually saves the model
        if not self.training_stats.empty:
            best_val_f1 = np.max(self.training_stats["val_f1"])
            if val_f1 > best_val_f1:
                if self.training_checkpoint_path:
                    self.save_weights(self.training_checkpoint_path)
                    print(f"Found best model with F1 = {np.round(val_f1, 4)} (+{np.round(val_f1 - best_val_f1, 4)}) "
                          f"and saved to {self.training_checkpoint_path}")
                else:
                    print(f"Found best model with F1 = {np.round(val_f1, 4)} (+{np.round(val_f1 - best_val_f1, 4)}) "
                          f"but a filepath where to save the checkpoint is not given")
        # updates stats
        self.training_stats = self.training_stats.append({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_f1": train_f1,
            "val_f1": val_f1
        }, ignore_index=True)
        print("\n", self.training_stats)
        self._last_train_epoch_stats, self._last_val_epoch_stats = [], []

    '''
    W E I G H T S
    M A N A G E M E N T
    '''

    def load_weights(self, weights_path):
        assert isinstance(weights_path, str) and exists(weights_path)
        self.cpu()
        self.load_state_dict(torch.load(join(weights_path)))
        self.to(self.device_str)

    def save_weights(self, weights_path):
        assert isinstance(weights_path, str)
        self.cpu()
        torch.save(self.state_dict(), weights_path)
        self.to(self.device_str)
