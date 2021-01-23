import time
from copy import deepcopy

import numpy as np
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.utils import show_img, plot_losses


class ASLRecognizerModel(nn.Module):
    def __init__(self, n_classes: int, frames_per_video: int,
                 lstm_num_layers: int = 100,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0,
                 lstm_hidden_size: int = 300,
                 device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else "cuda" if torch.cuda.is_available() else "cpu"

        super(ASLRecognizerModel, self).__init__()

        # gets the feature extractor from a pretrained CNN
        resnet_rgb = models.resnet34(pretrained=True)
        self.img_embeddings_size = list(resnet_rgb.children())[-1].weight.shape[-1]
        self.features_extractor_rgb = nn.Sequential(
            *list(resnet_rgb.children())[:-1],
            nn.Flatten()
        )

        # feature extractor for optical flow
        resnet_lk = models.resnet34(pretrained=False)
        resnet_lk.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features_extractor_lk = nn.Sequential(
            *list(resnet_lk.children())[:-1],
            nn.Flatten()
        )

        # lstm
        assert isinstance(n_classes, int) and n_classes >= 2
        self.n_classes = n_classes
        assert isinstance(frames_per_video, int) and frames_per_video >= 1
        assert isinstance(lstm_num_layers, int) and lstm_num_layers >= 1
        assert isinstance(lstm_bidirectional, bool)
        assert isinstance(lstm_hidden_size, int) and lstm_hidden_size >= 1
        assert not lstm_dropout or (isinstance(lstm_dropout, float) and 0 < lstm_dropout < 1)
        self.lstm = nn.LSTM(input_size=self.img_embeddings_size,
                            hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, bidirectional=lstm_bidirectional,
                            dropout=lstm_dropout if lstm_dropout else 0, batch_first=True)

        self.classification = nn.Linear(in_features=lstm_hidden_size,
                                        out_features=self.n_classes)
        self.to(self.device)

    def forward(self, X):
        in_dim = len(X.shape)
        if in_dim == 4:
            X = X.unsqueeze(0).to(self.device)

        # feature extraction from RGB image
        feature_vectors_rgb = torch.zeros(size=(X.shape[0], X.shape[1], self.img_embeddings_size)).to(self.device)
        for i, X_i in enumerate(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])(X).to(self.device)):
            feature_vectors_rgb[i] = self.features_extractor_rgb(X_i)

        # # feature extraction from optical flows
        # X_lk = self.get_optical_flow(X)
        # feature_vectors_lk = torch.zeros(size=(X_lk.shape[0], X.shape[1], self.img_embeddings_size)).to(self.device)
        # for i, X_i in enumerate(X_lk):
        #     feature_vectors_lk[i, 1:] = self.features_extractor_lk(X_i)
        #
        # # concatenates the two embeddings
        # feature_vectors = torch.cat([feature_vectors_rgb, feature_vectors_lk], dim=-1).to(self.device)
        feature_vectors = feature_vectors_rgb.to(self.device)
        # final prediction
        predictions = self.lstm(feature_vectors)[0][:, -1, :]
        predictions = self.classification(predictions)

        # softmax is automatically applied by the CrossEntropy loss during training
        if not self.training:
            predictions = F.softmax(predictions, dim=-1)

        if in_dim == 4:
            predictions = predictions.squeeze().to(self.device)

        return predictions


class ASLRecognizerModelFigo(nn.Module):
    def __init__(self, n_classes: int, frames_per_video: int,
                 lstm_hidden_size: int = 300,
                 device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else "cuda" if torch.cuda.is_available() else "cpu"

        super(ASLRecognizerModelFigo, self).__init__()

        assert isinstance(n_classes, int) and n_classes >= 2
        self.n_classes = n_classes

        # gets the feature extractor from a pretrained CNN
        resnet = models.video.r2plus1d_18(pretrained=False)
        self.img_embeddings_size = list(resnet.children())[-1].weight.shape[-1]
        self.layers = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
            nn.Linear(in_features=self.img_embeddings_size,
                      out_features=self.n_classes)
        )

        self.to(self.device)

    def forward(self, X: torch.FloatTensor):
        in_dim = len(X.shape)
        # reshapes the input
        X = X.permute(0, 2, 1, 3, 4)
        if in_dim == 4:
            X = X.unsqueeze(0).to(self.device)

        predictions = self.layers(X)

        # softmax is automatically applied by the CrossEntropy loss during training
        if not self.training:
            predictions = F.softmax(predictions, dim=-1)

        if in_dim == 4:
            predictions = predictions.squeeze().to(self.device)

        return predictions


def get_optical_flow(self, X):
    # retrieves the device where X is stored
    original_device = "cuda" if X.is_cuda else "cpu"

    def lucas_kanade(frame1, frame2):
        frame1, frame2 = (cv2.cvtColor(frame1.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY) * 255).astype(
            np.uint8), \
                         (cv2.cvtColor(frame2.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2GRAY) * 255).astype(
                             np.uint8)
        optical_flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow[np.isnan(optical_flow)] = 0
        optical_flow = np.clip(optical_flow, 0, 1)
        optical_flow = torch.as_tensor(optical_flow).to(original_device).permute(2, 0, 1)

        # mask = np.zeros((*frame1.shape, 3))
        # mask[..., 1] = 255
        # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # mask[..., 0] = angle * 180 / np.pi / 2
        # mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # # Converts HSV to RGB (BGR) color representation
        # rgb = cv2.cvtColor(mask.astype("float32"), cv2.COLOR_HSV2RGB)
        # rgb[np.isnan(rgb)] = 0
        # optical_flow = torch.as_tensor(rgb).to(self.device).permute(2, 0, 1) / 255
        return optical_flow

    X_lk = torch.zeros(size=(X.shape[0], X.shape[1] - 1, 2, *X.shape[3:])).to(original_device)
    for i_video, video in enumerate(X):
        for i_frame in range(X_lk.shape[1]):
            X_lk[i_video][i_frame] = lucas_kanade(X[i_video][i_frame], X[i_video][i_frame + 1])
    return X_lk


def train_model(model: nn.Module,
                train_dataloader: DataLoader, val_dataloader: DataLoader,
                lr: float = 1e-3, epochs=5,
                data_augmentation=True,
                filepath: str = None, verbose: bool = True):
    # checks about model's parameters
    assert isinstance(model, nn.Module)
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)
    if filepath:
        assert isinstance(filepath, str)
    # checks on other parameters
    assert isinstance(verbose, bool)
    assert isinstance(lr, float) and lr > 0
    assert isinstance(epochs, int) and epochs >= 1

    since = time.time()
    best_epoch_f1, best_model_weights = 0, \
                                        deepcopy(model.state_dict())

    # loss_function, optimizer = nn.CrossEntropyLoss(), \
    #                            torch.optim.Adam([
    #                                {"params": model.features_extractor_rgb.parameters(), "lr": 1e-5},
    #                                {"params": model.features_extractor_lk.parameters()},
    #                                {"params": model.lstm.parameters()},
    #                                {"params": model.classification.parameters()}
    #                            ], lr=lr)
    loss_function, optimizer = nn.CrossEntropyLoss(), \
                               torch.optim.Adam(model.parameters(), lr=lr)

    stats = pd.DataFrame()
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            data = train_dataloader if phase == "train" else val_dataloader
            if phase == 'train':
                if verbose:
                    print()
                model.train()
            else:
                model.eval()

            epoch_losses, epoch_f1 = np.zeros(shape=len(data)), \
                                     np.zeros(shape=len(data))
            for i_batch, batch in tqdm(enumerate(data), total=len(data),
                                       desc=f"{phase.capitalize()} epoch {epoch + 1}/{epochs}"):
                # gets input data
                X, y = batch[0].to(model.device), batch[1].to(model.device)

                # data augmentation
                if data_augmentation:
                    X_new = torch.zeros(size=(*X.shape[:3],
                                              224, 224)).to(model.device)
                    horizontal_flip_p, vertical_flip_p = 1 if np.random.random() < 0.75 else 0, \
                                                         1 if np.random.random() < 0.01 else 0
                    rotation_degrees = np.random.randint(-15, 16)
                    for i_sample, sample in enumerate(X):
                        transformations_train = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                            transforms.RandomVerticalFlip(p=vertical_flip_p),
                            transforms.Resize(size=256),
                            transforms.RandomRotation(degrees=(rotation_degrees, rotation_degrees)),
                            transforms.RandomCrop(size=224),
                            transforms.ToTensor()
                        ])
                        transformations_val = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor()
                        ])
                        transformations = transformations_train if phase == "train" else transformations_val
                        for i_frame, frame in enumerate(sample):
                            X_new[i_sample][i_frame] = transformations(frame)
                    X = X_new
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X)
                    loss = loss_function(y_pred, y)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_losses[i_batch], epoch_f1[i_batch] = loss, \
                                                               f1_score(y.cpu(),
                                                                        torch.argmax(y_pred, dim=-1).cpu(),
                                                                        average="macro")
            # print some stats
            stats = stats.append(pd.DataFrame(
                index=[len(stats)],
                data={
                    "epoch": epoch + 1,
                    "phase": phase,
                    "avg loss": np.mean(epoch_losses),
                    "avg F1": np.mean(epoch_f1)
                }))
            if verbose and phase == "val":
                print(f"\n", stats.to_string(index=False))
                if epoch > 1:
                    plot_losses(train_losses=stats.loc[(stats["phase"] == "train")]["avg loss"],
                                val_losses=stats.loc[(stats["phase"] == "val")]["avg loss"])

            # save the best model
            avg_epoch_f1 = np.mean(epoch_f1)
            if phase == 'val' and avg_epoch_f1 > best_epoch_f1:
                best_epoch_f1, best_model_weights = avg_epoch_f1, \
                                                    deepcopy(model.state_dict())
                original_device = model.device
                model = model.cpu()
                torch.save(model.state_dict(), filepath)
                model = model.to(original_device)
                if verbose:
                    print(f"Found best model with F1 {avg_epoch_f1} and saved to {filepath}")

    if verbose:
        time_elapsed = time.time() - since
        print(f'Training completed in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s '
              f'with a best F1 of {best_epoch_f1}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, best_epoch_f1
