import json
from typing import Union

import cv2
import numpy as np
import seaborn as sns

import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def save_json(content: dict, filepath: str):
    with open(filepath, 'w') as fp:
        json.dump(content, fp, indent=4)


def read_json(filepath: str):
    with open(filepath, 'r') as fp:
        return json.load(fp)


def show_img(img: torch.Tensor):
    assert isinstance(img, torch.Tensor)
    assert 3 <= len(img.shape) <= 4
    # if it's a single image
    if len(img.shape) == 3:
        plt.imshow(img.cpu().permute(1, 2, 0))
    # if it's a sequence
    if len(img.shape) == 4:
        fig, axs = plt.subplots(int(np.ceil(np.sqrt(img.shape[0]))),
                                int(np.floor(np.sqrt(img.shape[0]))))
        for i_frame, frame in enumerate(img):
            axs.flat[i_frame].imshow(frame.cpu().permute(1, 2, 0))
        for ax in axs.flat:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def build_vocab(targets: Union[list, tuple, set]):
    assert isinstance(targets, list) or isinstance(targets, tuple) or isinstance(targets, set)
    assert len(targets) > 0
    vocab = {}
    for target in targets:
        if target not in vocab:
            vocab[target] = len(vocab)
    return vocab


def resize_videos(X, size):
    assert isinstance(size, int) and size >= 1
    in_dim, original_device = len(X.shape), "cuda" if X.is_cuda else "cpu"
    if in_dim == 4:
        X = X.unsqueeze(0).to(original_device)
    resizing_transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=112),
        transforms.ToTensor()
    ])
    X_new = torch.zeros(size=(*X.shape[:3],
                              size, size)).to(original_device)
    for i_video, video in enumerate(X):
        for i_frame, frame in enumerate(video):
            X_new[i_video][i_frame] = resizing_transformations(frame)
    if in_dim == 4:
        X_new = X_new.squeeze(0).to(original_device)
    return X_new.to(original_device)

def get_optical_flow(X):
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

def plot_losses(train_losses, val_losses, title: str = None):
    # plots the loss chart
    sns.lineplot(y=train_losses, x=range(1, len(train_losses) + 1))
    sns.lineplot(y=val_losses, x=range(1, len(val_losses) + 1))
    plt.title(f'Loss {"" if not title else f"for model {title}"}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.show()
