import json
from typing import Union

import numpy as np
import seaborn as sns

import torch

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
