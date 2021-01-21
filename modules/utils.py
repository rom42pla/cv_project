import time
from copy import deepcopy
import json
import random
import re
from os import listdir
from os.path import join, exists, isdir, splitext
from typing import Union
import threading

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

from modules.data import VideoDataset


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


def load_dataset(samples_path: str, frames_per_video: int = None):
    assert isinstance(samples_path, str)
    assert exists(samples_path) and isdir(samples_path)
    assert isinstance(frames_per_video, int) and frames_per_video >= 1
    videos_paths = []
    for take in listdir(samples_path):
        if isdir(join(samples_path, take)):
            for sample_name in listdir(join(samples_path, take)):
                # ignores non-videos
                if not re.fullmatch(pattern=r"([a-z]|[A-Z])\.mp4", string=sample_name):
                    continue
                videos_paths += [join(samples_path, take, sample_name)]

    # # gets each video
    # dataset, lock = [], threading.Lock()
    #
    # def get_video(dataset):
    #     while len(videos_paths) > 0:
    #         lock.acquire()
    #         video_path = videos_paths.pop()
    #         lock.release()
    #         from os.path import split
    #         video, label = torchvision.io.read_video(video_path, pts_unit="sec")[0].permute(0, 3, 1, 2) / 255, \
    #                        splitext(split(video_path)[-1])[0]
    #         if frames_per_video:
    #             video = video[np.linspace(start=0, stop=len(video) - 1, num=frames_per_video, dtype=int)]
    #         lock.acquire()
    #         dataset += [(video, label)]
    #         lock.release()
    #
    # # starts the threads for getting the videos fastly
    # threads = [threading.Thread(target=get_video, args=[dataset]) for _ in range(100)]
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()

    dataset = VideoDataset(videos_paths=videos_paths, frames_per_video=frames_per_video)
    return dataset


def reshape_swap(a, d1, d2):
    shape = list(a.shape)
    tmp = shape[d1]
    shape[d1] = shape[d2]
    shape[d2] = tmp
    return a.reshape(shape)


def build_vocab(targets: Union[list, tuple, set]):
    assert isinstance(targets, list) or isinstance(targets, tuple) or isinstance(targets, set)
    assert len(targets) > 0
    vocab = {}
    for target in targets:
        if target not in vocab:
            vocab[target] = len(vocab)
    return vocab


def vocab_to_tensor(targets: Union[list, tuple, set], vocab: dict):
    assert isinstance(targets, list) or isinstance(targets, tuple) or isinstance(targets, set)
    assert len(targets) > 0
    return torch.as_tensor([vocab[target] for target in targets], dtype=torch.long)


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

    loss_function, optimizer = nn.CrossEntropyLoss(), \
                               torch.optim.Adam([
                                   {"params": model.features_extractor.parameters(), "lr": 1e-5},
                                   {"params": model.lstm.parameters()},
                                   {"params": model.classification.parameters()}
                               ], lr=lr)

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
                            transforms.Resize(size=224),
                            transforms.ToTensor()
                        ])
                        transformations = transformations_train if phase == "train" else transformations_val
                        for i_frame, frame in enumerate(sample):
                            X_new[i_sample][i_frame] = transformations(frame)
                    X = X_new
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X=X)
                    loss = loss_function(y_pred, y)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_losses[i_batch], epoch_f1[i_batch] = loss, \
                                                               f1_score(y.cpu(),
                                                                        torch.argmax(y_pred, dim=-1).cpu(),
                                                                        average="macro")

            # deep copy the model
            avg_epoch_f1 = np.mean(epoch_f1)
            if phase == 'val' and avg_epoch_f1 > best_epoch_f1:
                best_epoch_f1, best_model_weights = avg_epoch_f1, \
                                                    deepcopy(model.state_dict())
                if verbose:
                    print(f"Found best model with F1 {avg_epoch_f1}")
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

    if verbose:
        time_elapsed = time.time() - since
        print(f'Training completed in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s '
              f'with a best F1 of {best_epoch_f1}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    # saves to a file
    if filepath:
        original_device = model.device
        model = model.cpu()
        torch.save(model.state_dict(), filepath)
        model = model.to(original_device)
        if verbose:
            print(f"Model's weights saved to {filepath}")
    return model, best_epoch_f1
