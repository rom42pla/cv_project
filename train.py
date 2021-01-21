import time
from itertools import product
from os.path import join, isfile
from pprint import pprint
from collections import Counter

import numpy as np
import pandas as pd

np.random.seed(0)

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

torch.manual_seed(0)

from model import ASLRecognizerModel
from modules.utils import read_json, save_json, show_img, load_dataset, build_vocab, vocab_to_tensor, train_model

if __name__ == '__main__':
    assets_path = join(".", "assets")
    samples_path = join(assets_path, "samples")
    train_data_path, val_data_path = join(samples_path, "train"), \
                                     join(samples_path, "val")

    model_path = join(assets_path, "model")
    parameters_path, vocab_path = join(model_path, "parameters.json"), \
                                  join(model_path, "vocab.json")

    # retrieves the parameters from a file
    # or creates it
    overwrite_parameters = True
    if not isfile(parameters_path) or overwrite_parameters:
        parameters = {
            "transformations": {
                "resize_size": 256,
                "random_crop_size": 224,
                "random_horizontal_flip_probability": 0.5,
                "random_vertical_flip_probability": 0.01,
                "random_rotation_degrees": 15
            },
            "training": {
                "frames_per_video": 12,
                "epochs": 10,
                "learning_rate": 1e-4,
                "batch_size": 2,
                "lstm_num_layers": 1,
                "lstm_bidirectional": False,
                "lstm_hidden_size": 1024,
                "lstm_dropout": 0,
            }
        }
        save_json(content=parameters, filepath=parameters_path)
        print(f"Created a parameters file in {parameters_path}")
    else:
        parameters = read_json(filepath=parameters_path)
        print(f"Read parameters from {parameters_path}")
    pprint(parameters)

    print(f"Loading datasets...")
    ds_train, ds_val = load_dataset(samples_path=train_data_path,
                                    frames_per_video=parameters["training"]["frames_per_video"]), \
                       load_dataset(samples_path=val_data_path,
                                    frames_per_video=parameters["training"]["frames_per_video"])
    print(f"\t|train| = {len(ds_train)}\t|val| = {len(ds_val)}")

    vocab = build_vocab(targets=set(ds_train.get_labels()))
    if not isfile(vocab_path) or overwrite_parameters:
        save_json(content=vocab, filepath=vocab_path)
    ds_train.vocab, ds_val.vocab = vocab, vocab

    dl_train, dl_val = DataLoader(ds_train, batch_size=parameters["training"]["batch_size"], shuffle=True,
                                  pin_memory=True, num_workers=parameters["training"]["batch_size"]), \
                       DataLoader(ds_val, batch_size=parameters["training"]["batch_size"], shuffle=False,
                                  pin_memory=True, num_workers=parameters["training"]["batch_size"])

    model = ASLRecognizerModel(n_classes=len(vocab), frames_per_video=parameters["training"]["frames_per_video"],
                               lstm_num_layers=parameters["training"]["lstm_num_layers"],
                               lstm_bidirectional=parameters["training"]["lstm_bidirectional"],
                               lstm_hidden_size=parameters["training"]["lstm_hidden_size"],
                               lstm_dropout=parameters["training"]["lstm_dropout"])
    train_model(model=model, filepath=join(model_path, "ASLRecognizer_weights.pth"),
                epochs=parameters["training"]["epochs"], lr=parameters["training"]["learning_rate"],
                data_augmentation=True,
                train_dataloader=dl_train, val_dataloader=dl_val)
