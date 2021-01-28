import argparse
from os.path import join, isfile
from pprint import pprint

import numpy as np
import pandas as pd

from modules.data import load_dataset

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import ASLRecognizerModel
from modules.utils import read_json, save_json, show_img, build_vocab

if __name__ == '__main__':
    # seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # paths
    assets_path = join(".", "assets")
    samples_path = join(assets_path, "samples")
    train_data_path, val_data_path = join(samples_path, "train"), \
                                     join(samples_path, "val")

    model_path = join(assets_path, "model")
    parameters_path, vocab_path = join(model_path, "parameters.json"), \
                                  join(model_path, "vocab.json")
    training_checkpoint_path = join(model_path, "ASLRecognizer_weights.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parses the arguments from the console
    parser = argparse.ArgumentParser(description='Parameters for Alphabet Sign Recognition training')
    parser.add_argument("-osl", "--only_static_letters", dest='only_static_letters', action='store_true',
                        help='Whether to delete videos labeled with \'j\' and \'z\'')
    parser.add_argument("-da", "--data_augmentation", dest='data_augmentation', action='store_false',
                        help='Whether to not use data augmentation techniques')
    parser.add_argument("-e", "--epochs", dest='epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument("-bs", "--batch_size", dest='batch_size', type=int, default=4,
                        help='Number of videos per batch')
    parser.add_argument("-npr", "--not_pretrained_resnet", dest='not_pretrained_resnet', action='store_true',
                        help='Whether to use pretrained ResNet')
    parser.add_argument("-of", "--use_optical_flow", dest='use_optical_flow', action='store_true',
                        help='Whether to use optical flow in predictions (using Lucas Kanade algorithm)')
    args = parser.parse_args()
    assert isinstance(args.epochs, int) and args.epochs >= 1
    assert isinstance(args.batch_size, int) and args.batch_size >= 1
    assert isinstance(args.not_pretrained_resnet, bool)
    # retrieves the parameters from a file
    # or creates it
    overwrite_parameters = True
    if not isfile(parameters_path) or overwrite_parameters:
        parameters = {
            "training": {
                "only_static_letters": args.only_static_letters,
                "data_augmentation": args.data_augmentation,
                "frames_per_video": 16,
                "epochs": args.epochs,
                "lr_features_extractor": 1e-5,
                "lr_classification": 1e-3,
                "batch_size": args.batch_size,
                "num_workers": 2,
                "pretrained_resnet": not args.not_pretrained_resnet,
                "use_optical_flow": args.use_optical_flow
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
                                    frames_per_video=parameters["training"]["frames_per_video"],
                                    only_static_letters=args.only_static_letters), \
                       load_dataset(samples_path=val_data_path,
                                    frames_per_video=parameters["training"]["frames_per_video"],
                                    only_static_letters=args.only_static_letters)
    print(f"\t|train| = {len(ds_train)}\t|val| = {len(ds_val)}")

    vocab = build_vocab(targets=set(ds_train.get_labels()))
    if not isfile(vocab_path) or overwrite_parameters:
        save_json(content=vocab, filepath=vocab_path)
    ds_train.vocab, ds_val.vocab = vocab, vocab
    print(f"Vocabulary: {', '.join(sorted(vocab.keys())).strip()}")

    train_dataloader, val_dataloader = DataLoader(ds_train, shuffle=True, pin_memory=True,
                                                  batch_size=parameters["training"]["batch_size"],
                                                  num_workers=parameters["training"]["num_workers"]), \
                                       DataLoader(ds_val, shuffle=False, pin_memory=True,
                                                  batch_size=parameters["training"]["batch_size"],
                                                  num_workers=parameters["training"]["num_workers"])

    # defines the model
    model = ASLRecognizerModel(n_classes=len(vocab),
                               lr_features_extractor=parameters["training"]["lr_features_extractor"],
                               lr_classification=parameters["training"]["lr_classification"],
                               pretrained_resnet=parameters["training"]["pretrained_resnet"],
                               use_optical_flow=parameters["training"]["use_optical_flow"],
                               training_checkpoint_path=training_checkpoint_path,
                               device=device)

    # starts the training
    trainer = pl.Trainer(gpus=1 if model.device_str == "cuda" else 0,
                         precision=16, accumulate_grad_batches=1,
                         profiler=True,
                         max_epochs=parameters["training"]["epochs"])
    trainer.tune(model, train_dataloader, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)
