import re
from os import listdir
from os.path import splitext, split, exists, isdir, join
from typing import Optional, Iterable

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, videos_paths: Optional[Iterable] = [],
                 frames_per_video: Optional[int] = None):
        self.videos_paths = videos_paths
        self.frames_per_video = frames_per_video

        self.vocab = None

    def __len__(self):
        return len(self.videos_paths)

    def __getitem__(self, i):
        video_path = self.videos_paths[i]
        video, label = torchvision.io.read_video(video_path, pts_unit="sec")[0], \
                       splitext(split(video_path)[-1])[0]
        if self.frames_per_video:
            video = video[np.linspace(start=0, stop=len(video), num=self.frames_per_video,
                                      endpoint=False, dtype=int)]
        video = video.permute(0, 3, 1, 2) / 255
        if self.vocab:
            label = torch.as_tensor(self.vocab[label])
        return video, label

    def get_labels(self):
        labels = []
        for video_path in self.videos_paths:
            labels += splitext(split(video_path)[-1])[0]
        return labels


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

    dataset = VideoDataset(videos_paths=videos_paths, frames_per_video=frames_per_video)
    return dataset
