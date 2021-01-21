from os.path import splitext, split
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