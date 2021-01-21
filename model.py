import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from modules.utils import show_img

class ASLRecognizerModel(nn.Module):
    def __init__(self, n_classes: int, frames_per_video: int,
                 lstm_num_layers: int = 100,
                 lstm_bidirectional: bool = False,
                 lstm_dropout : float = 0,
                 lstm_hidden_size: int = 300,
                 device: str = "auto"):
        # checks that the device is correctly given
        assert device in {"cpu", "cuda", "auto"}
        self.device = device if device in {"cpu", "cuda"} else "cuda" if torch.cuda.is_available() else "cpu"

        super(ASLRecognizerModel, self).__init__()

        # gets the feature extractor from a pretrained CNN,
        # with frozen parameters
        resnet = models.resnet34(pretrained=True)
        self.features_extractor = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten()
        )
        self.img_embeddings_size = list(resnet.children())[-1].weight.shape[-1]

        # self.cnn_canny = nn.Sequential(
        #     nn.Conv2d(1, 2, kernel_size=(3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3)),
        #     nn.Conv2d(2, 4, kernel_size=(3, 3)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3)),
        #     nn.Flatten()
        # )
        # self.cnn_lk = nn.Sequential(
        #     nn.Conv2d(2, 4, kernel_size=(7, 7)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3)),
        #     nn.Conv2d(4, 8, kernel_size=(7, 7)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(3, 3)),
        #     nn.Flatten()
        # )

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
            X = X.unsqueeze(0)

        # feature extraction from RGB image
        X = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(X)
        feature_vectors = torch.zeros(size=(X.shape[0], X.shape[1], self.img_embeddings_size)).to(self.device)
        for i, X_i in enumerate(X):
            feature_vectors[i] = self.features_extractor(X_i)

        # lstm
        predictions = self.lstm(feature_vectors)[0][:, -1, :]

        predictions = self.classification(predictions)

        # softmax is automatically applied by the CrossEntropy loss during training
        if not self.training:
            predictions = F.softmax(predictions, dim=-1)

        if in_dim == 4:
            predictions = predictions.squeeze()

        return predictions
