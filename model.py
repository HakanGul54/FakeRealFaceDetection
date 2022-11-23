import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_channels, input_size, output_size):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_channels, input_size, 5, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_size, 128, 5, 2, 1, bias=False),
            nn.MaxPool2d(5, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, 2, 1, bias=False),
            nn.MaxPool2d(5, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(57600, output_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)