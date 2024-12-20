import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import functions
import modules
import optimizers
import utils

IMAGE_SIZE = 28
HIDDEN_SIZE = 512
NUM_CLASSES = 10

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

CHECKPOINT_DIRECTORY = "checkpoints/cv_checkpoints"


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_layer_1 = modules.SvegaFastConv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.convolution_layer_2 = modules.SvegaFastConv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=2)
        self.pooling_layer = modules.SvegaSlowMaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu_layer = modules.SvegaLeakyReLU(0.02)
        self.fc_layer_1 = modules.SvegaLinear(4 * 8 * 8, 512)
        self.fc_layer_2 = modules.SvegaLinear(512, 10)
        
    def forward(self, X):
        x = self.convolution_layer_1(X)
        x = self.leaky_relu_layer(x)
        x = self.pooling_layer(x)

        x = self.convolution_layer_2(x)
        x = self.leaky_relu_layer(x)
        x = self.pooling_layer(x)

        x = x.view(-1, 4 * 8 * 8)

        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)

        return x


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            modules.SvegaFlatten(),
            modules.SvegaLinear(IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE),
            modules.SvegaBatchNorm(HIDDEN_SIZE),
            modules.SvegaLeakyReLU(alpha=0.01),
            modules.SvegaDropout(probability=0.2),
            modules.SvegaLinear(HIDDEN_SIZE, HIDDEN_SIZE),
            modules.SvegaBatchNorm(HIDDEN_SIZE),
            modules.SvegaLeakyReLU(alpha=0.01),
            modules.SvegaDropout(probability=0.2),
            modules.SvegaLinear(HIDDEN_SIZE, NUM_CLASSES)
        )

    def forward(self, x):
        return self.stack(x)
    

def main():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = ConvolutionalNeuralNetwork()
    optimizer = optimizers.SvegaAdam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch} --------------------")

        utils.train_model(model, train_dataloader, functions.SvegaCrossEntropyLoss, optimizer)
        utils.test_model(model, test_dataloader, functions.SvegaCrossEntropyLoss)

        torch.save(model.state_dict(), f"{CHECKPOINT_DIRECTORY}/checkpoint_{epoch}.pth")

        print()


if __name__ == "__main__":
    main()
