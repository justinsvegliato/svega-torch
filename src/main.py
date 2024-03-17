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

LEARNING_RATE = 1e-2
BATCH_SIZE = 64
EPOCHS = 10

CHECKPOINT_DIRECTORY = "checkpoints"


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            modules.SvegaFlatten(),
            modules.SvegaLinear(IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE),
            modules.SvegaLayerNorm(HIDDEN_SIZE),
            modules.SvegaLeakyReLU(alpha=0.01),
            modules.SvegaLinear(HIDDEN_SIZE, HIDDEN_SIZE),
            modules.SvegaLayerNorm(HIDDEN_SIZE),
            modules.SvegaLeakyReLU(alpha=0.01),
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

    model = FeedForwardNeuralNetwork()
    optimizer = optimizers.SvegaSGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch} --------------------")

        utils.train_model(model, train_dataloader, functions.SvegaCrossEntropyLoss, optimizer)
        utils.test_model(model, test_dataloader, functions.SvegaCrossEntropyLoss)

        torch.save(model.state_dict(), f"{CHECKPOINT_DIRECTORY}/checkpoint_{epoch}.pth")

        print()


if __name__ == "__main__":
    main()
