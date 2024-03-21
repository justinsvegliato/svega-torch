import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import datasets
import functions
import modules
import optimizers
import utils

CORPUS_FILENAME = "data/shakespeare-corpus.txt"
DATASET_SPLIT = 0.90

SEQUENCE_LENGTH = 16
EMBEDDING_SIZE = 64
NUM_TRANSFORMER_BLOCKS = 8
NUM_HEADS = 8

LEARNING_RATE = 1e-2
BATCH_SIZE = 32
EPOCHS = 5
NUM_BATCHES = 2000

CHECKPOINT_DIRECTORY = "checkpoints/nlp_checkpoints"


def main():
    dataset = datasets.TextDataset(CORPUS_FILENAME, sequence_length=SEQUENCE_LENGTH)
    vocabulary_size = dataset.get_vocabulary_size()

    train_size = int(DATASET_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = modules.SvegaGPT(
        vocabulary_size, 
        SEQUENCE_LENGTH, 
        EMBEDDING_SIZE, 
        NUM_TRANSFORMER_BLOCKS, 
        NUM_HEADS
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch} --------------------")

        utils.train_model(model, train_dataloader, functions.SvegaCrossEntropyLoss, optimizer, NUM_BATCHES)
        utils.test_model(model, test_dataloader, functions.SvegaCrossEntropyLoss)

        torch.save(model.state_dict(), f"{CHECKPOINT_DIRECTORY}/checkpoint_{epoch}.pth")

        context = torch.zeros((1, 1), dtype=torch.long)
        tokens = model.generate(context, max_new_tokens=500)[0].tolist()
        print(dataset.decode(tokens))

        print()


if __name__ == "__main__":
    main()
