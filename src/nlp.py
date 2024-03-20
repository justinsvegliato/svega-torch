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

PREVIEW_LENGTH = 10

EMBEDDING_SIZE = 32
SEQUENCE_LENGTH = 8
NUM_TRANSFORMER_BLOCKS = 2
NUM_HEADS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_STEPS = 5000
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10

CHECKPOINT_DIRECTORY = "checkpoints"


def main():
    dataset = datasets.TextDataset(CORPUS_FILENAME, sequence_length=SEQUENCE_LENGTH)
    vocabulary_size = dataset.get_vocabulary_size()

    train_size = int(DATASET_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = modules.SvegaGPT(
        vocabulary_size, 
        SEQUENCE_LENGTH, 
        EMBEDDING_SIZE, 
        NUM_TRANSFORMER_BLOCKS, 
        NUM_HEADS
    )

    # input_token = torch.zeros((1, 1), dtype=torch.long)
    # input_character = decode(input_token[0].tolist())
    # print(f"Input Token: {input_token}")
    # print(f"Input Character: {input_character}")
    # print("========================================")

    # output_tokens = model.generate(input_token, max_new_tokens=300)
    # output_characters = decode(output_tokens[0].tolist())
    # print(f"Output Tokens: {output_tokens}")
    # print(f"Output Characters: {output_characters}")
    # print("========================================")

    optimizer = optimizers.SvegaSGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch} --------------------")

        utils.train_model(model, train_dataloader, functions.SvegaCrossEntropyLoss, optimizer)
        utils.test_model(model, test_dataloader, functions.SvegaCrossEntropyLoss)

        torch.save(model.state_dict(), f"{CHECKPOINT_DIRECTORY}/checkpoint_{epoch}.pth")

        print()

    # output_tokens = model.generate(input_token, max_new_tokens=300)
    # output_characters = decode(output_tokens[0].tolist())
    # print(f"Output Tokens: {output_tokens}")
    # print(f"Output Characters: {output_characters}")
    # print("========================================")


if __name__ == "__main__":
    main()