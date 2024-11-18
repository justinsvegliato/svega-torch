import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):
    def __init__(self, filename, sequence_length, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        if self.transform:
            text = self.transform(text)

        self.vocabulary = sorted(list(set(text)))

        self.char_to_int_map = {char: integer for integer, char in enumerate(self.vocabulary)}
        self.int_to_char_map = {integer: char for integer, char in enumerate(self.vocabulary)}

        self.dataset = torch.tensor([self.char_to_int_map[char] for char in text], dtype=torch.long)
        
    def __len__(self):
        return len(self.dataset) - self.sequence_length

    def __getitem__(self, index):
        sequence = self.dataset[index:index + self.sequence_length]
        target = self.dataset[index + 1:index + self.sequence_length + 1]
        return sequence, target
    
    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def encode(self, string):
        return [self.char_to_int_map[char] for char in string]

    def decode(self, tokens):
        return "".join([self.int_to_char_map[token] for token in tokens])
