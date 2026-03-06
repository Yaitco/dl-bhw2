from typing import Counter
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self, min_count=5):
        self.min_count = min_count

        self.pad_id = 0
        self.eos_id = 1
        self.bos_id = 2
        self.unk_id = 3

        self.word_to_idx = {
            '<pad>': 0,
            '<eos>': 1,
            '<bos>': 2,
            '<unk>': 3,
        }

        self.idx_to_word = [
            '<pad>',
            '<eos>',
            '<bos>',
            '<unk>',
        ]

    def __len__(self):
        return len(self.idx_to_word)

    def encode(self, line, bos_eos=False):
        encoded = [self.word_to_idx.get(word, self.unk_id) for word in line.split()]
        if bos_eos:
            encoded = [self.bos_id] + encoded + [self.eos_id]
        return encoded

    def decode(self, ids):
        return [self.idx_to_word[i] for i in ids]
    
    def fit(self, texts):
        counter = Counter()
        for line in texts:
            counter.update(line.split())

        for token, count in counter.most_common():
            if count >= self.min_count:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word.append(token)
        return self

    def transform(self, texts, bos_eos=False):
        return [self.encode(line, bos_eos=bos_eos) for line in texts]


class TranslationDataset(Dataset):
    def __init__(self, source_path, target_path, source_vocab: Vocabulary, target_vocab: Vocabulary):
        super().__init__()
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        with open(source_path, "r", encoding="utf-8") as f:
            self.source = [line.strip() for line in f]

        with open(target_path, "r", encoding="utf-8") as f:
            self.target = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_encoded = self.source_vocab.encode(self.source[index])
        target_encoded = self.target_vocab.encode(self.target[index])
        return source_encoded, target_encoded
    

class TestDataset(Dataset):
    def __init__(self, source_path):
        super().__init__()
        with open(source_path, "r", encoding="utf-8") as f:
            self.source = [line.strip("\n") for line in f]
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index]
    