import torch
from torch.utils.data import Dataset
from collections import Counter

class NextTokenDataset(Dataset):
    def __init__(self, texts, vocab=None, max_len=32, vocab_size_limit=1000):
        self.max_len = max_len
        self.texts = [text.split()[:max_len] for text in texts]
        
        if vocab is None:
            all_words = [word for text in self.texts for word in text]
            counter = Counter(all_words)
            most_common = counter.most_common(vocab_size_limit - 3)
            vocab_words = ['<pad>', '<unk>', '<eos>'] + [w for w, _ in most_common]
            self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        else:
            self.word2idx = vocab
        
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx]
        indices = [self.word2idx.get(w, self.word2idx['<unk>']) for w in words]
        indices.append(self.word2idx['<eos>'])
        indices = indices[:self.max_len + 1]
        return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    x = padded[:, :-1]
    y = padded[:, 1:]
    return x, y