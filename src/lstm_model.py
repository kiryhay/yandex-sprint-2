import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out)
        return logits

    def generate(self, input_ids, max_new_tokens=10, eos_token_id=2):
        self.eval()
        with torch.no_grad():
            current = input_ids
            for _ in range(max_new_tokens):
                logits = self(current)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                current = torch.cat([current, next_token], dim=1)
                if next_token.item() == eos_token_id:
                    break
        return current