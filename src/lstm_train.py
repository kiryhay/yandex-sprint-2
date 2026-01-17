import torch
from torch.utils.data import DataLoader
from .lstm_model import LSTMLanguageModel
from .next_token_dataset import NextTokenDataset, collate_fn
import os

def train_lstm():
    os.makedirs('models', exist_ok=True)
    
    with open('data/train.csv', encoding='utf-8') as f:
        train_texts = [line.strip() for line in f if line.strip()]
    with open('data/val.csv', encoding='utf-8') as f:
        val_texts = [line.strip() for line in f if line.strip()]

    train_dataset = NextTokenDataset(train_texts, max_len=32)
    val_dataset = NextTokenDataset(val_texts, vocab=train_dataset.word2idx, max_len=32)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = LSTMLanguageModel(
        vocab_size=train_dataset.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/3 | Loss: {avg_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': train_dataset.word2idx
    }, 'models/lstm.pth')
    return model, train_dataset.word2idx