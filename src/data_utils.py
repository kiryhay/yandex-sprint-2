import pandas as pd
import re
import os
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()

def prepare_dataset():
    os.makedirs('data', exist_ok=True)
    raw_path = 'data/raw_dataset.csv'
    
    if not os.path.exists(raw_path):
        url = "https://code.s3.yandex.net/deep-learning/tweets.txt"
        urlretrieve(url, "data/tweets.txt")

        with open("data/tweets.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Загружено {len(lines)} строк. Очищаем...")
        cleaned = []
        for line in lines:
            ct = clean_text(line)
            if len(ct.split()) >= 3:
                cleaned.append(ct)

        tokenized = [' '.join(tokenize(t)) for t in cleaned]

        pd.Series(tokenized).to_csv(raw_path, index=False, header=False, encoding='utf-8')

    df = pd.read_csv(raw_path, header=None, names=['text'], encoding='utf-8')
    texts = df['text'].dropna().tolist()

    train, temp = train_test_split(texts, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    pd.Series(train).to_csv('data/train.csv', index=False, header=False, encoding='utf-8')
    pd.Series(val).to_csv('data/val.csv', index=False, header=False, encoding='utf-8')
    pd.Series(test).to_csv('data/test.csv', index=False, header=False, encoding='utf-8')

    return train, val, test