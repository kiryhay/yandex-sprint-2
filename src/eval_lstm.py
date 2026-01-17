import torch
from .lstm_model import LSTMLanguageModel
from rouge_score import rouge_scorer

def evaluate_lstm():
    checkpoint = torch.load('models/lstm.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu')
    vocab = checkpoint['vocab']
    model = LSTMLanguageModel(vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with open('data/test.csv', encoding='utf-8') as f:
        test_texts = [line.strip() for line in f if line.strip()][:100]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    idx2word = {i: w for w, i in vocab.items()}
    predictions, references = [], []

    for text in test_texts:
        words = text.split()
        if len(words) < 4:
            continue
        split_idx = max(1, int(0.75 * len(words)))
        prefix_words = words[:split_idx]
        target = ' '.join(words[split_idx:])

        input_ids = [vocab.get(w, vocab['<unk>']) for w in prefix_words]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output_ids = model.generate(input_tensor, max_new_tokens=10, eos_token_id=vocab['<eos>'])

        output_cpu = output_ids.cpu().numpy()[0]
        gen_words = []
        for idx in output_cpu[len(input_ids):]:
            if idx == vocab['<eos>']:
                break
            gen_words.append(idx2word.get(int(idx), '<unk>'))
        pred = ' '.join(gen_words)

        if pred.strip() and target.strip():
            predictions.append(pred)
            references.append(target)

    if not predictions:
        return {'rouge1': 0, 'rouge2': 0}, []

    rouge1 = [scorer.score(ref, pred)['rouge1'].fmeasure for ref, pred in zip(references, predictions)]
    rouge2 = [scorer.score(ref, pred)['rouge2'].fmeasure for ref, pred in zip(references, predictions)]

    return {
        'rouge1': sum(rouge1) / len(rouge1),
        'rouge2': sum(rouge2) / len(rouge2)
    }, list(zip(references[:5], predictions[:5]))