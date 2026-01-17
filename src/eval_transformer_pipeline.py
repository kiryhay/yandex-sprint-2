from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings("ignore")

def evaluate_transformer():
    generator = pipeline("text-generation", model="distilgpt2", device=0)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    test_texts = open('data/test.csv', encoding='utf-8').read().splitlines()[:100]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)

    predictions = []
    references = []

    for text in test_texts:
        words = text.split()
        if len(words) < 4:
            continue
        split_idx = max(1, int(0.75 * len(words)))
        prefix = ' '.join(words[:split_idx])
        target = ' '.join(words[split_idx:])

        try:
            result = generator(
                prefix,
                max_length=30,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                truncation=True
            )
            generated = result[0]['generated_text']
            if generated.startswith(prefix):
                pred = generated[len(prefix):].strip()
            else:
                pred = generated.strip()
        except:
            pred = ""

        if pred.strip() and target.strip():
            predictions.append(pred)
            references.append(target)

    rouge1, rouge2 = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)

    avg_rouge1 = sum(rouge1) / len(rouge1) if rouge1 else 0
    avg_rouge2 = sum(rouge2) / len(rouge2) if rouge2 else 0

    examples = list(zip(references[:5], predictions[:5]))
    return {'rouge1': avg_rouge1, 'rouge2': avg_rouge2}, examples