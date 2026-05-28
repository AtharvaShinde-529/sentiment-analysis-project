RoBERTa + BiLSTM Hybrid Sentiment Model(Higher neutral weight, mean pooling, scheduler, early stopping)

The model processes a review in five stages:

1.Tokenization – RoBERTa tokenizer converts text to fixed‑length token IDs (max 128) with an attention mask.
2.RoBERTa encoder – 12 transformer layers produce contextualised embeddings (768‑d per token), capturing word meaning based on surrounding text.
3.BiLSTM – A bidirectional LSTM (128 hidden units each direction) reads the sequence and outputs refined hidden states (256‑d per token), capturing order‑dependent sentiment flow.
4.Mean pooling – Averages the BiLSTM outputs over the entire sequence, producing a single 256‑d vector that represents the whole review.
5.Classification – A linear layer maps the vector to 3 logits (Negative/Neutral/Positive); softmax gives probabilities.

Key training improvements:

1.Class weights (Neg:2.5, Neu:8.0, Pos:0.5) to handle imbalance.
2.Learning rate warmup + linear decay for stable fine‑tuning.
3.Gradient clipping (max norm 1.0) to prevent exploding gradients.
4.Early stopping based on validation macro F1 (patience=2).

Note on working:(acc.:90.60%)
RoBERTa provides deep context, BiLSTM adds sequential reasoning, mean pooling retains full‑review information, and weighted loss forces the model to learn rare neutral cases.
