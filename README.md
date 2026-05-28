Short Summary: RoBERTa + BiLSTM Hybrid Sentiment Model
The model processes a review in five stages:

Tokenization – RoBERTa tokenizer converts text to fixed‑length token IDs (max 128) with an attention mask.

RoBERTa encoder – 12 transformer layers produce contextualised embeddings (768‑d per token), capturing word meaning based on surrounding text.

BiLSTM – A bidirectional LSTM (128 hidden units each direction) reads the sequence and outputs refined hidden states (256‑d per token), capturing order‑dependent sentiment flow.

Mean pooling – Averages the BiLSTM outputs over the entire sequence, producing a single 256‑d vector that represents the whole review.

Classification – A linear layer maps the vector to 3 logits (Negative/Neutral/Positive); softmax gives probabilities.

Key training improvements:

Class weights (Neg:2.5, Neu:8.0, Pos:0.5) to handle imbalance.

Learning rate warmup + linear decay for stable fine‑tuning.

Gradient clipping (max norm 1.0) to prevent exploding gradients.

Early stopping based on validation macro F1 (patience=2).

Why it works well (≈90% accuracy):
RoBERTa provides deep context, BiLSTM adds sequential reasoning, mean pooling retains full‑review information, and weighted loss forces the model to learn rare neutral cases.
