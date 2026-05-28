# =========================================================
# IMPROVED HYBRID MODEL: RoBERTa + BiLSTM (SINGLE CELL)
# Higher neutral weight, mean pooling, scheduler, early stopping
# =========================================================

# Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets (files in current directory)
print("\nLoading datasets...")
df1 = pd.read_csv("1429_1.csv")
df2 = pd.read_excel("Customer_Sentiment.xlsx")
df3 = pd.read_csv("Dataset-SA.csv")
df4 = pd.read_excel("Multilingual_Meesho_Review_Dataset.xlsx")
print("Datasets loaded.")

# Helper functions
def convert_rating(rating):
    return 0 if rating <= 2 else (1 if rating == 3 else 2)

# Preprocess each dataset
# Dataset 1
df1 = df1[["reviews.text", "reviews.rating"]].dropna()
df1["sentiment"] = df1["reviews.rating"].apply(convert_rating)
df1 = df1.rename(columns={"reviews.text": "review"})[["review", "sentiment"]]

# Dataset 2
df2 = df2.dropna()
df2["sentiment"] = df2["sentiment"].map({"negative":0, "neutral":1, "positive":2})
df2 = df2.rename(columns={"review_text": "review"})[["review", "sentiment"]]

# Dataset 3
df3 = df3[["Review", "Sentiment"]].dropna()
df3["sentiment"] = df3["Sentiment"].map({"negative":0, "neutral":1, "positive":2})
df3 = df3.rename(columns={"Review": "review"})[["review", "sentiment"]]

# Dataset 4
df4 = df4[["comments", "ratings"]].dropna()
df4["sentiment"] = df4["ratings"].apply(convert_rating)
df4 = df4.rename(columns={"comments": "review"})[["review", "sentiment"]]

# Combine all datasets
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
print("\nFull dataset shape:", combined_df.shape)
print("Original distribution:\n", combined_df["sentiment"].value_counts())

# Optional subsampling (uncomment if needed)
# pos = combined_df[combined_df["sentiment"]==2].sample(n=50000, random_state=42)
# neg = combined_df[combined_df["sentiment"]==0]
# neu = combined_df[combined_df["sentiment"]==1]
# combined_df = pd.concat([neg, neu, pos]).sample(frac=1, random_state=42)
# print("After subsampling:\n", combined_df["sentiment"].value_counts())

# Train/validation/test split (stratified)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    combined_df["review"], combined_df["sentiment"],
    test_size=0.3, stratify=combined_df["sentiment"], random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)
print(f"\nTrain: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# Class weights – higher weight for neutral
class_weights = torch.tensor([2.5, 8.0, 0.5], dtype=torch.float).to(device)
print("Class weights (neg, neu, pos):", class_weights.cpu().numpy())

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Create datasets and loaders
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
val_dataset   = ReviewDataset(val_texts, val_labels, tokenizer)
test_dataset  = ReviewDataset(test_texts, test_labels, tokenizer)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# Improved hybrid model with mean pooling
class RobertaBiLSTM(nn.Module):
    def __init__(self):
        super(RobertaBiLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 3)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        pooled = torch.mean(lstm_output, dim=1)
        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits

model = RobertaBiLSTM().to(device)

# Loss, optimizer, scheduler
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

EPOCHS = 5
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Early stopping based on validation macro F1
best_val_f1 = 0.0
patience = 2
epochs_no_improve = 0

print("\n=== STARTING TRAINING ===\n")
for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_f1_macro = f1_score(val_labels_list, val_preds, average='macro')
    val_acc = accuracy_score(val_labels_list, val_preds)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Acc = {val_acc:.4f}, Val Macro F1 = {val_f1_macro:.4f}")

    # Early stopping
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        torch.save(model.state_dict(), "best_hybrid_model.pth")
        print("  --> Best model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after epoch {epoch+1}")
            break

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_hybrid_model.pth"))
print("\n=== FINAL TEST EVALUATION ===")

model.eval()
test_preds, test_labels_list = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels_list.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels_list, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels_list, test_preds, target_names=["Negative", "Neutral", "Positive"]))
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels_list, test_preds))

# Prediction function
def predict_sentiment(text):
    model.eval()
    encoding = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).item()
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[pred]

# Live interactive demo
print("\n=== LIVE PREDICTION DEMO ===")
while True:
    user_input = input("\nEnter review (or 'exit'): ")
    if user_input.lower() == "exit":
        break
    print("Sentiment:", predict_sentiment(user_input))
    