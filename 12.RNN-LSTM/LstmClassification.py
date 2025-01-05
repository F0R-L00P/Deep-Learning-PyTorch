import time
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


# General settings
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

NUM_EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if GPU is available
print("GPU is available" if torch.cuda.is_available() else "GPU is not available")

# Load data
df = pd.read_csv("movie_data.csv.gz", compression="gzip")
texts = df["review"].tolist()
labels = df["sentiment"].tolist()

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=RANDOM_SEED
)

# Convert to Hugging Face Dataset format
train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_data = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Initialize Hugging Face Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Tokenizer function
def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
        add_special_tokens=True,
    )
    tokenized["labels"] = examples["label"]
    return tokenized


# Tokenize datasets
train_data = train_data.map(preprocess_function, batched=True, remove_columns=["text"])
test_data = test_data.map(preprocess_function, batched=True, remove_columns=["text"])

# Verify tokenized dataset
print(train_data[0])

# Initialize DataCollator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare DataLoaders
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator
)

# Test the DataLoader
print("Train DataLoader:")
for batch in train_loader:
    print(f"Text matrix size: {batch['input_ids'].size()}")  # Token IDs
    print(f"Attention mask size: {batch['attention_mask'].size()}")  # Padding mask
    print(f"Target vector size: {batch['labels'].size()}")  # Labels
    break


# model definition/architecture
class RNN_LSTM(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text dim: [sentence length, batch size]
        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded)
        # hidden dim: [1, batch size, hidden dim]
        hidden = hidden.squeeze(0)  # Remove the singleton dimension
        # hidden dim: [batch size, hidden dim]
        return self.fc(hidden)  # Output shape: [batch_size, output_dim]


# Ensure reproducibility
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Check vocabulary size
print(f"Vocabulary size: {tokenizer.vocab_size}")

# Initialize model
model = RNN_LSTM(
    input_dim=tokenizer.vocab_size,  # Use tokenizer's vocabulary size
    embedding_dim=128,
    hidden_dim=256,
    output_dim=2,  # Use NUM_CLASSES=2 for binary classification
)

# Move model to GPU
model = model.to(DEVICE)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()  # For multi-class or binary with NUM_CLASSES=2
criterion = criterion.to(DEVICE)


# Helper function for accuracy computation
def compute_accuracy(model, data_loader, device):
    model.eval()  # Evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in data_loader:
            text = (
                batch_data["input_ids"].transpose(0, 1).to(device)
            )  # Transpose input_ids
            labels = batch_data["labels"].to(device)

            logits = model(text)
            _, predicted = torch.max(logits, 1)  # Get predicted classes
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100


# Training loop
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        text = batch_data["input_ids"].transpose(0, 1).to(DEVICE)  # Transpose input_ids
        labels = batch_data["labels"].to(DEVICE)

        # Forward pass and loss computation
        logits = model(text)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % 50 == 0:
            print(
                f"Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | "
                f"Batch {batch_idx:03d}/{len(train_loader):03d} | "
                f"Loss: {loss:.4f}"
            )

    # Compute training and validation accuracy
    train_acc = compute_accuracy(model, train_loader, DEVICE)
    valid_acc = compute_accuracy(model, test_loader, DEVICE)

    print(f"\nEpoch {epoch+1:03d}/{NUM_EPOCHS:03d} Summary:")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {valid_acc:.2f}%")
    print(f"Time elapsed: {(time.time() - start_time)/60:.2f} min\n")

# Final evaluation on test set
test_acc = compute_accuracy(model, test_loader, DEVICE)
print(f"Total Training Time: {(time.time() - start_time)/60:.2f} min")
print(f"Test Accuracy: {test_acc:.2f}%")


def predict_sentiment(model, sentence, tokenizer, device):
    model.eval()  # Set the model to evaluation mode

    # Tokenize and prepare input
    tokenized = tokenizer(
        sentence,
        truncation=True,
        max_length=512,
        padding="max_length",  # Ensure consistent input length
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Move input tensors to the correct device
    input_ids = tokenized["input_ids"].to(device).transpose(0, 1)  # Transpose for LSTM
    attention_mask = tokenized["attention_mask"].to(device)

    # Perform prediction
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    return probabilities[0].cpu().numpy()  # Convert to NumPy for easy access


# Positive sentiment example
sentence1 = "This is such an awesome movie, I really love it!"
probabilities1 = predict_sentiment(model, sentence1, tokenizer, DEVICE)
print(f"Probability positive: {probabilities1[1]:.4f}")  # index 1 = positive
print(f"Probability negative: {probabilities1[0]:.4f}")  # index 0 = negative

# Negative sentiment example
sentence2 = "I really hate this movie. It is really bad and sucks!"
probabilities2 = predict_sentiment(model, sentence2, tokenizer, DEVICE)
print(f"Probability positive: {probabilities2[1]:.4f}")
print(f"Probability negative: {probabilities2[0]:.4f}")
