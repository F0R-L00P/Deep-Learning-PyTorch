import re
import time
import torch
import random
import string
import unidecode
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Configuration
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
RANDOM_SEED = 123
TEXT_PORTION_SIZE = 200
NUM_ITER = 2000
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
BATCH_SIZE = 512
LOG_INTERVAL = 250

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Automatic mixed precision
scaler = torch.amp.GradScaler()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
#                   Dataset Definition
# --------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        # Map each character to an index and store on the GPU (if available)
        self.text = torch.tensor(
            [char_to_idx[c] for c in text], dtype=torch.long, device=DEVICE
        )
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        return (
            self.text[idx : idx + self.seq_length],
            self.text[idx + 1 : idx + self.seq_length + 1],
        )


# --------------------------------------------------
#                   Data Preparation
# --------------------------------------------------
text = unidecode.unidecode(open("covid19-faq.txt", "r", encoding="utf-8").read())
text = re.sub(r"\s+", " ", text).strip()

chars = string.printable
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Replace characters not in `chars` with a space
text = "".join(c if c in chars else " " for c in text)

dataset = TextDataset(text, TEXT_PORTION_SIZE)
BATCH_SIZE = min(BATCH_SIZE, len(dataset))

# Use a single worker on Windows if no __name__ guard
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
)


# --------------------------------------------------
#                   Model Definition
# --------------------------------------------------
class CharLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(len(chars), EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(
            EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, num_layers=8
        )
        self.fc = torch.nn.Linear(HIDDEN_DIM, len(chars))

    def forward(self, x, hidden=None):
        x = self.embed(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


model = CharLSTM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optional PyTorch 2.0 compile for optimization
# Comment this out if using an older PyTorch version
model = torch.compile(model)


# --------------------------------------------------
#                   Training Loop
# --------------------------------------------------
def train():
    model.train()
    total_loss = 0
    start_time = time.time()
    loss_history = []
    iteration = 0
    target_iterations = NUM_ITER

    # Keep looping until we hit the total number of iterations
    while iteration < target_iterations:
        for data, targets in dataloader:
            if iteration >= target_iterations:
                break

            with torch.amp.autocast("cuda"):
                outputs, _ = model(data)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, len(chars)), targets.view(-1)
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss_val = loss.item()
            loss_history.append(current_loss_val)
            total_loss += current_loss_val

            # Logging
            if iteration % LOG_INTERVAL == 0 and iteration > 0:
                avg_loss = total_loss / LOG_INTERVAL
                elapsed = time.time() - start_time
                print(
                    f"| Iteration {iteration:5d} | "
                    f"lr {LEARNING_RATE:.4f} | "
                    f"ms/iteration {elapsed * 1000 / LOG_INTERVAL:5.2f} | "
                    f"loss {avg_loss:5.2f}"
                )
                total_loss = 0
                start_time = time.time()

            iteration += 1

    return loss_history


# --------------------------------------------------
#             Run Training + Visualization
# --------------------------------------------------
losses = train()
torch.save(model.state_dict(), "lstm_final.pt")

# Plotting the recorded losses
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Iteration (batch)")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss over Iterations")
plt.legend()
plt.show()

# --------------------------------------------------


# --- Sample from the Model ---
def generate_text(model, start_str="<START>", length=200):
    """
    Generate text from the model given a starting string.

    Args:
      model: Trained CharLSTM model.
      start_str: The initial text to start generation.
      length: Number of characters to generate.

    Returns:
      A string containing the generated text.
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Convert start string to a list of indices
    input_indices = [char_to_idx.get(c, char_to_idx[" "]) for c in start_str]
    input_tensor = torch.tensor(
        input_indices, dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    # Initialize LSTM hidden state
    hidden = None

    # We'll collect predicted characters here
    generated = start_str

    with torch.no_grad():
        for _ in range(length):
            # Forward pass: last character(s)
            output, hidden = model(input_tensor, hidden)

            # The output at the final timestep is what we use for the next char
            # output shape: [batch_size, seq_length, num_chars]
            # We only want the last timestep -> output[:, -1, :]
            last_step_logits = output[:, -1, :]

            # Sample from the probability distribution using a multinomial draw
            # (Alternatively, you can use argmax for a 'greedy' approach)
            probs = torch.softmax(last_step_logits, dim=1)
            next_char_idx = torch.multinomial(probs, 1).item()

            # Append predicted character to generated text
            generated += idx_to_char[next_char_idx]

            # Update input_tensor to be the predicted character (for the next iteration)
            input_tensor = torch.tensor(
                [[next_char_idx]], dtype=torch.long, device=DEVICE
            )

    return generated


# Generate text from the model
output = generate_text(model, start_str="Lung", length=200)
print(output)
