import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import io
import os
import matplotlib.pyplot as plt


# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load and preprocess dataset
data_path = 'shakespeare_train.txt'  # Replace with your dataset path
with io.open(data_path, 'r', encoding='utf8') as f:
    text = f.read()

# Vocabulary and mappings
vocab = sorted(set(text))
vocab_size = len(vocab)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = {i: c for i, c in enumerate(vocab)}

# Encode data as integers
data = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

# Load and preprocess validation dataset
valid_path = 'shakespeare_valid.txt'  # Replace with the path to your validation file
with io.open(valid_path, 'r', encoding='utf8') as f:
    valid_text = f.read()

# Encode validation data as integers
valid_data = np.array([vocab_to_int[c] for c in valid_text], dtype=np.int32)


# Hyperparameters
sequence_length = 150
batch_size = 64
hidden_size = 512
num_layers = 2
learning_rate = 0.002
epochs = 20
loss_compare = [] 

# Data loader
def get_batches(data, batch_size, seq_length):
    num_batches = len(data) // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = data.reshape((batch_size, -1))
    for n in range(0, data.shape[1], seq_length):
        x = data[:, n:n + seq_length]
        y = np.zeros_like(x)
        if n + seq_length < data.shape[1]:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, n + seq_length]
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Model definitions
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, rnn_type="LSTM"):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        if rnn_type == "RNN":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose 'RNN' or 'LSTM'.")
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Training function
# Training function with BPC minimization
import csv  # For saving BPC values in a CSV file

def train(model, data, epochs, batch_size, seq_length, lr):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_bpc_values = []  # Store training BPC values
    valid_bpc_values = []  # Store validation BPC values
    # Open a CSV file to record epoch and BPC
    with open("training_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "BPC", "VALID_BPC", "train_error_rate", "valid_error_rate"])  # Write header
        
        for epoch in range(epochs):
            # Initialize hidden state to zero at the start of each epoch
            hidden = model.init_hidden(batch_size)
            total_loss, total_errors, total_predictions = 0, 0, 0
            model.train()

            for batch_idx, (x, y) in enumerate(get_batches(data, batch_size, seq_length)):
                x, y = x.to(device), y.to(device)

                # Detach hidden state after the first batch in an epoch
                if batch_idx > 0:
                    hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

                output, hidden = model(x, hidden)

                # Calculate loss
                loss = criterion(output.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()
                
                # Compute predictions and error rate
                predictions = torch.argmax(output, dim=-1)
                total_errors += (predictions != y).sum().item()
                total_predictions += y.numel()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate average BPC
            avg_loss = total_loss / len(data) * batch_size * seq_length
            bpc = avg_loss / np.log(2)  # Convert loss to BPC
            train_bpc_values.append(bpc)
            train_error_rate = (total_errors / total_predictions) * 100
            
            valid_bpc, valid_error_rate = evaluate(model, valid_data, batch_size, seq_length)
            valid_bpc_values.append(valid_bpc)

            # Write to CSV
            writer.writerow([epoch + 1, bpc, valid_bpc, train_error_rate, valid_error_rate])
            
            print(f"Epoch {epoch+1}/{epochs}, Train BPC: {bpc:.4f}, Valid BPC: {valid_bpc:.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Train Error: {train_error_rate:.4f}, Valid Error: {valid_error_rate:.4f}")
            generate_text(model, "JULIET", 1000)
            if epoch == epochs-1:
                loss_compare.append(bpc)


def evaluate(model, valid_data, batch_size, seq_length):
    model.eval()
    total_loss, total_errors, total_predictions = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for x, y in get_batches(valid_data, batch_size, seq_length):
            x, y = x.to(device), y.to(device)
            output, hidden = model(x, hidden)

            # Compute loss
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()

            # Compute predictions and error rate
            predictions = torch.argmax(output, dim=-1)
            total_errors += (predictions != y).sum().item()
            total_predictions += y.numel()

            hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()

    # Calculate validation metrics
    valid_bpc = (total_loss / len(valid_data) * batch_size * seq_length) / np.log(2)
    valid_error_rate = (total_errors / total_predictions) * 100
    return valid_bpc, valid_error_rate




# Text generation
def generate_text(model, start_text, length):
    model.eval()
    input_text = torch.tensor([vocab_to_int[c] for c in start_text], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    result = start_text
    
    for _ in range(length):
        output, hidden = model(input_text, hidden)
        prob = torch.nn.functional.softmax(output[0, -1], dim=-1).data  # Fix: Select correct output
        char_idx = torch.multinomial(prob, 1).item()
        result += int_to_vocab[char_idx]
        input_text = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    print(f"Generated text: {result}\n")
    return result


# Instantiate and train RNN
# hidden_size = 32
# sequence_length = 25
for i in range(1):
    rnn_model = CharRNN(vocab_size, hidden_size, num_layers, rnn_type="LSTM")
    train(rnn_model, data, epochs, batch_size, sequence_length, learning_rate)
    # hidden_size = hidden_size * 2
    # sequence_length = sequence_length + 25
print(f"loss_compare_hidden_size: {loss_compare}")



