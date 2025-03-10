import torch
import torch.nn as nn
import torch.optim as optim
import os

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step output
        return out

    def train_lstm(self, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
        """
        Trains the LSTM model using train and validation DataLoaders.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of epochs to train.
            learning_rate: Learning rate for optimizer.

        Returns:
            Trained model with best validation performance.
        """
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        best_val_loss = float("inf")
        patience = 3  # Early stopping patience
        no_improve = 0

        for epoch in range(num_epochs):
            # Training Phase
            self.train()
            total_train_loss = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation Phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self(data)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                self.save_model("best_lstm_model.pth")  # Save best model
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete!")
        return self

    def forecast(self, input_seq):
        """ Forecast the next value based on input sequence. """
        self.eval()
        with torch.no_grad():
            input_seq = input_seq.to(self.device)
            output = self(input_seq)
        return output.cpu().numpy()

    def extract_features(self, input_seq):
        """ Extracts LSTM hidden state as features. """
        self.eval()
        with torch.no_grad():
            input_seq = input_seq.to(self.device)
            h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(self.device)

            _, (hn, _) = self.lstm(input_seq, (h0, c0))
        return hn[-1].cpu().numpy()  # Take the last hidden state layer

    def save_model(self, file_path="lstm_model.pth"):
        """ Saves the model to a .pth file. """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path="lstm_model.pth"):
        """ Loads the model from a .pth file. """
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return False

        try:
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            self.to(self.device)
            print(f"Model successfully loaded from '{file_path}'")
            return True
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            return False
        
    def main_train(self, train_loader, val_loader, num_epochs, learning_rate):
        self = self.train_lstm(self, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, device=device)

    # Example data (random tensor as placeholder for real training)
    sample_input = torch.randn(1, 10, input_size).to(device)  # (batch, sequence_length, input_size)
    
    # Forecast
    forecasted_value = model.forecast(sample_input)
    print("Forecasted Value:", forecasted_value)

    # Feature Extraction
    features = model.extract_features(sample_input)
    print("Extracted Features Shape:", features.shape)

    # Save Model
    model.save_model("lstm_model.pth")

    # Load Model
    new_model = LSTMModel(input_size, hidden_size, num_layers, output_size, device=device)
    new_model.load_model("lstm_model.pth")
