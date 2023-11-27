import pandas as pd
import torch
import chess
import chess.pgn
import numpy as np
import io
import re
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from chessCNN import ChessCNN

# Step 1: get the data
chess_eval_games = pd.read_csv('chess-engine/chessData.csv')

print(chess_eval_games)

# Step 2: preprocessing
def parse_fen(fen):
    parts = re.split(" ", fen)
    return {
        'piece_placement': re.split("/", parts[0]),
        'active_color': parts[1],
        'castling_rights': parts[2],
        'en_passant': parts[3],
        'halfmove_clock': int(parts[4]),
        'fullmove_clock': int(parts[5])
    }

def place_pieces(bit_vector, piece_placement, piece_to_layer):
    for r, row in enumerate(piece_placement):
        c = 0
        for piece in row:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece], r, c] = 1
                c += 1
            else:
                c += int(piece)

def encode_special_moves(bit_vector, parsed_fen):
    # Castling rights
    for char in parsed_fen['castling_rights']:
        if char == 'K':  bit_vector[0, 7, 7] = 1  # White can castle kingside
        elif char == 'Q':  bit_vector[0, 7, 0] = 1  # White can castle queenside
        elif char == 'k':  bit_vector[0, 0, 7] = 1  # Black can castle kingside
        elif char == 'q':  bit_vector[0, 0, 0] = 1  # Black can castle queenside
    
    # En passant square
    if parsed_fen['en_passant'] != '-':
        col = ord(parsed_fen['en_passant'][0]) - ord('a')
        row = int(parsed_fen['en_passant'][1]) - 1
        bit_vector[0, row, col] = 1
    
    # Active color
    if parsed_fen['active_color'] == 'w':
        bit_vector[0, 7, 4] = 1  # Indicate white's turn
    else:
        bit_vector[0, 0, 4] = 1  # Indicate black's turn
    
    # Halfmove clock
    # Encoding in the 3rd row, with binary encoding
    halfmove_clock = parsed_fen['halfmove_clock']
    for c in range(8):
        bit_vector[0, 2, c] = halfmove_clock % 2
        halfmove_clock = halfmove_clock // 2

    # Fullmove number
    # Encoding in the 4th row, with binary encoding
    fullmove_clock = parsed_fen['fullmove_clock']
    for c in range(8):
        bit_vector[0, 3, c] = fullmove_clock % 2
        fullmove_clock = fullmove_clock // 2

def fen_to_bit_vector(fen):
    parsed_fen = parse_fen(fen)
    
    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)
    piece_to_layer = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    place_pieces(bit_vector, parsed_fen['piece_placement'], piece_to_layer)
    encode_special_moves(bit_vector, parsed_fen)

    return bit_vector

def eval_to_int(evaluation):
    try:
        return int(evaluation)
    except ValueError:
        if '#' in evaluation:
            return 8500 if '+' in evaluation else -8500

# apply functions to handle checkmate positions then get the board representations

# Step 3: split into training and validation
scaler = MinMaxScaler(feature_range=(-1, 1))

train_data, val_data = train_test_split(chess_eval_games[:400000], test_size=0.2, random_state=42)

# Convert FEN to bit-vector representations
X_train = np.stack(train_data['FEN'].apply(fen_to_bit_vector).values)
X_val = np.stack(val_data['FEN'].apply(fen_to_bit_vector).values)

# Apply the function to convert evaluations to integers
# Note: Ensure the 'Evaluation' column is of string type before applying the function
train_data['Evaluation'] = train_data['Evaluation'].astype(str).apply(eval_to_int)
val_data['Evaluation'] = val_data['Evaluation'].astype(str).apply(eval_to_int)

# Normalize the evaluation labels
y_train = train_data['Evaluation'].values.reshape(-1, 1)
y_val = val_data['Evaluation'].values.reshape(-1, 1)

y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)  

# Step 4: training

# convert data to PyTorch tensors + create DataLoader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensuring the correct shape

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  # Adjust batch size as needed

# init the model, define loss function and optimizer
model = ChessCNN()
criterion = nn.MSELoss()  # using MSE for loss
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# now we train the model
num_epochs = 25 

for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # perform forward pass
        outputs = model(inputs)
        
        # calculate the loss
        loss = criterion(outputs, labels)
        
        # go backward and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

# step 5: validation
# Assuming X_val and y_val are your validation data and labels (unscaled)
# Convert to tensors
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Create DataLoader
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=32)  # Adjust batch size as needed

model.eval()  # Set the model to evaluation mode
val_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():  # No need to track gradients
    for inputs, labels in val_loader:
        # Forward pass
        outputs = model(inputs)
        
        # Compute validation loss
        loss = criterion(outputs, labels)
        
        val_loss += loss.item()
        
        # Save predictions and actual labels for further analysis
        all_preds.append(outputs.numpy())
        all_labels.append(labels.numpy())

# Concatenate all predictions and labels (they were saved in batches)
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Compute additional evaluation metrics
mae = mean_absolute_error(all_labels, all_preds)

# Print evaluation metrics
print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
print(f'Mean Absolute Error on Validation Set: {mae:.4f}')

all_preds = scaler.inverse_transform(all_preds)
all_labels = scaler.inverse_transform(all_labels)

# Example predictions
num_examples = 100  # Number of examples to print
print("\nExample Predictions:")
print("Predicted | Actual")
for i in range(num_examples):
    print(f"{all_preds[i, 0]:.2f} | {all_labels[i, 0]:.2f}")

# Step 6: making a move
def make_a_move(model, board):
    best_move = None
    best_value = -float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        current_fen = board.fen()
        bit_vector = fen_to_bit_vector(current_fen)
        input_tensor = torch.tensor(bit_vector, dtype=torch.float32).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            value = model(input_tensor).item()
        
        if value > best_value:
            best_value = value
            best_move = move
        
        board.pop()
    
    return best_move.uci()  # return move in algebraic notation

# Step 7: play game vs engine w/ outputs on every move
def play_game(model, num_moves=100):
    board = chess.Board()
    
    for _ in range(num_moves):
        # Display the board
        print(board)
        
        # Check whose turn it is and make a move
        if not board.turn:  # True for white, False for black
            # User's move
            valid_move = False
            while not valid_move:
                user_move = input("Enter your move: ")
                try:
                    board.push_san(user_move)
                    valid_move = True
                except ValueError:  # Invalid move
                    print("Invalid move. Please use standard algebraic notation (e.g., e4, Nf3).")
        else:
            # Model's move
            move = make_a_move(model, board)
            print(f"Model moves: {move}")
            board.push_uci(move)
        
        # Show the evaluation
        current_fen = board.fen()
        bit_vector = fen_to_bit_vector(current_fen)
        input_tensor = torch.tensor(bit_vector, dtype=torch.float32).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            value = model(input_tensor).item()
        
        # Rescale the value to the original score
        value = scaler.inverse_transform([[value]])[0][0]
        print(f"Evaluation: {value:.2f}")
        
        # Check for game over
        if board.is_game_over():
            print("Game Over")
            print("Result: " + board.result())
            break

play_game(model)