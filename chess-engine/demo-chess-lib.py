import chess

# Initialize the chess board
board = chess.Board()

# Display the chess board
print(board)

print("\n------ MOVE ONE ------\n")

# Make move as white
move = chess.Move.from_uci("e2e4")
if move in board.legal_moves:
    board.push(move)

# Make move as black
move = chess.Move.from_uci("e7e5")
if move in board.legal_moves:
    board.push(move)

# Display the chess board after moves are made
print(board)