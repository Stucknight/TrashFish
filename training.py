import chess
import random
import numpy as np
import chess.engine
from tensorflow.keras import models, layers, optimizers


def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-windows-x86-64-avx2.exe') as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        return result['score'].white().score()


def create_board(depth):
    board = chess.Board()
    board_depth = random.randint(0, depth)

    for _ in range(board_depth):
        move = random.choice(list(board.legal_moves))
        board.push(move)
        if board.is_game_over():
            return None
    return board


def convert_matrix(board):
    board3d = np.zeros((13, 8, 8), dtype=np.int8)
    for square, piece in board.piece_map().items():
        color = piece.color
        index = np.unravel_index(square, (8, 8))
        p_index = piece.piece_type - 1 if color == chess.WHITE else piece.piece_type + 5
        board3d[p_index][7 - index[0]][index[1]] = 1
    return board3d


def train_model(size):
    model = models.Sequential([
        layers.Conv2D(size, kernel_size=4, padding='same', activation='relu', input_shape=(13, 8, 8)),
        layers.Conv2D(size, kernel_size=4, padding='same', activation='relu'),
        layers.Conv2D(size, kernel_size=4, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(), loss='mean_squared_error')
    return model


def generate_data(num_samples, depth):
    X, y = [], []
    for _ in range(num_samples):
        board = create_board(depth)
        if board is not None:
            X.append(convert_matrix(board))
            y.append(stockfish(board, 10))
    return np.array(X), np.array(y)


def train(size, num_samples, epochs):
    model = train_model(size)
    X, y = generate_data(num_samples, 20)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    return model


model = train(size=32, num_samples=100, epochs=100)
model.save('model.keras')
