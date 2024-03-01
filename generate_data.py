import random

import chess
import chess.engine
import numpy as np


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
    board3d = np.zeros((12, 8, 8), dtype=np.int8)
    for square, piece in board.piece_map().items():
        color = piece.color
        index = np.unravel_index(square, (8, 8))
        p_index = piece.piece_type - 1 if color == chess.WHITE else piece.piece_type + 5
        board3d[p_index][7 - index[0]][index[1]] = 1
    return board3d


def generate_data(num_samples, depth, filename):
    X, y = [], []
    for _ in range(num_samples):
        board = create_board(depth)
        if board is not None:
            X.append(convert_matrix(board))
            y.append(stockfish(board, 20))
    X = np.array(X)
    y = np.array(y)
    print(y)
    np.savez(filename, X=X, y=y)


generate_data(100, 20, "dataset.npz")
