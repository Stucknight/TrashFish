import random

import chess
import chess.engine
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models

model = models.load_model('model.keras')


def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci('stockfish/stockfish-windows-x86-64-avx2.exe') as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score()
        return score


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


def minimax_eval(board):
    board3d = convert_matrix(board)
    board3d = np.expand_dims(board3d, 0)
    return model(board3d)[0][0]


def test():
    board = create_board(20)
    if board is None:
        return None
    t_evaluation = minimax_eval(board)
    evaluation = str(t_evaluation)[10:str(t_evaluation).find(",")]
    s_evaluation = stockfish(board, 10)
    p_diff = abs((float(evaluation) - s_evaluation) / (1 / 2 * (float(evaluation) + s_evaluation))) * 100
    return p_diff


x = []
y = []
for i in range(100):
    a = test()
    if a is not None:
        x.append(i)
        y.append(a)

plt.plot(np.array(x), np.array(y))
plt.show()
