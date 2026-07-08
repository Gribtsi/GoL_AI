from collections import namedtuple
from typing import Tuple, Union

from numba import njit
import numpy as np

from board import get_opponent, LAZY_PERMIT, PRECIESE_PERMIT, BoardData, BoardTemp, undo_board_numba, \
    DeltaBoardArray, apply_delta_board_numba, get_legal_moves_mask_lazy_numba, is_legal_board_move_numba, \
    make_board_move_numba, get_network_input_pytorch_numba, fast_territories_numba, build_board_data, build_board_temp, \
    i32_scalar, clear_board_numba, copyto_numba, get_territories, board_with_permissions_as_text
from config import possible_moves_total, MAX_KOMI, EMPTY, BLACK, WHITE, symbols, MAX_MOVES_PER_GAME, DRAW_AT_MAX_TURNS, \
    BOARD_SIZE, board_size_sqr, pass_code, swap_code, MAX_HISTORY
from move import encode_move, decode_move

GameData = namedtuple("GameData", [
    "move_history",
    "current_move",
    "game_over",
    "has_mask",
    "legal_mask",


    "komi_norm",
    "max_moves",

    "board",
    "board_temp"
])

DeltaGameArray = namedtuple(
    "DeltaGameArray", [
        "next_mask",
        "move"
    ]
)


@njit()
def get_territories_from_game(game: GameData, out: np.ndarray):
    get_territories(game.board.current_state, game.board_temp.temp_visited, out, game.board_temp.temp_positions_queue)



@njit
def get_consecutive_passes_numba(game: GameData):

    move = game.current_move[0]

    passes = 0

    while move > 0 and game.move_history[move - 1] == pass_code:
        passes += 1
        move -= 1
    return passes

@njit()
def is_interrupted_numba(game: GameData):
    return game.game_over[0] and (game.current_move[0] < game.max_moves[0]) and (get_consecutive_passes_numba(game) < 2)

@njit()
def undo_game_numba(game: GameData):

    if game.current_move[0] <= 0:
        raise Exception("Trying to undo empty game!")

    current_move = game.current_move[0]
    last_pass = current_move > 0 and game.move_history[current_move - 1] == pass_code
    last_swap = current_move == 2 and game.move_history[current_move - 1] == swap_code

    if last_swap:
        game.komi_norm[0] = -game.komi_norm[0]

    if not last_pass:
        undo_board_numba(game.board)

    game.has_mask[0] = False

    game.game_over[0] = False

    game.current_move[0] -= 1

@njit()
def apply_delta_game_numba(game: GameData, delta_boards: DeltaBoardArray, deltas: DeltaGameArray, ptr: int):


    apply_delta_board_numba(delta_boards, ptr, game.board)

    copyto_numba(game.legal_mask, deltas.next_mask[ptr])
    game.has_mask[0] = True

    move = deltas.move[ptr]
    game.move_history[game.current_move[0]] = move

    if move == swap_code:
        game.komi_norm[0] = -game.komi_norm[0]

    game.current_move[0] = game.current_move[0] + 1

    if get_consecutive_passes_numba(game) >= 2:
        game.game_over[0] = True
    if game.current_move[0] >= game.max_moves[0]:
        game.game_over[0] = True


@njit()
def update_lazy_legal_mask(game: GameData):
    if not game.has_mask[0]:
        get_legal_moves_mask_lazy_numba(game.board, game.legal_mask)
        # Пас всегда легален
        game.legal_mask[pass_code] = PRECIESE_PERMIT
        # Свап легален на первый ход белых
        game.legal_mask[swap_code] = PRECIESE_PERMIT if (game.komi_norm == 0.0 and game.current_move[0] == 1) else 0

        game.has_mask[0] = True

@njit()
def is_legal_game_move_numba(game: GameData, encoded_move : int, deltas: DeltaGameArray, delta_ptr: int) -> bool:
    """
    Проверить легальность хода с учетом правил игры.

    Args:
        move: Объект хода

    Returns:
        True если ход легален, False иначе
    """

    # Проверка 2: По маске пробить

    update_lazy_legal_mask(game)

    if game.legal_mask[encoded_move] == 0:
        return False
    elif game.legal_mask[encoded_move] == PRECIESE_PERMIT:
        return True

    x, y, gol, _, _ = decode_move(encoded_move)

    idx_no_life = x * BOARD_SIZE + y
    idx_w_life = idx_no_life + board_size_sqr

    if game.legal_mask[idx_no_life] == 0:
        return False

    #Ленивое разрешение - считаем точно
    placement_permit, gol_permit = is_legal_board_move_numba(game.board, game.board_temp, get_current_player_numba(game), x, y, gol)

    # Это кароче такая хитрая хуита, чтобы меньше счиатть
    # Ходу с ГОЛ обязательно нужна легальность позиции - которая сама является ходом
    # Так можно с одной проверки прокликать 2 позиции в маске
    if placement_permit != PRECIESE_PERMIT:
        game.legal_mask[idx_no_life] = 0
        game.legal_mask[idx_w_life] = 0

        deltas.next_mask[delta_ptr, idx_no_life] = 0
        deltas.next_mask[delta_ptr, idx_w_life] = 0
    else:
        game.legal_mask[idx_no_life] = PRECIESE_PERMIT

        deltas.next_mask[delta_ptr, idx_no_life] = PRECIESE_PERMIT

    if gol_permit == 0:
        game.legal_mask[idx_w_life] = 0

        deltas.next_mask[delta_ptr, idx_w_life] = 0
    elif gol_permit == PRECIESE_PERMIT:
        game.legal_mask[idx_w_life] = PRECIESE_PERMIT

        deltas.next_mask[delta_ptr, idx_no_life] = PRECIESE_PERMIT

    result = gol_permit if gol == 1 else placement_permit

    return result > 0

@njit()
def get_current_player_numba(game: GameData):
    return game.current_move[0] % 2 + 1

@njit()
def try_make_game_move_numba(game: GameData, board_deltas: DeltaBoardArray, game_deltas: DeltaGameArray, ptr: int, encoded_move: int, validate = True) -> bool:

    if validate and not is_legal_game_move_numba(game, encoded_move, game_deltas, ptr):
        return False



    make_board_move_numba(board_deltas, ptr, game.board, game.board_temp, encoded_move, get_current_player_numba(game))

    if encoded_move == swap_code:
        game.komi_norm[0] = -game.komi_norm[0]

    game.move_history[game.current_move[0]] = encoded_move

    game.current_move[0] += 1

    if get_consecutive_passes_numba(game) >= 2:
        game.game_over[0] = True

    if game.current_move[0] >= game.max_moves[0]:
        game.game_over[0] = True

    game.has_mask[0] = False
    update_lazy_legal_mask(game)

    copyto_numba(game_deltas.next_mask[ptr], game.legal_mask)
    game_deltas.move[ptr] = encoded_move

    return True

@njit()
def get_komi_for_current_player_numba(game:GameData):
    if get_current_player_numba(game) == WHITE:
        return game.komi_norm[0]
    else:
        return -game.komi_norm[0]

@njit()
def get_network_input_from_game(game: GameData) -> np.ndarray:
    """
    Получить входной тензор для PyTorch (17×19×19).

    Returns:
        Тензор в формате (C, H, W)
    """
    return get_network_input_pytorch_numba(game.board, game.board_temp, get_current_player_numba(game), get_komi_for_current_player_numba(game))

@njit()
def get_winner_and_margin_numba(game: GameData) -> Tuple[int, int]:


    black, white = fast_territories_numba(game.board, game.board_temp)

    komi = MAX_KOMI * game.komi_norm[0]

    if (DRAW_AT_MAX_TURNS and (game.current_move[0] >= game.max_moves[0])) or (black == (white + komi)):
        winner = EMPTY
        margin = 0
    elif black > white + komi:
        winner = BLACK
        margin = black - white
    else:
        winner = WHITE
        margin = white - black

    return winner, margin

def get_score_from_game(game: GameData) -> dict:
    """
    Получить подсчет очков с учетом коми (китайские правила).

    Returns:
        Словарь с результатами подсчета:
        {
            'black': float - итоговые очки черных (с коми),
            'white': float - итоговые очки белых (с коми),
            'black_base': int - очки черных без коми,
            'white_base': int - очки белых без коми,
            'black_komi': float - коми черных,
            'white_komi': float - коми белых,
            'winner': int - цвет победителя (BLACK или WHITE) или None при ничьей,
            'margin': float - преимущество победителя,
            ... (остальные поля из board.calculate_score())
        }
    """
    # Получаем базовый подсчет с доски (без коми)
    np.equal(game.board.current_state, BLACK, out=game.board_temp.temp_visited)
    black_stones = np.count_nonzero(game.board_temp.temp_visited)

    np.equal(game.board.current_state, WHITE, out=game.board_temp.temp_visited)
    white_stones = np.count_nonzero(game.board_temp.temp_visited)

    black_territory, white_territory = fast_territories_numba(game.board, game.board_temp)

    neutral_territory = board_size_sqr - black_territory - white_territory

    # Добавляем коми

    komi = float(MAX_KOMI) * game.komi_norm[0]

    komi_black = -komi if komi < 0 else 0
    komi_white = komi if komi > 0 else 0

    black_total = black_territory +  komi_black
    white_total = white_territory + komi_white


    winner, margin_no_komi = get_winner_and_margin_numba(game)


    #print(f"Winner is {winner} where b{black_territory} w{white_territory} komi{game.komi_norm[0]}")

    margin = margin_no_komi + komi_black + komi_white

    result = {
        'neutral': neutral_territory,
        'black_stones': black_stones,
        'white_stones': white_stones,
        'black_territory': black_territory - black_stones,
        'white_territory': white_territory - white_stones,

        'black_base': black_territory,
        'white_base': white_territory,

        'black_komi': komi_black,
        'white_komi': komi_white,
        'max_komi': MAX_KOMI,

        'black': black_total,
        'white': white_total,

        'winner': winner,
        'margin': margin,
        'margin_no_komi' : margin_no_komi
    }

    return result

def build_delta_game_array(max_nodes: int):
    return DeltaGameArray(
        next_mask=np.zeros((max_nodes, possible_moves_total), dtype=np.int8),
        move=np.zeros(max_nodes, dtype=np.int32)
    )

def get_board_text(game: GameData) -> str:
    return board_with_permissions_as_text(game.board.current_state, game.legal_mask)

def build_game_data(komi_norm: float):
   return GameData(
       board=build_board_data(),
       board_temp=build_board_temp(),

       move_history = np.zeros(MAX_MOVES_PER_GAME, dtype=np.int32),
       current_move = np.zeros(1, dtype=np.int32),
       game_over = np.zeros(1, dtype=np.bool_),
       has_mask=np.zeros(1, dtype=np.bool_),
       legal_mask=np.zeros(possible_moves_total, dtype=np.int8),

       komi_norm = np.array([komi_norm], dtype=np.float32),

       max_moves = np.array([MAX_MOVES_PER_GAME], dtype=np.int32)
   )

@njit()
def clear_game_numba(game: GameData):
    clear_board_numba(game.board)
    game.move_history.fill(0)
    game.current_move[0] = 0
    game.game_over[0] = False
    game.has_mask[0] = False

import json
class NumpyEncoder(json.JSONEncoder):
    """
    Специальный JSON-энкодер для объектов NumPy.
    Он проверяет, является ли объект массивом NumPy, и если да,
    преобразует его в список. В противном случае, использует
    стандартный энкодер.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

