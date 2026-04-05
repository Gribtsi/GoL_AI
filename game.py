import time

import numpy as np
from collections import deque
from typing import Tuple, Optional, List, Set
from copy import deepcopy

from numpy.random import random

from board import Board, Move, WHITE, BLACK, BOARD_SIZE, EMPTY, position_permissions_numba

from numba import njit, prange




# Предполагается, что у вас уже есть njit-версии этих функций
# find_group_and_liberties, check_captures, remove_captured_stones, apply_game_of_life

possible_moves_total = BOARD_SIZE * BOARD_SIZE * 2 + 1
board_size_sqr = BOARD_SIZE * BOARD_SIZE

def decode_action(action_idx: int, color: int) -> Move:
    """
    Декодировать индекс действия в объект Move.

    Args:
        action_idx: Индекс от 0 до 722
        color: Цвет текущего игрока

    Returns:
        Объект Move
    """
    if action_idx == board_size_sqr * 2:
        # Пас
        return Move(None, 0, color, True)
    else:
        # Обычный ход
        position_idx = action_idx % board_size_sqr
        life_cycle = action_idx // board_size_sqr

        x = position_idx // BOARD_SIZE
        y = position_idx % BOARD_SIZE

        return Move((x, y), life_cycle, color)


def get_opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK

@njit(parallel=True)
def get_legal_moves_mask_numba(
        current_state: np.ndarray,
        history_stack: np.ndarray,
        current_player: int
) -> np.ndarray:
    mask = np.zeros(BOARD_SIZE * BOARD_SIZE * 2 + 1, dtype=np.float32)

    # prange распараллеливает внешний цикл
    for x in prange(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            # Проверка хода без цикла жизни

            placement, gol = position_permissions_numba(current_state, history_stack, current_player, x, y)

            if placement:
                idx_no_life = x * BOARD_SIZE + y
                mask[idx_no_life] = 1.0

            if gol:
                idx_with_life = BOARD_SIZE * BOARD_SIZE + (x * BOARD_SIZE + y)
                mask[idx_with_life] = 0.0


    # Пас всегда легален
    mask[BOARD_SIZE * BOARD_SIZE * 2] = 1.0

    return mask

def get_comi_for_white():
    rng = np.random.uniform(0,1)
    if rng > 0.7:
        return 1.5
    else:
        return 0.5

class Game:
    """
    Класс управления игрой Go + Game of Life.

    Отвечает за:
    - Отслеживание текущего игрока
    - Валидацию ходов
    - Логирование истории ходов
    - Определение окончания игры
    - Получение входных тензоров для нейросети
    - Подсчет очков с учетом коми
    """

    def __init__(self, komi: dict = None, board: Board = None, max_moves : int = 100):
        """
        Инициализация новой игры.

        Args:
            komi: Словарь компенсации {Board.BLACK: 0.0, Board.WHITE: 6.5}.
                  По умолчанию белые получают 6.5 очков компенсации.
        """

        self.history = []

        self.winner = None

        self.board = Board()
        if board is not None:
            self.board.current_state = board.current_state.copy()
            self.board.history_stack = deque(board.history_stack, maxlen=8)

        self.current_player = Board.BLACK  # Черные ходят первыми

        self.consecutive_passes = 0  # Счетчик последовательных пасов
        self.game_over = False

        # Коми (компенсация для белых за то, что черные ходят первыми)
        if komi is None:
            self.komi = {
                Board.BLACK: 0.0,
                Board.WHITE: 0.5
            }
        else:
            self.komi = komi

        self.max_moves = max_moves
        self.current_move = 0

    def get_opponent(self, color: int) -> int:
        """Получить цвет оппонента."""
        return Board.WHITE if color == Board.BLACK else Board.BLACK

    def get_legal_moves_mask_simple(self) -> np.ndarray:
        """
        Быстрая генерация маски легальных ходов с использованием Numba.
        """

        history_np = self.board.get_history_np()

        # 2. Вызываем быструю Numba-функцию
        mask = get_legal_moves_mask_numba(
            self.board.current_state,
            history_np,
            self.current_player
        )
        return mask

    def is_valid_move(self, move: Move) -> Tuple[bool, 'Game']:
        """
        Проверить легальность хода с учетом правил игры.

        Args:
            move: Объект хода

        Returns:
            True если ход легален, False иначе
        """
        # Проверка 1: Цвет хода должен совпадать с текущим игроком

        if move.color != self.current_player:
            return (False, None)

        # Проверка 2: Цикл жизни без установки камня запрещен в нормальной игре
        if not move.is_pass() and move.position is None and move.life_cycles > 0:
            return (False, None)

        future_game = self.clone()

        move_result = future_game.board.make_move(move)
        success = move_result[0]
        if not success:
            return (False, None)


        if move.is_pass():
            future_game.consecutive_passes += 1
        else:
            future_game.consecutive_passes = 0

        if future_game.consecutive_passes >= 2:
            future_game.game_over = True

        future_game.current_player = self.get_opponent(self.current_player)

        # Пытаемся применить ход на тестовой доске
        return (True, future_game)

    def make_move(self, move: Move, validate = True) -> bool:
        """
        Применить ход в игре.

        Args:
            move: Объект хода

        Returns:
            True если ход успешно применен, False если отклонен
        """
        if self.game_over:
            return False

        # Валидация хода
        if validate and not self.is_valid_move(move):
            return False

        # Применяем ход на доске
        move_result = self.board.make_move(move)
        success = move_result[0]

        if not success:
            return False

        self.current_move += 1

        # Обновляем счетчик пасов
        if move.is_pass():
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        # Проверяем окончание игры (два паса подряд)
        if self.consecutive_passes >= 2 or self.current_move > self.max_moves:
            self.game_over = True


        # Переключаем игрока
        self.current_player = self.get_opponent(self.current_player)

        return True

    def get_network_input_pytorch(self, states_to_save) -> np.ndarray:
        """
        Получить входной тензор для PyTorch (17×19×19).

        Returns:
            Тензор в формате (C, H, W)
        """
        return self.board.get_network_input_pytorch(self.current_player, states_to_save)

    def get_score(self) -> dict:
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
        base_score = self.board.get_score()

        # Добавляем коми
        black_total = base_score['black'] + self.komi[Board.BLACK]
        white_total = base_score['white'] + self.komi[Board.WHITE]

        # Определяем победителя
        if black_total > white_total:
            winner = Board.BLACK
            margin = black_total - white_total
            margin_no_komi = base_score['black'] - base_score['white']
        elif white_total > black_total:
            winner = Board.WHITE
            margin = white_total - black_total
            margin_no_komi = base_score['white'] - base_score['black']
        else:
            winner = None
            margin = 0.0
            margin_no_komi = 0

        # Объединяем результаты
        result = base_score.copy()
        result.update({
            'black_base': base_score['black'],
            'white_base': base_score['white'],
            'black_komi': self.komi[Board.BLACK],
            'white_komi': self.komi[Board.WHITE],
            'black': black_total,
            'white': white_total,
            'winner': winner,
            'margin': margin,
            'margin_no_komi' : margin_no_komi
        })

        return result

    def print_board(self, permissions = True):
        """Вывести текущее состояние доски."""
        if permissions:
            print(self.board.display_board_with_permissions(self.current_player))
        else:
            print(self.board.display_board())

        player_str = "⚫ ЧЕРНЫЕ" if self.current_player == Board.BLACK else "🔴 БЕЛЫЕ"
        print(f"\nТекущий игрок: {player_str}")
        print(f"Последовательных пасов: {self.consecutive_passes}")
        if self.game_over:
            print("*** ИГРА ОКОНЧЕНА ***")

    def print_score(self):
        """Вывести результаты подсчета очков с учетом коми."""
        score = self.get_score()

        print("=" * 60)
        print("ПОДСЧЕТ ОЧКОВ (Китайские правила)")
        print("=" * 60)
        print(f"⚫ ЧЕРНЫЕ:")
        print(f"   Камни на доске: {score['black_stones']}")
        print(f"   Территория:     {score['black_territory']}")
        print(f"   Коми:           {score['black_komi']}")
        print(f"   ИТОГО:          {score['black']}")
        print()
        print(f"🔴 БЕЛЫЕ:")
        print(f"   Камни на доске: {score['white_stones']}")
        print(f"   Территория:     {score['white_territory']}")
        print(f"   Коми:           {score['white_komi']}")
        print(f"   ИТОГО:          {score['white']}")
        print()
        print(f"⚪ Нейтральные пункты: {score['neutral']}")
        print("=" * 60)

        # Определяем победителя
        if score['winner'] == Board.BLACK:
            print(f"⚫ ЧЕРНЫЕ ВЫИГРАЛИ с преимуществом {score['margin']:.1f} очков")
            self.winner = BLACK
        elif score['winner'] == Board.WHITE:
            print(f"🔴 БЕЛЫЕ ВЫИГРАЛИ с преимуществом {score['margin']:.1f} очков")
            self.winner = WHITE
        else:
            print("НИЧЬЯ")
        print("=" * 60)

    def get_current_leader(self):
        score = self.get_score()
        return score['winner']

    def get_winner(self):
        return self.winner

    def clone(self) -> 'Game':
        """
        Создать полную копию игры (для MCTS симуляций).

        Returns:
            Копия объекта Game
        """
        new_game = Game(komi=self.komi)

        new_game.board.current_state = self.board.current_state.copy()
        new_game.board.history_stack = deque(self.board.history_stack)

        new_game.current_player = self.current_player
        new_game.consecutive_passes = self.consecutive_passes
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def __repr__(self) -> str:
        """Строковое представление игры."""
        player_str = "BLACK" if self.current_player == Board.BLACK else "WHITE"
        status = "FINISHED" if self.game_over else "IN PROGRESS"
        return f"Game(player={player_str}, status={status}, komi={self.komi})"


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
