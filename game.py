from typing import Tuple, Union

from fontTools.ttLib.tables.V_O_R_G_ import VOriginRecord
from numba import njit

import numpy as np
from sympy.solvers.polysys import factor_system_bool
from torchvision.transforms.v2.functional import elastic_mask, resize

from board import Board, get_opponent, DeltaBoard, LAZY_PERMIT, PRECIESE_PERMIT
from config import possible_moves_total, MAX_KOMI, EMPTY, BLACK, WHITE, symbols, MAX_MOVES_PER_GAME, DRAW_AT_MAX_TURNS, \
    BOARD_SIZE, board_size_sqr, pass_code, swap_code
from move import encode_move, decode_move


class DeltaGame:
    def __init__(self):
        self.next_mask =  np.zeros(possible_moves_total, dtype=np.int8)
        self.delta_board = DeltaBoard()

        self.is_pass = False
        self.is_swap = False

        self.valid = False


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

    def __init__(self, komi: float = 0, max_moves : int = MAX_MOVES_PER_GAME):
        """
        Инициализация новой игры.

        Args:
            komi: Словарь компенсации {Board.BLACK: 0.0, Board.WHITE: 6.5}.
                  По умолчанию белые получают 6.5 очков компенсации.
        """

        self.winner = EMPTY

        self.board = Board()


        self.legal_mask = np.zeros(possible_moves_total, dtype=np.int8)
        self.has_mask = False

        self.current_player = BLACK  # Черные ходят первыми

        self.game_over = False

        self.pass_history = np.zeros(MAX_MOVES_PER_GAME, dtype=np.bool_)
        self.had_swap = False

        # Коми в пользу БЕЛЫХ
        self.komi = komi
        self.komi_norm = komi / MAX_KOMI

        self.max_moves = max_moves
        self.current_move = 0

    def interrupted(self) -> bool:
        return self.game_over and (self.current_move < MAX_MOVES_PER_GAME) and (self.get_consequtive_passes() < 2)

    def get_consequtive_passes(self):
        move = self.current_move

        passes = 0

        while move > 0 and self.pass_history[move - 1]:
            passes += 1
            move -= 1
        return passes

    def undo(self):

        if self.current_move <= 0:
            raise Exception("Trying to undo empty game!")

        last_pass = self.current_move > 0 and self.pass_history[self.current_move - 1]

        if self.current_move - 1 == 1 and self.had_swap:
            self.set_komi(-self.komi)
            self.had_swap = False

        if not last_pass:
            self.board.undo()

        self.current_player = get_opponent(self.current_player)
        self.has_mask = False

        self.game_over = False

        self.current_move -= 1


    def apply_delta(self, delta_game: DeltaGame):

        if not delta_game.valid:
            return

        self.board.apply_delta(delta_game.delta_board)

        np.copyto(self.legal_mask, delta_game.next_mask)
        self.has_mask = True

        self.current_player = get_opponent(self.current_player)

        self.pass_history[self.current_move] = delta_game.is_pass

        self.had_swap = self.had_swap or delta_game.is_swap
        if delta_game.is_swap:
            self.set_komi(-self.komi)

        self.current_move += 1

        if self.get_consequtive_passes() >= 2:
            self.game_over = True
        if self.current_move >= self.max_moves:
            self.game_over = True



    def set_komi(self, komi: float):
        self.komi = komi
        self.komi_norm = self.komi / MAX_KOMI

    def clear(self):
        self.winner = EMPTY

        self.board.clear()
        self.has_mask = False

        self.current_player = BLACK

        self.pass_history.fill(0)

        self.had_swap = False
        self.game_over = False

        self.current_move = 0

    def update_legal_moves_mask(self):
        if not self.has_mask:

            self.board.get_legal_moves_mask_lazy(self.legal_mask)
            # Пас всегда легален
            self.legal_mask[board_size_sqr * 2] = PRECIESE_PERMIT
            # Свап легален на первый ход белых
            self.legal_mask[board_size_sqr * 2 + 1] = PRECIESE_PERMIT if self.current_move == 1 else 0

            self.has_mask = True

    def get_legal_moves_mask_lazy(self, out: np.ndarray):
        self.update_legal_moves_mask()

        for i in range(possible_moves_total):
            out[i] = 1.0 if self.legal_mask[i] > 0 else 0.0

    def get_legal_moves_mask_preciese(self, out : np.ndarray):

        self.board.get_legal_moves_mask(self.current_player, out)

        # Пас всегда легален
        out[board_size_sqr * 2] = 1.0
        # Свап легален на первый ход белых
        out[board_size_sqr * 2 + 1] = 1.0 if self.current_move == 1 else 0.0


    def is_valid_move(self, encoded_move : int, out_mask : np.ndarray) -> bool:
        """
        Проверить легальность хода с учетом правил игры.

        Args:
            move: Объект хода

        Returns:
            True если ход легален, False иначе
        """

        # Проверка 2: По маске пробить

        self.update_legal_moves_mask()

        if self.legal_mask[encoded_move] == 0:
            return False

        if self.legal_mask[encoded_move] == PRECIESE_PERMIT:
            return True

        x, y, gol, _, _ = decode_move(encoded_move)

        idx_no_life = x * BOARD_SIZE + y
        idx_w_life = idx_no_life + board_size_sqr

        if self.legal_mask[idx_no_life] == 0:
            return False

        #Ленивое разрешение - считаем точно
        placement_permit, gol_permit = self.board.is_move_legal(self.current_player, x,y,gol)

        #Пласемент не должен быть нанкой
        if placement_permit is None:
            raise Exception("Legality check resulted in None in placement permission")

        # Это кароче такая хитрая хуита, чтобы меньше счиатть
        # Ходу с ГОЛ обязательно нужна легальность позиции - которая сама является ходом
        # Так можно с одной проверки прокликать 2 позиции в маске
        if not placement_permit:
            self.legal_mask[idx_no_life] = 0
            self.legal_mask[idx_w_life] = 0
            out_mask[idx_no_life] = 0
            out_mask[idx_w_life] = 0
        else:
            self.legal_mask[idx_no_life] = PRECIESE_PERMIT
            out_mask[idx_no_life] = PRECIESE_PERMIT

        if gol_permit is not None:
            if not gol_permit:
                self.legal_mask[idx_w_life] = 0
                out_mask[idx_w_life] = 0
            else:
                self.legal_mask[idx_w_life] = PRECIESE_PERMIT

            out_mask[idx_w_life] = self.legal_mask[idx_w_life]

        result = gol_permit if gol == 1 else placement_permit

        return result

    def make_move(self, encoded_move : int, out: DeltaGame, validate = True) -> bool:

        if validate and not self.is_valid_move(encoded_move, self.legal_mask):
            out.valid = False
            return False

        self.board.make_move(encoded_move, self.current_player, out.delta_board)

        is_pass = encoded_move == pass_code
        is_swap = encoded_move == swap_code

        out.is_pass = is_pass
        out.is_swap = is_swap

        if is_swap:
            self.set_komi(-self.komi)
            self.had_swap = True

        self.pass_history[self.current_move] = is_pass

        self.current_move += 1

        if self.get_consequtive_passes() >= 2:
            self.game_over = True

        if self.current_move >= self.max_moves:
            self.game_over = True

        self.current_player = get_opponent(self.current_player)

        self.has_mask = False
        self.update_legal_moves_mask()
        np.copyto(out.next_mask, self.legal_mask)

        out.valid = True

        return True

    def get_komi_for_current_player(self):
        if self.current_player == WHITE:
            return self.komi_norm
        else:
            return -self.komi_norm

    def get_network_input_pytorch(self) -> np.ndarray:
        """
        Получить входной тензор для PyTorch (17×19×19).

        Returns:
            Тензор в формате (C, H, W)
        """
        return self.board.get_network_input_pytorch(self.current_player, self.get_komi_for_current_player())

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

        komi_black = -self.komi if self.komi < 0 else 0
        komi_white = self.komi if self.komi > 0 else 0

        black_total = base_score['black'] +  komi_black
        white_total = base_score['white'] + komi_white

        winner, margin_no_komi = self.get_winner_and_margin_fast()
        margin = margin_no_komi + komi_black + komi_white

        # Объединяем результаты
        result = base_score.copy()
        result.update({
            'black_base': base_score['black'],
            'white_base': base_score['white'],

            'black_komi': komi_black,
            'white_komi': komi_white,
            'max_komi': MAX_KOMI,

            'black': black_total,
            'white': white_total,

            'winner': winner,
            'margin': margin,
            'margin_no_komi' : margin_no_komi
        })

        return result

    def get_header_text(self) -> str:

        player_str = f"{symbols[self.current_player]} " + "ЧЕРНЫЕ" if self.current_player == BLACK else "БЕЛЫЕ"
        current_player = f"Текущий игрок: {player_str}\n"

        return current_player

    def get_winner_text(self) -> str:
        if self.winner == EMPTY:
            return f"Победитель не определен"

        player_str = f"{symbols[self.winner]} " + "ЧЕРНЫЕ" if self.winner == BLACK else "БЕЛЫЕ"
        winner = f"Победитель: {player_str}\n"

        return winner

    def get_score_text(self) -> str:
        """Сформировать строку с результатами подсчета очков с учетом коми."""
        score = self.get_score()

        # Обновляем состояние победителя и текст результата
        if score['winner'] == BLACK:
            self.winner = BLACK
            result_text = f"⚫ ЧЕРНЫЕ ВЫИГРАЛИ с преимуществом {score['margin']:.1f} очков"
        elif score['winner'] == WHITE:
            self.winner = WHITE
            result_text = f"🔴 БЕЛЫЕ ВЫИГРАЛИ с преимуществом {score['margin']:.1f} очков"
        else:
            result_text = "НИЧЬЯ"

        # Собираем все строки в список для удобного объединения
        lines = [
            "=" * 60,
            "⚫ ЧЕРНЫЕ:",
            f"   Камни на доске: {score['black_stones']}",
            f"   Территория:     {score['black_territory']}",
            f"   Коми:           {score['black_komi']}",
            f"   ИТОГО:          {score['black']}",
            "",
            "🔴 БЕЛЫЕ:",
            f"   Камни на доске: {score['white_stones']}",
            f"   Территория:     {score['white_territory']}",
            f"   Коми:           {score['white_komi']}",
            f"   ИТОГО:          {score['white']}",
            "",
            f"⚪ Нейтральные пункты: {score['neutral']}",
            "=" * 60,
            result_text,
            "=" * 60
        ]

        return "\n".join(lines)

    def get_board_text(self, mask:np.ndarray):
        return self.board.board_with_permissions_as_text(mask)

    def print_score(self):
        """Вывести результаты подсчета очков с учетом коми."""
        print(self.get_score_text())

    def get_current_leader(self):
        return self.get_winner()

    def get_winner_and_margin_fast(self) -> Tuple[int,int]:
        black, white = self.board.fast_territories()

        if (DRAW_AT_MAX_TURNS and (self.current_move >= MAX_MOVES_PER_GAME)) or (black == (white + self.komi)):
            self.winner = EMPTY
            margin = 0
        elif black > white + self.komi:
            self.winner = BLACK
            margin = black - white
        else:
            self.winner  = WHITE
            margin = white - black

        return self.winner, margin


    def get_winner(self):
        self.get_winner_and_margin_fast()
        return self.winner

    def clone(self) -> 'Game':
        """
        Создать полную копию игры (для MCTS симуляций).

        Returns:
            Копия объекта Game
        """
        new_game = Game(komi=self.komi)

        new_game.board = self.board.clone()

        new_game.current_player = self.current_player

        np.copyto(new_game.pass_history, self.pass_history)
        new_game.had_swap = self.had_swap

        new_game.current_move += self.current_move

        new_game.game_over = self.game_over
        new_game.winner = self.winner

        return new_game

    def copy(self, target:'Game'):

        self.board.copy(target.board)

        target.current_player = self.current_player

        target.current_move = self.current_move

        target.game_over = self.game_over
        target.winner = self.winner

        target.has_mask = self.has_mask
        np.copyto(target.legal_mask, self.legal_mask)

        target.komi = self.komi
        target.komi_norm = self.komi_norm

        np.copyto(target.pass_history, self.pass_history)
        target.had_swap = self.had_swap

    def __repr__(self) -> str:
        """Строковое представление игры."""
        player_str = "BLACK" if self.current_player == BLACK else "WHITE"
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

