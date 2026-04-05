import numpy as np
from collections import deque
import json
from typing import Tuple, Optional, List, Set, Dict, Any
from copy import deepcopy

from matplotlib.style.core import available
from numba import njit
from scipy.linalg import hilbert


class Move:
    """Класс для представления хода в игре Go + Game of Life."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, position: Optional[Tuple[int, int]], life_cycles: int, color: int, is_pass: bool = False):
        """
        Args:
            position: Координаты (x, y) для установки камня, None для паса
            life_cycles: Количество циклов жизни после установки (0 или 1)
            color: Цвет игрока (1 - черные, 2 - белые)
        """
        self.position = position
        self.life_cycles = life_cycles
        self.color = color
        self.__is_pass = is_pass

    def is_pass(self) -> bool:
        """Проверка, является ли ход пасом."""
        return self.__is_pass

    def __repr__(self) -> str:
        if self.is_pass():
            return f"Move(PASS, color={self.color})"
        return f"Move(pos={self.position}, life_cycles={self.life_cycles}, color={self.color})"

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует объект в словарь для JSON-сериализации."""
        return {
            "position": self.position,
            "life_cycles": self.life_cycles,
            "color": self.color,
            "is_pass": self.__is_pass
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Move":
        """Создает объект из словаря (после JSON-десериализации)."""
        # JSON конвертирует tuple в list, поэтому возвращаем формат обратно
        pos = data.get("position")
        position = tuple(pos) if pos is not None else None

        return cls(
            position=position,
            life_cycles=data.get("life_cycles", 0),
            color=data.get("color", cls.EMPTY),
            is_pass=data.get("is_pass", False)
        )


import numpy as np
from numba import njit
from typing import Tuple, Optional

# --- Константы и вспомогательные функции, также помеченные @njit ---

BOARD_SIZE = 11
EMPTY = 0
BLACK = 1
WHITE = 2


@njit
def is_valid_position(x: int, y: int) -> bool:
    """Проверяет, находится ли позиция в пределах доски."""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

@njit
def is_valid_move_numba(
        current_state: np.ndarray,
        history_stack: np.ndarray,  # История как (N, 19, 19) numpy-массив
        current_player: int,
        x: int, y: int,
        life_cycles: int
) -> bool:
    # --- Быстрые проверки ---
    # Ход за пределы доски
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False
    # Ход в занятую клетку
    if current_state[x, y] != EMPTY:
        return False

    # --- Симуляция хода (эквивалент make_move(commit=False)) ---

    # ШАГ 1: Установка камня и проверка захватов
    future_state_1 = current_state.copy()
    future_state_1[x, y] = current_player

    # Проверяем и снимаем камни оппонента
    marked_opponent = check_captures(future_state_1)
    remove_captured_stones(future_state_1, marked_opponent, exclude_color=current_player)

    # Проверка на самоубийственный ход
    _, liberties = find_group_and_liberties(x, y, future_state_1)
    if liberties == 0:
        return False

    # Проверка правила "ко"
    for i in range(history_stack.shape[0]):
        if np.array_equal(future_state_1, history_stack[i]):
            return False

    # --- ШАГ 2: Симуляция цикла жизни (если нужно) ---
    final_state = future_state_1
    if life_cycles == 1:
        future_state_2 = apply_game_of_life(future_state_1, current_player)

        # Снимаем камни оппонента после цикла жизни
        marked_after_life_opp = check_captures(future_state_2)
        remove_captured_stones(future_state_2, marked_after_life_opp, exclude_color=current_player)

        # Проверяем и снимаем любые "самоубийственные" группы после цикла жизни
        marked_after_life_any = check_captures(future_state_2)
        remove_captured_stones(future_state_2, marked_after_life_any, exclude_color=-1)  # -1 значит никого не исключать

        # Финальная проверка "ко"
        for i in range(history_stack.shape[0]):
            if np.array_equal(future_state_2, history_stack[i]):
                return False

        final_state = future_state_2

    # Если все проверки пройдены, ход легален
    return True


@njit(fastmath=True, cache=True)
def check_superko_numba(current_state: np.ndarray, history_stack: np.ndarray) -> bool:
    """
    Проверяет правило ситуационного Суперко.

    Args:
        current_state: Текущее состояние доски (H, W)
        history_stack: Стек истории состояний (N, H, W)

    Returns:
        True если ход легален (нет повторения), False если нарушает правило.
    """
    n_history = history_stack.shape[0]

    # Если история пуста или слишком коротка, проверять нечего
    if n_history == 0:
        return True

    rows = current_state.shape[0]
    cols = current_state.shape[1]

    # Итерируемся с конца стека:
    # range(start, stop, step)
    # start = n_history - 2 (предпоследний элемент)
    # stop = -1 (до 0 включительно)
    # step = -2 (ситуационное суперко: проверяем только ходы текущего игрока)
    for i in range(n_history - 4, -1, -4):

        # Ручной цикл сравнения массивов быстрее np.array_equal в Numba,
        # так как позволяет сделать break при первом же несовпадении
        is_equal = True
        for r in range(rows):
            for c in range(cols):
                if history_stack[i, r, c] != current_state[r, c]:
                    is_equal = False
                    break  # Прерываем внутренний цикл (по столбцам)
            if not is_equal:
                break  # Прерываем внешний цикл (по строкам) - переходим к следующему состоянию в истории

        # Если после проверки is_equal осталось True, значит мы нашли полное совпадение
        if is_equal:
            return False

    return True


@njit(fastmath=True, cache=True)
def check_megako_numba(current_state: np.ndarray,
                       history_stack: np.ndarray,
                       player_color: int) -> bool:
    """
    Проверяет правило "Мега-Ко":
    Запрещено повторять конфигурацию СВОИХ камней, которая уже встречалась
    в прошлом (на своих ходах).

    Args:
        current_state: (H, W) массив
        history_stack: (N, H, W) массив
        player_color: целое число, обозначающее камни текущего игрока (например, 1)

    Returns:
        True если ход легален, False если конфигурация камней игрока повторяется.
    """
    n_history = history_stack.shape[0]
    if n_history == 0:
        return True

    rows = current_state.shape[0]
    cols = current_state.shape[1]

    # Проходим по истории с шагом 2 (только свои предыдущие ходы)
    # range(start, stop, step) -> от предпоследнего до 0
    for i in range(n_history - 2, -1, -4):

        # Предполагаем, что конфигурация камней совпадает (Illegal)
        stones_match = True

        # Сканируем доску
        for r in range(rows):
            for c in range(cols):
                # Нас волнует ТОЛЬКО совпадение наличия камня игрока.
                # Есть ли камень игрока сейчас?
                curr_is_player = (current_state[r, c] == player_color)
                # Был ли камень игрока тогда?
                hist_is_player = (history_stack[i, r, c] == player_color) or (history_stack[i + 1, r, c] == player_color)

                # Если в одной позиции есть камень, а в другой нет (или наоборот)
                # -> конфигурация НЕ совпадает, можно переходить к следующему снимку истории.
                if curr_is_player != hist_is_player:
                    stones_match = False
                    break  # Break inner loop (cols)

            if not stones_match:
                break  # Break outer loop (rows)

        # Если после полного прохода по доске мы не нашли отличий в расположении
        # камней игрока -> это повтор.
        if stones_match:
            return False

    return True

@njit
def position_permissions_numba(
        current_state: np.ndarray,
        history_stack: np.ndarray,  # История как (N, 19, 19) numpy-массив
        current_player: int,
        x: int, y: int
) -> Tuple[bool, bool]:
    # --- Быстрые проверки ---
    # Ход за пределы доски
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return (False, False)
    # Ход в занятую клетку
    if current_state[x, y] != EMPTY:
        return (False, False)

    # --- Симуляция хода (эквивалент make_move(commit=False)) ---

    # ШАГ 1: Установка камня и проверка захватов
    future_state_1 = current_state.copy()
    future_state_1[x, y] = current_player

    should_check_captures = False
    for nx, ny in get_neighbors_4(x, y):
        if current_state[nx,ny] != EMPTY and current_state[nx,ny] != current_player:
            should_check_captures = True
            break

    # Проверяем и снимаем камни оппонента
    if should_check_captures:
        marked_opponent = check_captures(future_state_1)
        remove_captured_stones(future_state_1, marked_opponent, exclude_color=current_player)

    # Проверка на самоубийственный ход
    _, liberties = find_group_and_liberties(x, y, future_state_1)
    if liberties == 0:
        return (False, False)

    # Проверка правила "ко"

    if not check_megako_numba(future_state_1, history_stack, current_player):
        return (False, False)

    # --- ШАГ 2: Симуляция цикла жизни (если нужно) ---

    future_state_2 = apply_game_of_life(future_state_1, current_player)

    # Снимаем камни оппонента после цикла жизни
    marked_after_life_opp = check_captures(future_state_2)
    remove_captured_stones(future_state_2, marked_after_life_opp, exclude_color=current_player)

    # Проверяем и снимаем любые "самоубийственные" группы после цикла жизни
    marked_after_life_any = check_captures(future_state_2)
    remove_captured_stones(future_state_2, marked_after_life_any, exclude_color=-1)  # -1 значит никого не исключать

    # Финальная проверка "ко"
    if not check_megako_numba(future_state_2, history_stack, current_player):
        return (True, False)

    # Если все проверки пройдены, ход легален
    return (True, True)



@njit
def get_neighbors_4(x: int, y: int):
    """Получает 4 соседа (вверх, вниз, влево, вправо)."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if is_valid_position(nx, ny):
            neighbors.append((nx, ny))
    return neighbors


# --- Основные оптимизированные функции ---

@njit
def find_group_and_liberties(x: int, y: int, state: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Находит группу камней и подсчитывает её дыхания (liberties).
    Реализовано с использованием numpy-массивов вместо deque и set для совместимости с Numba.
    """
    color = state[x, y]
    if color == EMPTY:
        # Возвращаем пустой массив координат
        return np.empty((0, 2), dtype=np.int32), 0

    # Ручная реализация очереди на numpy-массивах
    q = np.empty((BOARD_SIZE * BOARD_SIZE, 2), dtype=np.int16)
    q_head, q_tail = 0, 0

    q[q_tail] = np.array([x, y], dtype=np.int16)
    q_tail += 1

    visited_local = np.zeros_like(state, dtype=np.bool_)
    visited_local[x, y] = True

    group_members = []
    liberties = set()

    while q_head < q_tail:
        cx, cy = q[q_head]
        q_head += 1

        group_members.append((cx, cy))

        # Проверяем соседей
        for nx, ny in get_neighbors_4(cx, cy):
            if not visited_local[nx, ny]:
                visited_local[nx, ny] = True
                if state[nx, ny] == EMPTY:
                    liberties.add((nx, ny))
                elif state[nx, ny] == color:
                    q[q_tail] = np.array([nx, ny], dtype=np.int16)
                    q_tail += 1

    # Конвертируем список группы в numpy-массив
    group_array = np.empty((len(group_members), 2), dtype=np.int32)
    for i, (gx, gy) in enumerate(group_members):
        group_array[i, 0] = gx
        group_array[i, 1] = gy

    return group_array, len(liberties)


@njit
def check_captures(state: np.ndarray) -> np.ndarray:
    """
    Проверяет и отмечает группы камней без дыханий для захвата.
    """
    visited = np.zeros_like(state, dtype=np.bool_)
    marked_for_death = np.zeros_like(state, dtype=np.int8)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if visited[x, y] or state[x, y] == EMPTY:
                continue

            group, liberties = find_group_and_liberties(x, y, state)

            # Отмечаем группу как проверенную
            for i in range(group.shape[0]):
                gx, gy = group[i]
                visited[gx, gy] = True

            # Если дыханий нет - отмечаем к смерти
            if liberties == 0:
                stone_color = state[group[0, 0], group[0, 1]]
                for i in range(group.shape[0]):
                    gx, gy = group[i]
                    marked_for_death[gx, gy] = stone_color

    return marked_for_death


@njit
def remove_captured_stones(state: np.ndarray, marked: np.ndarray, exclude_color: int = -1):
    """
    Снимает с доски отмеченные камни. exclude_color используется для защиты своих камней.
    """
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if marked[x, y] != EMPTY and marked[x, y] != exclude_color:
                state[x, y] = EMPTY


@njit
def apply_game_of_life(state: np.ndarray, color: int) -> np.ndarray:
    """
    Применяет один цикл Game of Life для камней указанного цвета.
    """
    new_state = state.copy()

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            # Подсчет соседей 8-связности
            same_color_count = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if is_valid_position(nx, ny) and state[nx, ny] == color:
                        same_color_count += 1

            # Применение правил
            if state[x, y] == color:
                # Правило выживания
                if same_color_count not in (2, 3):
                    new_state[x, y] = EMPTY
            elif state[x, y] == EMPTY:
                # Правило рождения
                if same_color_count == 3:
                    new_state[x, y] = color

    return new_state


class Board:
    """
    Класс доски для игры Go + Game of Life.

    Управляет состоянием игры, валидацией и применением ходов.
    """

    # Константы
    BOARD_SIZE = BOARD_SIZE
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self):
        """Инициализация доски."""
        self.current_state = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.future_state_1 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.future_state_2 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.history_stack = deque()
        for i in range(4):
            empty = np.zeros((Board.BOARD_SIZE, Board.BOARD_SIZE), dtype=np.int8)
            self.history_stack.append(empty)

    def get_opponent(self, color: int) -> int:
        """Получить цвет оппонента."""
        return self.WHITE if color == self.BLACK else self.BLACK

    def is_valid_position(self, x: int, y: int) -> bool:
        """Проверить, находится ли позиция в пределах доски."""
        return 0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE

    def find_group_and_liberties(self, x: int, y: int, state: np.ndarray) -> Tuple[np.ndarray, int]:
        """Обертка для вызова njit-версии find_group_and_liberties."""
        return find_group_and_liberties(x, y, state)

    def check_captures(self, state: np.ndarray) -> np.ndarray:
        """Обертка для вызова njit-версии check_captures."""
        return check_captures(state)

    def remove_captured_stones(self, state: np.ndarray, marked: np.ndarray, exclude_color: Optional[int] = None):
        """Обертка для вызова njit-версии remove_captured_stones."""
        # Numba плохо работает с Optional[int], поэтому передаем -1 как маркер отсутствия значения
        exclude_val = exclude_color if exclude_color is not None else -1
        remove_captured_stones(state, marked, exclude_val)

    def apply_game_of_life(self, state: np.ndarray, color: int) -> np.ndarray:
        """Обертка для вызова njit-версии apply_game_of_life."""
        return apply_game_of_life(state, color)

    def check_ko_rule(self, state: np.ndarray, active_color : int) -> bool:
        history_np = self.get_history_np()
        return check_megako_numba(state, history_np, active_color)

    def get_history_np(self):
        #if len(self.history_stack) > 4:
        #    history_np = np.stack(list(self.history_stack), axis=0)
        #else:
        #    history_np = np.zeros((4, Board.BOARD_SIZE, Board.BOARD_SIZE), dtype=np.int8)

        history_np = np.stack(list(self.history_stack), axis=0)
        return history_np

    def make_move(self, move: Move, commit = True) -> (bool, str):
        """
        Применить ход на доске.

        Args:
            move: Объект хода

        Returns:
            True если ход легален и применен, False если отклонен
        """
        # Обработка паса
        if move.is_pass():
            # Пас не меняет доску, просто переключаем игрока
            return (True, None)

        active_color = move.color

        if move.position is not None:
            x, y = move.position

            # Проверка базовой легальности позиции
            if not self.is_valid_position(x, y):
                return (False, f"Position {x}:{y} is out of range!")

            if self.current_state[x, y] != self.EMPTY:
                return (False, f"Position {x}:{y} is not empty!")

            # ===== ШАГ 1: Установка камня и проверка захватов =====
            self.future_state_1 = self.current_state.copy()
            self.future_state_1[x, y] = active_color

            # Проверяем захваты камней оппонента
            marked = self.check_captures(self.future_state_1)

            # Снимаем камни оппонента
            self.remove_captured_stones(self.future_state_1, marked, exclude_color=active_color)

            # Проверяем дыхание группы установленного камня
            group, liberties = self.find_group_and_liberties(x, y, self.future_state_1)

            # Если у группы нет дыханий - самоубийственный ход (нелегален)
            if liberties == 0:
                return (False, f"Move {x}:{y} will result in suicide!")

            # Проверка правила "ко" после установки камня
            if not self.check_ko_rule(self.future_state_1, active_color):
                return (False, f"Move {x}:{y} violates ko rule!")

        else:
            self.future_state_1 = self.current_state.copy()

        # ===== ШАГ 2: Цикл жизни (если инициирован) =====
        if move.life_cycles == 1:

            # Применяем цикл жизни к камням активного игрока
            self.future_state_2 = self.apply_game_of_life(self.future_state_1, active_color)

            # Первая проверка захватов: снимаем камни оппонента
            marked = self.check_captures(self.future_state_2)
            self.remove_captured_stones(self.future_state_2, marked, exclude_color=active_color)

            # Вторая проверка захватов: снимаем любые камни без дыханий (самоубийственный цикл)
            marked = self.check_captures(self.future_state_2)
            self.remove_captured_stones(self.future_state_2, marked, exclude_color=None)

            # Проверка правила "ко" после цикла жизни
            if not self.check_ko_rule(self.future_state_2, active_color):
                return (False, f"GoL iteration will violate ko rule!")

            # Применяем финальное состояние
            final_state = self.future_state_2
        else:
            # Без цикла жизни
            final_state = self.future_state_1

        # ===== ШАГ 3: Применение хода =====
        # Сохраняем текущее состояние в историю
        if commit:
            self.history_stack.append(self.current_state)
            self.history_stack.append(self.future_state_1)

            # Обновляем текущее состояние
            self.current_state = final_state

        return (True, "Success")

    def get_current_state(self) -> np.ndarray:
        """Получить текущее состояние доски."""
        return self.current_state.copy()

    def __repr__(self) -> str:
        """Строковое представление доски."""
        stones_black = np.sum(self.current_state == self.BLACK)
        stones_white = np.sum(self.current_state == self.WHITE)
        return f"Board(black={stones_black}, white={stones_white})"

    def display_board(self) -> str:
        """
        Визуализировать текущее состояние доски с эмодзи.

        Легенда:
        - ⚪ (белый круг) = пустая ячейка
        - ⚫ (черный круг) = черный камень
        - 🔴 (красный круг) = белый камень

        Returns:
            Строковое представление доски
        """
        symbols = {
            self.EMPTY: '⚪',
            self.BLACK: '⚫',
            self.WHITE: '🔴'
        }

        lines = []

        for x in range(BOARD_SIZE):
            row_str = f"{x:2d}|"
            for y in range(BOARD_SIZE):
                row_str += symbols[self.current_state[x, y]] + ""
            lines.append(row_str)

        return "\n".join(lines)

    def display_board_with_permissions(self, current_player : int) -> str:
        """
        Визуализировать текущее состояние доски с эмодзи.

        Легенда:
        - ⚪ (белый круг) = пустая ячейка
        - ⚫ (черный круг) = черный камень
        - 🔴 (красный круг) = белый камень

        Returns:
            Строковое представление доски
        """
        symbols = {
            self.BLACK: '⚫',
            self.WHITE: '🔴',
            self.EMPTY + 10 : '❎',
            self.EMPTY + 100 : '🟩',
            self.EMPTY : '⬜',
        }

        lines = []

        history_np = self.get_history_np()

        for x in range(BOARD_SIZE):
            row_str = f"{x:2d}|"
            for y in range(BOARD_SIZE):
                if self.current_state[x,y] != self.EMPTY:
                    row_str += symbols[self.current_state[x, y]] + ""
                else:
                    stone, gol = position_permissions_numba(self.current_state, history_np, current_player, x, y)
                    res = self.EMPTY + (100 if stone and gol else 0) + (10 if stone and not gol else 0)
                    row_str += symbols[res] + ""


            lines.append(row_str)

        return "\n".join(lines)


    def get_score(self) -> dict:
        """
        Подсчет очков по китайским правилам.

        Китайские правила: очки = камни на доске + полностью окруженная территория.
        Территория принадлежит игроку, если все соседи этой пустой области - камни одного цвета.
        Если у пустой области есть соседи обоих цветов - территория нейтральная (ничья).

        Returns:
            dict: {
                'black': int - очки черных,
                'white': int - очки белых,
                'neutral': int - нейтральные пункты,
                'black_stones': int - камни черных,
                'white_stones': int - камни белых,
                'black_territory': int - территория черных,
                'white_territory': int - территория белых
            }
        """
        # Счетчики
        black_stones = np.sum(self.current_state == self.BLACK)
        white_stones = np.sum(self.current_state == self.WHITE)

        black_territory = 0
        white_territory = 0
        neutral_territory = 0

        # Матрица для отслеживания проверенных ячеек
        visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)

        # Обходим доску и ищем пустые регионы
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                # Пропускаем уже проверенные ячейки и занятые камнями
                if visited[x, y] or self.current_state[x, y] != self.EMPTY:
                    continue

                # Нашли непроверенную пустую ячейку - исследуем регион
                region, neighbor_colors = self._explore_empty_region(x, y, visited)

                # Определяем владельца региона по цветам соседей
                if self.BLACK in neighbor_colors and self.WHITE not in neighbor_colors:
                    # Регион окружен только черными камнями
                    black_territory += len(region)
                elif self.WHITE in neighbor_colors and self.BLACK not in neighbor_colors:
                    # Регион окружен только белыми камнями
                    white_territory += len(region)
                else:
                    # Регион граничит с обоими цветами - нейтральный
                    neutral_territory += len(region)

        # Итоговый подсчет по китайским правилам
        black_score = black_stones + black_territory
        white_score = white_stones + white_territory

        return {
            'black': black_score,
            'white': white_score,
            'neutral': neutral_territory,
            'black_stones': black_stones,
            'white_stones': white_stones,
            'black_territory': black_territory,
            'white_territory': white_territory
        }

    def get_territories(self) -> np.ndarray:
        """
        Создает карту контроля территорий по правилам Тромпа-Тейлора (Area scoring).

        Returns:
            np.ndarray: Матрица размера (BOARD_SIZE, BOARD_SIZE) подконтрольных территорий:
        """
        size = self.BOARD_SIZE
        visited = np.zeros((size, size), dtype=bool)

        # Копируем текущее состояние доски, чтобы сразу учесть стоящие камни
        territory_map = np.copy(self.current_state)

        # Исследуем только пустые регионы
        for x in range(size):
            for y in range(size):
                if self.current_state[x, y] == self.EMPTY and not visited[x, y]:
                    # Находим пустой регион и цвета окружающих его камней
                    region, neighbor_colors = self._explore_empty_region(x, y, visited)

                    # Если пустой регион окружен камнями только одного цвета,
                    # он становится территорией этого цвета
                    if len(neighbor_colors) == 1:
                        owner_color = neighbor_colors.pop()

                        # Закрашиваем весь пустой регион цветом владельца
                        for rx, ry in region:
                            territory_map[rx, ry] = owner_color

                    # Если len(neighbor_colors) == 2 (касается обоих цветов) - это дамэ, оставляем EMPTY
                    # Если len(neighbor_colors) == 0 (полностью пустая доска) - оставляем EMPTY

        return territory_map

    def _explore_empty_region(self, x: int, y: int, visited: np.ndarray) -> Tuple[Set[Tuple[int, int]], Set[int]]:
        """
        Исследовать пустой регион методом BFS и найти цвета соседних камней.

        Args:
            x, y: Начальная позиция пустой ячейки
            visited: Матрица посещенных ячеек

        Returns:
            (region, neighbor_colors):
                - region: множество координат пустых ячеек в регионе
                - neighbor_colors: множество цветов камней, граничащих с регионом
        """
        region = set()
        neighbor_colors = set()
        queue = deque([(x, y)])

        while queue:
            cx, cy = queue.popleft()

            if visited[cx, cy]:
                continue

            # Проверяем, что ячейка пустая
            if self.current_state[cx, cy] != self.EMPTY:
                continue

            visited[cx, cy] = True
            region.add((cx, cy))

            # Проверяем 4 соседей
            for nx, ny in get_neighbors_4(cx, cy):
                if self.current_state[nx, ny] == self.EMPTY and not visited[nx, ny]:
                    # Сосед тоже пустой - добавляем в очередь
                    queue.append((nx, ny))
                elif self.current_state[nx, ny] != self.EMPTY:
                    # Сосед - камень, запоминаем его цвет
                    neighbor_colors.add(self.current_state[nx, ny])

        return region, neighbor_colors

    def print_score(self):
        """Вывести результаты подсчета очков."""
        score = self.get_score()

        print("=" * 60)
        print("ПОДСЧЕТ ОЧКОВ (Китайские правила)")
        print("=" * 60)
        print(f"⚫ ЧЕРНЫЕ:")
        print(f"   Камни на доске: {score['black_stones']}")
        print(f"   Территория:     {score['black_territory']}")
        print(f"   ИТОГО:          {score['black']}")
        print()
        print(f"🔴 БЕЛЫЕ:")
        print(f"   Камни на доске: {score['white_stones']}")
        print(f"   Территория:     {score['white_territory']}")
        print(f"   ИТОГО:          {score['white']}")
        print()
        print(f"⚪ Нейтральные пункты: {score['neutral']}")
        print("=" * 60)

    def get_network_input(self, current_player_color: int, states_to_save : int) -> np.ndarray:
        """
        Получить входной тензор для нейросети в формате AlphaGo Zero.

        Структура тензора size×size×states_to_save + 1:
        - Каналы 0-states_to_save / 2 - 1: Позиции камней текущего игрока (последние 8 ходов, от новых к старым)
        - Каналы states_to_save / 2 - states_to_save - 1: Позиции камней оппонента (последние 8 ходов, от новых к старым)
        - Канал states_to_save: Индикатор цвета (1.0 если текущий игрок = BLACK, 0.0 если WHITE)

        Args:
            current_player_color: Цвет текущего игрока (BLACK=1 или WHITE=2)

        Returns:
            Тензор размера (BOARD_SIZE, BOARD_SIZE, states_to_save+1) с float32 значениями в диапазоне [0, 1]
        """
        tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, states_to_save + 1), dtype=np.float32)

        # Определяем цвет оппонента
        opponent_color = self.get_opponent(current_player_color)

        # Получаем историю позиций (текущая + предыдущие)
        # history_stack содержит предыдущие состояния, текущее состояние отдельно
        history_list = list(self.history_stack)  # Копия истории

        all_states =  history_list + [self.current_state.copy()]

        n = int(states_to_save / 2) - 1
        states_to_use = all_states[-n:][::-1]

        # Заполняем первые 2n каналов историей позиций
        for i, state in enumerate(states_to_use):
            # Канал i (0 - n-1): Камни текущего игрока
            tensor[:, :, i + 1] = (state == current_player_color).astype(np.float32)

            # Канал i+8 (n - 2n-1): Камни оппонента
            tensor[:, :, i + 1 + int(states_to_save / 2)] = (state == opponent_color).astype(np.float32)

        # Закидываем прогноз на будущее
        current_state_after_GoL = self.apply_game_of_life(self.current_state, current_player_color)
        marked = self.check_captures(current_state_after_GoL)
        self.remove_captured_stones(current_state_after_GoL, marked, exclude_color=current_player_color)
        marked = self.check_captures(current_state_after_GoL)
        self.remove_captured_stones(current_state_after_GoL, marked, exclude_color=None)
        tensor[:,:, 0] = (current_state_after_GoL == current_player_color).astype(np.float32)

        current_state_after_GoL = self.apply_game_of_life(self.current_state, opponent_color)
        marked = self.check_captures(current_state_after_GoL)
        self.remove_captured_stones(current_state_after_GoL, marked, exclude_color=opponent_color)
        marked = self.check_captures(current_state_after_GoL)
        self.remove_captured_stones(current_state_after_GoL, marked, exclude_color=None)
        tensor[:,:, int(states_to_save / 2)] = (current_state_after_GoL == opponent_color).astype(np.float32)


        # Канал states_to_save: Индикатор цвета текущего игрока
        # 1.0 для черных (BLACK=1), 0.0 для белых (WHITE=2)
        if current_player_color == self.BLACK:
            tensor[:, :, states_to_save] = 1.0
        else:  # WHITE
            tensor[:, :, states_to_save] = 0.0

        return tensor

    def get_network_input_pytorch(self, current_player_color: int, states_to_save = 32) -> np.ndarray:
        """
        Получить входной тензор в формате PyTorch (C, H, W).

        Args:
            current_player_color: Цвет текущего игрока

        Returns:
            Тензор размера (17, 19, 19) для PyTorch Conv2d
        """
        # Получаем стандартный тензор (H, W, C)
        tensor_hwc = self.get_network_input(current_player_color, states_to_save)

        # Транспонируем в (C, H, W) для PyTorch
        tensor_chw = np.transpose(tensor_hwc, (2, 0, 1))

        return tensor_chw

    def from_network_input(self, tensor: np.ndarray) -> int:
        """
        Восстанавливает текущее состояние (current_state) и историю (history_stack)
        из входного тензора нейросети.

        Args:
            tensor: Тензор размера (BOARD_SIZE, BOARD_SIZE, states_to_save + 1)

        Returns:
            int: Цвет текущего игрока (восстановленный из тензора)
        """
        board_size_y, board_size_x, channels = tensor.shape
        states_to_save = channels - 1
        half_states = int(states_to_save / 2)
        n = half_states - 1

        # 1. Определяем цвет текущего игрока из последнего канала
        # Проверяем значение в ячейке (0, 0), так как весь слой заполнен одним числом
        if tensor[0, 0, states_to_save] > 0.0:
            current_player_color = self.BLACK
        else:
            current_player_color = self.WHITE

        opponent_color = self.get_opponent(current_player_color)

        reconstructed_states = []

        # 2. Восстанавливаем состояния
        # Каналы 0 и half_states пропускаем - там лежат прогнозы Game of Life.
        # Реальные исторические состояния лежат в индексах от 1 до n.
        for i in range(1, n + 1):
            # Создаем пустую доску (0 = пусто)
            state = np.zeros((board_size_y, board_size_x), dtype=np.int32)

            # Накладываем камни текущего игрока
            player_mask = tensor[:, :, i] > 0.5
            state[player_mask] = current_player_color

            # Накладываем камни оппонента
            opponent_mask = tensor[:, :, i + half_states] > 0.5
            state[opponent_mask] = opponent_color

            reconstructed_states.append(state)

        # 3. Приводим хронологию в правильный порядок
        # В тензоре состояния хранились от самого нового (1) к старым (n).
        # Переворачиваем список, чтобы получить естественный ход времени (от старых к новым).
        reconstructed_states.reverse()

        # 4. Записываем данные в свойства объекта
        if reconstructed_states:
            # Самое новое (последнее в перевернутом списке) - это текущее состояние
            self.current_state = reconstructed_states[-1]

            # Все предыдущие - это история
            history_part = reconstructed_states[:-1]

            # Поддерживаем случай, если history_stack реализован как collections.deque
            if isinstance(self.history_stack, deque):
                self.history_stack.clear()
                self.history_stack.extend(history_part)
            else:
                self.history_stack = history_part

        return current_player_color
