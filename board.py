from typing import Tuple
import numpy as np
from numba import njit

from config import BOARD_SIZE, board_size_sqr, EMPTY, BLACK, WHITE, NN_HISTORY, MAX_HISTORY, STATES_TO_NN, IN_CHANNELS, \
    symbols


# --- Константы и вспомогательные функции, также помеченные @njit ---

"""
Легенда:
- ⬜ = пустая ячейка
- ⚫ = черный камень
- 🔴 = белый камень
- ❎ = можно поставить камень, но не провести цикл жизни
- 🟩 = можно установить камень и затем провести цикл жизни
"""


@njit
def is_valid_position(x: int, y: int) -> bool:
    """Проверяет, находится ли позиция в пределах доски."""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

# --- Основные оптимизированные функции ---

@njit
def find_group_and_liberties(x: int, y: int, state: np.ndarray, group_array: np.ndarray, visited: np.ndarray) -> Tuple[int, int]:
    """
    Находит группу камней, подсчитывает её размер и количество дыханий (liberties).
    Зачищает group_array,
    НЕ ЗАЧИЩАЕТ visited!!!
    Returns:
        group_array: np.ndarray размера (BOARD_SIZE * BOARD_SIZE, 2) с координатами.
        group_size: int, реальное количество камней в группе.
        liberties_count: int, количество уникальных степеней свободы.
    """


    # Создаем массив фиксированного размера под группу.
    # Используем np.empty для скорости (остальной мусор отсечется параметром group_size)

    #group_array = np.empty((board_size_sqr, 2), dtype=np.int32)

    color = state[x, y]
    if color == EMPTY:
        return 0, 0

    group_array.fill(0)

    visited[x, y] = True

    # Первая точка группы
    group_array[0, 0] = x
    group_array[0, 1] = y

    head = 0
    tail = 1  # tail в итоге будет равен размеру группы
    liberties_count = 0

    # Векторы смещений для 4 соседей (встроено для максимальной скорости)
    dx = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy = np.array([0, 0, -1, 1], dtype=np.int32)

    while head < tail:
        cx = group_array[head, 0]
        cy = group_array[head, 1]
        head += 1

        for i in range(4):
            nx = cx + dx[i]
            ny = cy + dy[i]

            # Проверка границ доски
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if not visited[nx, ny]:
                    neighbor_color = state[nx, ny]

                    if neighbor_color == EMPTY:
                        # Нашли уникальное дыхание (свободную клетку)
                        visited[nx, ny] = True
                        liberties_count += 1

                    elif neighbor_color == color:
                        # Нашли камень той же группы
                        visited[nx, ny] = True
                        group_array[tail, 0] = nx
                        group_array[tail, 1] = ny
                        tail += 1

    # В group_array первые `tail` элементов — это координаты нашей группы
    # tail — это group_size
    return tail, liberties_count

@njit
def check_captures_local(x: int, y: int, state: np.ndarray, marked_for_death : np.ndarray, visited: np.ndarray, group_array: np.ndarray) -> bool:
    """
    Быстрая проверка захватов только вокруг сыгранного камня (x, y).
    Сначала проверяет группы противника на захват.
    Затем проверяет саму группу сыгранного камня (на случай суицидального хода).

    Args:
        x: x позиция только что поставленного камня.
        y: y позиция только что поставленного камня.
        state: Текущее состояние доски (после постановки камня).

    Returns:
        np.ndarray: Матрица такого же размера, где ненулевые элементы
                    показывают цвет захваченных камней.
    """

    played_color = state[x, y]
    if played_color == EMPTY:
        return False

    marked_for_death.fill(0)
    visited.fill(0)

    # Векторы смещений для 4 соседей
    dx = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy = np.array([0, 0, -1, 1], dtype=np.int32)

    has_captures = False
    # 1. Сначала проверяем 4 соседей (камни противника)
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]

        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            neighbor_color = state[nx, ny]
            # Проверяем только камни противника, которые мы еще не проверяли
            if neighbor_color != EMPTY and neighbor_color != played_color and not visited[nx, ny]:

                size, liberties = find_group_and_liberties(nx, ny, state, group_array, visited)

                # Помечаем всю группу противника как проверенную
                for j in range(size):
                    gx, gy = group_array[j, 0], group_array[j, 1]
                    visited[gx, gy] = True

                # Если у группы противника не осталось дыханий - она захвачена
                if liberties == 0:
                    for j in range(size):
                        gx, gy = group_array[j, 0], group_array[j, 1]
                        marked_for_death[gx, gy] = neighbor_color
                        has_captures = True

    return has_captures

@njit
def check_captures(state: np.ndarray, visited: np.ndarray, marked_for_death: np.ndarray, group_array: np.ndarray) -> None:
    """
    Проверяет всю доску и отмечает группы камней без дыханий для захвата.
    Исправлено: цикл идет только по реальному размеру группы (size).
    """

    visited.fill(0)
    marked_for_death.fill(0)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if visited[x, y] or state[x, y] == EMPTY:
                continue

            size, liberties = find_group_and_liberties(x, y, state, group_array, visited)

            # Если дыханий нет - отмечаем к смерти
            if liberties == 0:
                stone_color = state[group_array[0, 0], group_array[0, 1]]
                for i in range(size):
                    gx = group_array[i, 0]
                    gy = group_array[i, 1]
                    marked_for_death[gx, gy] = stone_color

    return

@njit
def remove_captured_stones(state: np.ndarray, marked: np.ndarray, exclude_color: int = EMPTY) -> None:
    """
    Снимает с доски отмеченные камни. exclude_color используется для защиты своих камней.
    """
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if marked[x, y] != EMPTY and marked[x, y] != exclude_color:
                state[x, y] = EMPTY

@njit
def apply_game_of_life(state: np.ndarray, color: int, future_state: np.ndarray) -> None:
    """
    Применяет один цикл Game of Life для камней указанного цвета.
    """

    future_state[:] = state

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
                    future_state[x, y] = EMPTY
            elif state[x, y] == EMPTY:
                # Правило рождения
                if same_color_count == 3:
                    future_state[x, y] = color

    return

@njit
def get_territories(board: np.ndarray, visited: np.ndarray, territory_map: np.ndarray, queue_x : np.ndarray, queue_y : np.ndarray) -> None:
    """
    Numba-метод для вычисления территорий по правилам Тромпа-Тейлора.
    """
    visited.fill(0)

    territory_map[:] = board


    # Заранее выделяем память под массивы для BFS очереди
    queue_x.fill(0)
    queue_y.fill(0)
    #queue_y = np.zeros(board_size_sqr, dtype=np.int32)

    # Векторы направлений для 4 соседей
    dx = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy = np.array([0, 0, -1, 1], dtype=np.int32)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x, y] == EMPTY and not visited[x, y]:
                # Инициализация BFS
                head = 0
                tail = 0

                # Добавляем стартовую ячейку
                queue_x[tail] = x
                queue_y[tail] = y
                tail += 1
                visited[x, y] = True

                found_color = EMPTY
                is_mixed = False

                # BFS-цикл
                while head < tail:
                    cx = queue_x[head]
                    cy = queue_y[head]
                    head += 1

                    # Проверяем 4 соседей
                    for i in range(4):
                        nx = cx + dx[i]
                        ny = cy + dy[i]

                        # Проверка границ доски
                        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                            n_val = board[nx, ny]

                            if n_val == EMPTY:
                                if not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    queue_x[tail] = nx
                                    queue_y[tail] = ny
                                    tail += 1
                            else:
                                # Сосед — камень. Проверяем цвета.
                                if found_color == EMPTY:
                                    found_color = n_val
                                elif found_color != n_val:
                                    is_mixed = True

                # Если регион окружен камнями только одного цвета (и это не полностью пустая доска)
                if found_color != EMPTY and not is_mixed:
                    # Вся история посещений региона уже лежит в массиве queue от 0 до tail
                    for i in range(tail):
                        rx = queue_x[i]
                        ry = queue_y[i]
                        territory_map[rx, ry] = found_color

    return

@njit
def get_opponent(color: int) -> int:
    """Получить цвет оппонента."""
    return WHITE if color == BLACK else BLACK

def generate_zobrist_table() -> np.ndarray:
    """
    Создает таблицу 256-битных (4 x uint64) случайных чисел для каждой клетки и состояния.
    Размер: (BOARD_SIZE, BOARD_SIZE, 3, 4), где 3 - это состояния (Пусто, Игрок 1, Игрок 2).
    """

    z_table = np.zeros((BOARD_SIZE, BOARD_SIZE, 3, 4), dtype=np.uint64)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for state_idx in range(3):
                for part in range(4):
                    # Генерируем два 32-битных числа и склеиваем в 64-битное
                    # (это платформонезависимый и безопасный способ генерации uint64)
                    high = np.random.randint(0, 2 ** 32, dtype=np.uint64)
                    low = np.random.randint(0, 2 ** 32, dtype=np.uint64)
                    z_table[r, c, state_idx, part] = (high << np.uint64(32)) | low
    return z_table

@njit(cache=True, fastmath=True)
def compute_zobrist_hash_numba(state: np.ndarray, z_table: np.ndarray, zobrist_hash: np.ndarray) -> None:
    """
    Вычисляет 256-битный хеш доски (возвращает массив из 4 uint64).
    """
    rows, cols = state.shape

    #h = np.zeros(4, dtype=np.uint64)
    zobrist_hash.fill(0)

    for r in range(rows):
        for c in range(cols):
            val = state[r, c]

            # Маппинг цвета камня в индекс таблицы [0, 1, 2].
            # Подкорректируйте, если ваши цвета (1 и -1)
            idx = 0
            if val != EMPTY:
                idx = 1 if val == BLACK else 2  # Например, 1 -> 1, 2 (или -1) -> 2

            zobrist_hash[0] ^= z_table[r, c, idx, 0]
            zobrist_hash[1] ^= z_table[r, c, idx, 1]
            zobrist_hash[2] ^= z_table[r, c, idx, 2]
            zobrist_hash[3] ^= z_table[r, c, idx, 3]

    return

@njit
def game_of_life_with_captures(current_state: np.ndarray, future_state: np.ndarray, visited: np.ndarray, marked_for_death:np.ndarray, group_array:np.ndarray, player_color : int = EMPTY) -> None:

    apply_game_of_life(current_state, player_color, future_state)

    check_captures(future_state, visited, marked_for_death, group_array)

    remove_captured_stones(future_state, marked_for_death, exclude_color=player_color)

    check_captures(future_state, visited, marked_for_death, group_array)

    remove_captured_stones(future_state, marked_for_death)

    return

@njit(cache=True)
def check_superko_numba(state_hash: np.ndarray, hash_history: np.ndarray, count: int) -> bool:
    """
    Проверяет, встречался ли state_hash в истории hash_history.
    Returns: True - если ход легален (повторов нет), False - если это суперко (повтор).
    """
    for i in range(count):
        # Строгое совпадение всех 256 бит (всех 4 частей)
        if (hash_history[i, 0] == state_hash[0] and
                hash_history[i, 1] == state_hash[1] and
                hash_history[i, 2] == state_hash[2] and
                hash_history[i, 3] == state_hash[3]):
            return False
    return True

@njit(cache=True)
def position_permissions_numba(
        current_state: np.ndarray,
        history_hashes: np.ndarray,  # Передаем массив хешей вместо 3D-массива досок
        hash_count: int,  # Текущее количество хешей в буфере
        zobrist_table: np.ndarray,  # Таблица для вычисления хешей
        current_player: int,
        x: int, y: int,
        future_state_1 : np.ndarray,
        future_state_2 : np.ndarray,

        visited: np.ndarray,
        marked_for_death:np.ndarray,
        group_array:np.ndarray,

        zobrist_hash1 :np.ndarray,
        zobrist_hash2 :np.ndarray

) -> Tuple[bool, bool]:
    # --- Быстрые проверки ---
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False, False

    if current_state[x, y] != EMPTY:
        return False, False

    # --- ШАГ 1: Симуляция постановки камня ---

    future_state_1[:] = current_state

    future_state_1[x, y] = current_player


    any_capture = check_captures_local(x,y, future_state_1, marked_for_death, visited, group_array)
    if any_capture:
        remove_captured_stones(future_state_1, marked_for_death, current_player)

    # Проверка на самоубийственный ход (используем распаковку 3 значений из новой версии функции)
    visited.fill(0)
    size, liberties = find_group_and_liberties(x, y, future_state_1, group_array, visited)
    if liberties == 0:
        return False, False

    # Проверка правила Суперко (Зобристово хеширование)
    compute_zobrist_hash_numba(future_state_1, zobrist_table, zobrist_hash1)
    if not check_superko_numba(zobrist_hash1, history_hashes, hash_count):
        return False, False

    # --- ШАГ 2: Симуляция цикла жизни (Game of Life) ---
    game_of_life_with_captures(future_state_1, future_state_2, visited, marked_for_death, group_array, current_player)

    # Финальная проверка Суперко для второй стадии
    compute_zobrist_hash_numba(future_state_2, zobrist_table, zobrist_hash2)
    if not check_superko_numba(zobrist_hash2, history_hashes, hash_count):
        return True, False

    # Проверка, не сцепился ли Game of Life с камнем, вернув доску в future_state_1
    if (zobrist_hash1[0] == zobrist_hash2[0] and zobrist_hash1[1] == zobrist_hash2[1] and
            zobrist_hash1[2] == zobrist_hash2[2] and zobrist_hash1[3] == zobrist_hash2[3]):
        return True, False

    # Если все проверки пройдены, ход полностью легален
    return True, True

@njit
def get_legal_moves_mask_numba(
        current_state: np.ndarray,
        history_hashes: np.ndarray,  # Передаем массив хешей вместо 3D-массива досок
        hash_count: int,  # Текущее количество хешей в буфере
        zobrist_table: np.ndarray,  # Таблица для вычисления хешей
        current_player: int,
        out: np.ndarray,

        future_state_1: np.ndarray,
        future_state_2: np.ndarray,

        visited: np.ndarray,
        marked_for_death: np.ndarray,
        group_array: np.ndarray,

        zobrist_hash1: np.ndarray,
        zobrist_hash2: np.ndarray
) -> None:

    # prange распараллеливает внешний цикл

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            placement, gol = position_permissions_numba(current_state, history_hashes, hash_count, zobrist_table, current_player, x, y,
                                                        future_state_1, future_state_2, visited, marked_for_death, group_array, zobrist_hash1, zobrist_hash2)

            idx_no_life = x * BOARD_SIZE + y
            if placement:
                out[idx_no_life] = 1
            else:
                out[idx_no_life] = 0

            idx_with_life = BOARD_SIZE * BOARD_SIZE + (x * BOARD_SIZE + y)
            if gol:
                out[idx_with_life] = 1
            else:
                out[idx_with_life] = 0

    # Пас всегда легален
    out[BOARD_SIZE * BOARD_SIZE * 2] = 1.0


class DeltaBoard:
    def __init__(self):
        #Потому что ход прогоняет 2 стейта
        #Хранит новый куррент и стейт после текущего куррента
        self.next_state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        #Хранит хеш текущего куррента и стейта после него (next_state по 1 индексу)
        self.next_hash = np.zeros((2,4), dtype=np.uint64)

        #Когда пас, типа "заигнорь меня"
        self.valid = False


class Board:
    """
    Класс доски для игры Go + Game of Life.

    Управляет состоянием игры, валидацией и применением ходов.
    """

    cached_zobrist_table = None

    def __init__(self):
        """Инициализация доски."""

        self.current_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.history_stack = np.zeros((NN_HISTORY, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        # ИСТОРИЯ ХЕШЕЙ (нужна исключительно для быстрой работы правила Суперко)
        self.history_hashes = np.zeros((MAX_HISTORY, 4), dtype=np.uint64)


        if Board.cached_zobrist_table is None:
            self.zobrist_table = generate_zobrist_table()
            Board.cached_zobrist_table = self.zobrist_table
        else:
            self.zobrist_table = Board.cached_zobrist_table



        self.history_ptr = 0
        self.history_hash_ptr = 0

        self.history_count = 0
        self.history_hash_count = 0

        self.future_state_1 = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.future_state_2 = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        self.temp_hash_1 =  np.zeros(4, dtype=np.uint64)
        self.temp_hash_2 =  np.zeros(4, dtype=np.uint64)

        self.temp_group_array = np.empty((board_size_sqr, 2), dtype=np.int32)
        self.temp_marked_for_death = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.temp_visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)

        self.temp_queue_x = np.zeros(board_size_sqr, dtype=np.int32)
        self.temp_queue_y = np.zeros(board_size_sqr, dtype=np.int32)

        self.tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, IN_CHANNELS), dtype=np.float32)

    def clear(self):
        self.current_state.fill(0)
        self.history_stack.fill(0)
        self.history_hashes.fill(0)

        self.history_ptr = 0
        self.history_hash_ptr = 0

        self.history_count = 0
        self.history_hash_count = 0

    def clone(self):
        new_board = self.__class__.__new__(self.__class__)

        # Копируем текущие и будущие состояния
        new_board.current_state = self.current_state.copy()
        new_board.future_state_1 = self.future_state_1.copy()
        new_board.future_state_2 = self.future_state_2.copy()

        # Zobrist-таблица неизменна для конкретного размера доски,
        # поэтому просто передаем ссылку (экономим память и время)
        new_board.zobrist_table = self.zobrist_table

        # Копируем массивы истории
        new_board.history_stack = self.history_stack.copy()
        new_board.history_hashes = self.history_hashes.copy()

        # Копируем скалярные значения указателей и счетчиков (передаются по значению)
        new_board.history_ptr = self.history_ptr
        new_board.history_hash_ptr = self.history_hash_ptr
        new_board.history_count = self.history_count
        new_board.history_hash_count = self.history_hash_count

        new_board.temp_hash_1 =  np.zeros(4, dtype=np.uint64)
        new_board.temp_hash_2 =  np.zeros(4, dtype=np.uint64)

        new_board.temp_group_array = np.empty((board_size_sqr, 2), dtype=np.int32)
        new_board.temp_marked_for_death = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        new_board.temp_visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)

        new_board.temp_queue_x = np.zeros(board_size_sqr, dtype=np.int32)
        new_board.temp_queue_y = np.zeros(board_size_sqr, dtype=np.int32)

        new_board.tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, IN_CHANNELS), dtype=np.float32)

        return new_board

    def undo(self):



        self.dec_ptr()
        self.dec_ptr()

        self.dec_hash_ptr()
        self.dec_hash_ptr()

        np.copyto(self.current_state, self.history_stack[self.history_ptr])



    def apply_delta(self, delta_board : DeltaBoard):


        if not delta_board.valid:
            return

        np.copyto(self.history_stack[self.history_ptr], self.current_state)
        np.copyto(self.history_hashes[self.history_hash_ptr], delta_board.next_hash[1])

        self.inc_ptr()
        self.inc_hash_ptr()

        np.copyto(self.history_stack[self.history_ptr], delta_board.next_state[1])
        np.copyto(self.history_hashes[self.history_hash_ptr], delta_board.next_hash[0])

        self.inc_ptr()
        self.inc_hash_ptr()

        np.copyto(self.current_state, delta_board.next_state[0])

    def inc_ptr(self):
        self.history_ptr = (self.history_ptr + 1) % NN_HISTORY
        if self.history_count < NN_HISTORY:
            self.history_count += 1

    def dec_ptr(self):
        self.history_ptr = (self.history_ptr - 1 + NN_HISTORY) % NN_HISTORY
        if self.history_count > 0:
            self.history_count -= 1

    def inc_hash_ptr(self):
        self.history_hash_ptr = (self.history_hash_ptr + 1) % MAX_HISTORY
        if self.history_hash_count < MAX_HISTORY:
            self.history_hash_count += 1

    def dec_hash_ptr(self):
        self.history_hash_ptr = (self.history_hash_ptr - 1 + MAX_HISTORY) % MAX_HISTORY
        if self.history_hash_count > 0:
            self.history_hash_count -= 1

    def copy(self, target: 'Board'):
        # Копируем содержимое массивов (in-place перезапись памяти)

        np.copyto(target.current_state, self.current_state)
        np.copyto(target.history_stack, self.history_stack)
        np.copyto(target.history_hashes, self.history_hashes)

        # Копируем примитивные типы (скаляры)
        target.history_ptr = self.history_ptr
        target.history_hash_ptr = self.history_hash_ptr
        target.history_count = self.history_count
        target.history_hash_count = self.history_hash_count

        np.copyto(target.tensor, self.tensor)

    def push_to_history(self, state: np.ndarray):
        """
        Добавляет состояние в кольцевой буфер истории позиций и сохраняет его хеш.
        """
        # Считаем 256-битный хеш
        compute_zobrist_hash_numba(state, self.zobrist_table, self.temp_hash_1)

        np.copyto(self.history_stack[self.history_ptr], state)
        np.copyto(self.history_hashes[self.history_hash_ptr], self.temp_hash_1)


        self.inc_ptr()
        self.inc_hash_ptr()

    def get_legal_moves_mask(self, current_player: int, out: np.ndarray):
        return get_legal_moves_mask_numba(self.current_state, self.history_hashes, self.history_hash_count, self.zobrist_table,
                                          current_player, out, self.future_state_1, self.future_state_2, self.temp_visited, self.temp_marked_for_death,
                                          self.temp_group_array, self.temp_hash_1, self.temp_hash_2)

    def make_move(self, x:int, y:int, life_cycles:int, color:int, is_pass:bool, out: DeltaBoard) -> (bool, str):
        """
        Применить ход на доске.
        НЕ ПРОВЕРЯЕТ ХОД. ПРОВЕРКА ОТДЕЛЬНО

        Args:
            move: Объект хода
            commit: Менять ли состояние доски
            validate: Проверять ли легальность хода

        Returns:
            True если ход легален и применен, False если отклонен
        """
        # Обработка паса
        if is_pass:
            # Пас не меняет доску, просто переключаем игрока
            out.valid = False
            return True, "Pass"

        np.copyto(self.future_state_1, self.current_state)
        self.future_state_1[x,y] = color

        any_capture = check_captures_local(x,y, self.future_state_1, self.temp_marked_for_death, self.temp_visited, self.temp_group_array)
        if any_capture:
            remove_captured_stones(self.future_state_1, self.temp_marked_for_death, exclude_color=color)

        if life_cycles == 1:
            game_of_life_with_captures(self.future_state_1, self.future_state_2, self.temp_visited, self.temp_marked_for_death, self.temp_group_array, color)
        else:
            np.copyto(self.future_state_2, self.future_state_1)



        # Каждый шаг записывается как отдельная позиция в историю
        self.push_to_history(self.current_state)
        np.copyto(out.next_hash[1], self.temp_hash_1)

        self.push_to_history(self.future_state_1)
        np.copyto(out.next_hash[0], self.temp_hash_1)

        # Обновляем текущее состояние
        np.copyto(self.current_state, self.future_state_2)

        np.copyto(out.next_state[0], self.future_state_2)
        np.copyto(out.next_state[1], self.future_state_1)

        out.valid = True

        return True, "Success"


    def __repr__(self) -> str:
        """Строковое представление доски."""
        stones_black = np.sum(self.current_state == BLACK)
        stones_white = np.sum(self.current_state == WHITE)
        return f"Board(black={stones_black}, white={stones_white})"

    def board_as_text(self) -> str:
        """
        Визуализировать текущее состояние доски с эмодзи.
        """
        lines = []

        for x in range(BOARD_SIZE):
            row_str = f"{x:2d}|"
            for y in range(BOARD_SIZE):
                row_str += symbols[self.current_state[x, y]] + ""
            lines.append(row_str)

        return "\n".join(lines)

    def board_with_permissions_as_text(self, current_player : int, mask: np.ndarray) -> str:
        """
        Визуализировать текущее состояние доски с эмодзи.
        """

        lines = []


        for x in range(BOARD_SIZE):
            row_str = f"{x:2d}|"
            for y in range(BOARD_SIZE):
                if self.current_state[x,y] != EMPTY:
                    row_str += symbols[self.current_state[x, y]] + ""
                else:
                    stone, gol = mask[x * BOARD_SIZE + y], mask[board_size_sqr + x * BOARD_SIZE + y]
                    res = EMPTY + (100 if stone and gol else 0) + (10 if stone and not gol else 0)
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

        np.equal(self.current_state, BLACK, out=self.temp_visited)
        black_stones = np.count_nonzero(self.temp_visited)

        np.equal(self.current_state, WHITE, out=self.temp_visited)
        white_stones = np.count_nonzero(self.temp_visited)

        get_territories(self.current_state, self.temp_visited, self.future_state_1, self.temp_queue_x, self.temp_queue_y)

        np.equal(self.current_state, BLACK, out=self.temp_visited)
        black_territory = np.count_nonzero(self.temp_visited) - black_stones

        np.equal(self.current_state, WHITE, out=self.temp_visited)
        white_territory = np.count_nonzero(self.temp_visited) - white_stones




        # Итоговый подсчет по китайским правилам
        black_score = black_stones + black_territory
        white_score = white_stones + white_territory

        neutral_territory = board_size_sqr - black_score - white_score

        return {
            'black': black_score,
            'white': white_score,
            'neutral': neutral_territory,
            'black_stones': black_stones,
            'white_stones': white_stones,
            'black_territory': black_territory,
            'white_territory': white_territory
        }


    def fast_territories(self) -> Tuple[int, int]:
        get_territories(self.current_state, self.temp_visited, self.future_state_1, self.temp_queue_x,
                        self.temp_queue_y)

        np.equal(self.future_state_1, BLACK, out=self.temp_visited)
        black_territory = 0 + np.count_nonzero(self.temp_visited)

        np.equal(self.future_state_1, WHITE, out=self.temp_visited)
        white_territory = 0 + np.count_nonzero(self.temp_visited)

        return black_territory, white_territory


    def update_network_input(self, current_player_color: int, komi: float) -> None:
        """
        Получить входной тензор для нейросети в формате AlphaGo Zero напрямую из массивов.
        (Zero-allocation версия)
        """
        half_states = STATES_TO_NN // 2
        opponent_color = get_opponent(current_player_color)

        # Очищаем весь тензор нулями in-place перед началом записи
        self.tensor.fill(0.0)

        # === 1. ЗАГЛЯДЫВАНИЕ В БУДУЩЕЕ ===
        game_of_life_with_captures(self.current_state, self.future_state_1, self.temp_visited,
                                   self.temp_marked_for_death, self.temp_group_array, current_player_color)

        # Используем np.equal с записью прямо в temp_visited (у него тип bool), чтобы не выделять память
        np.equal(self.future_state_1, current_player_color, out=self.temp_visited)
        # Копируем булевы значения в float-тензор (NumPy сам скастует True->1.0, False->0.0)
        np.copyto(self.tensor[:, :, 0], self.temp_visited)

        game_of_life_with_captures(self.current_state, self.future_state_1, self.temp_visited,
                                   self.temp_marked_for_death, self.temp_group_array, opponent_color)

        np.equal(self.future_state_1, opponent_color, out=self.temp_visited)
        np.copyto(self.tensor[:, :, half_states], self.temp_visited)

        # === 2. ТЕКУЩЕЕ СОСТОЯНИЕ ===
        if half_states > 1:
            np.equal(self.current_state, current_player_color, out=self.temp_visited)
            np.copyto(self.tensor[:, :, 1], self.temp_visited)

            np.equal(self.current_state, opponent_color, out=self.temp_visited)
            np.copyto(self.tensor[:, :, half_states + 1], self.temp_visited)

        # === 3. ПРОШЛЫЕ СОСТОЯНИЯ (ИЗ КОЛЬЦЕВОГО БУФЕРА) ===
        history_needed = half_states - 2

        if history_needed > 0 and self.history_count > 0:
            states_to_pull = min(history_needed, self.history_count)

            # arange создает маленький массив, но это копейки. Для полной паранойи можно держать его предсозданным
            idx_array = (self.history_ptr - 1 - np.arange(states_to_pull)) % NN_HISTORY
            past_states = self.history_stack[idx_array]  # Это создает 3D view (если повезет) или массив

            # Для 3D истории используем np.equal.
            # Если вы не хотите выделять временный 3D булев массив,
            # можно делать это в цикле по каждому состоянию (states_to_pull обычно маленькое, <= 7)
            for i in range(states_to_pull):
                state_layer = past_states[i]

                # Слой текущего игрока
                np.equal(state_layer, current_player_color, out=self.temp_visited)
                np.copyto(self.tensor[:, :, 2 + i], self.temp_visited)

                # Слой оппонента
                np.equal(state_layer, opponent_color, out=self.temp_visited)
                np.copyto(self.tensor[:, :, half_states + 2 + i], self.temp_visited)

        # === 4. КАНАЛ ЦВЕТА/КОМИ ===
        # .fill() работает in-place для среза
        if current_player_color == BLACK:
            self.tensor[:, :, STATES_TO_NN].fill(-komi)
        else:
            self.tensor[:, :, STATES_TO_NN].fill(komi)


    def get_network_input_pytorch(self, current_player_color: int, komi: float) -> np.ndarray:
        """
        Получить входной тензор в формате PyTorch (C, H, W).
        """
        self.update_network_input(current_player_color, komi)
        tensor_chw = np.transpose(self.tensor, (2, 0, 1))
        return tensor_chw
