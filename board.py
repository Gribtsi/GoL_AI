from collections import namedtuple
from typing import Tuple, Union
import numpy as np
from numba import njit

from config import BOARD_SIZE, board_size_sqr, EMPTY, BLACK, WHITE, NN_HISTORY, MAX_HISTORY, STATES_TO_NN, IN_CHANNELS, \
    symbols, possible_moves_total, pass_code
from move import decode_move




BoardTemp = namedtuple("BoardTemp", [
    "temp_state_1",
    "temp_state_2",
    "temp_hash_1",
    "temp_hash_2",
    "temp_positions_queue",
    "temp_marked_for_death",
    "temp_visited",
    "tensor",
])

DeltaBoardArray = namedtuple("DeltaBoardArray",[
    "next_state",
    "next_hash",
    "valid"
])

BoardData = namedtuple("BoardData",[
    "current_state",
    "history_stack",
    "history_hashes",
    "zobrist_table",

    "history_ptr",
    "history_hash_ptr",
    "history_count",
    "history_hash_count",
])

LAZY_PERMIT = 1
PRECIESE_PERMIT = 2

@njit
def is_valid_position(x: int, y: int) -> bool:
    """Проверяет, находится ли позиция в пределах доски."""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

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
    if (color == EMPTY) or ((visited[x,y] & 1) == 1):
        return 0, 0

    for tx in range(BOARD_SIZE):
        for ty in range(BOARD_SIZE):
            visited[tx,ty] &= ~2

    group_array.fill(0)

    visited[x, y] |= 1

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
            if (0 <= nx < BOARD_SIZE) and (0 <= ny < BOARD_SIZE):
                if visited[nx, ny] == 0:
                    neighbor_color = state[nx, ny]

                    if neighbor_color == color:
                        # Нашли камень той же группы
                        visited[nx, ny] |= 1
                        group_array[tail, 0] = nx
                        group_array[tail, 1] = ny
                        tail += 1

                    elif neighbor_color == EMPTY:
                        # Нашли уникальное дыхание (свободную клетку)
                        visited[nx, ny] |= 2
                        liberties_count += 1



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
            if (neighbor_color != EMPTY) and (neighbor_color != played_color) and not ((visited[nx, ny] & 1) == 1):

                size, liberties = find_group_and_liberties(nx, ny, state, group_array, visited)

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
            if ((visited[x, y] & 1) == 1) or (state[x, y] == EMPTY):
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
            if (marked[x, y] != EMPTY) and (marked[x, y] != exclude_color):
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
                    if (dx == 0) and (dy == 0):
                        continue
                    nx, ny = x + dx, y + dy
                    if is_valid_position(nx, ny) and (state[nx, ny] == color):
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
def get_territories(board: np.ndarray, visited: np.ndarray, territory_map: np.ndarray, queue : np.ndarray) -> None:
    """
    Numba-метод для вычисления территорий по правилам Тромпа-Тейлора.
    """
    visited.fill(0)

    territory_map[:] = board


    # Заранее выделяем память под массивы для BFS очереди
    queue.fill(0)
    #queue_y = np.zeros(board_size_sqr, dtype=np.int32)

    # Векторы направлений для 4 соседей
    dx = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy = np.array([0, 0, -1, 1], dtype=np.int32)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if (board[x, y] == EMPTY) and not ((visited[x, y] & 1) == 1):
                # Инициализация BFS
                head = 0
                tail = 0

                # Добавляем стартовую ячейку
                queue[tail, 0] = x
                queue[tail, 1] = y
                tail += 1
                visited[x, y] |= 1

                found_color = EMPTY
                is_mixed = False

                # BFS-цикл
                while head < tail:
                    cx = queue[head, 0]
                    cy = queue[head, 1]
                    head += 1

                    # Проверяем 4 соседей
                    for i in range(4):
                        nx = cx + dx[i]
                        ny = cy + dy[i]

                        # Проверка границ доски
                        if (0 <= nx < BOARD_SIZE) and (0 <= ny < BOARD_SIZE):
                            n_val = board[nx, ny]

                            if n_val == EMPTY:
                                if not ((visited[nx, ny] & 1) == 1):
                                    visited[nx, ny] |= 1
                                    queue[tail, 0] = nx
                                    queue[tail, 1] = ny
                                    tail += 1

                            else:
                                # Сосед — камень. Проверяем цвета.
                                if found_color == EMPTY:
                                    found_color = n_val

                                elif found_color != n_val:
                                    is_mixed = True

                # Если регион окружен камнями только одного цвета (и это не полностью пустая доска)
                if (found_color != EMPTY) and not is_mixed:
                    # Вся история посещений региона уже лежит в массиве queue от 0 до tail
                    for i in range(tail):
                        rx = queue[i, 0]
                        ry = queue[i, 1]
                        territory_map[rx, ry] = found_color


@njit
def get_opponent(color: int) -> int:
    """Получить цвет оппонента."""
    if color == EMPTY:
        return EMPTY

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
    #h = np.zeros(4, dtype=np.uint64)
    zobrist_hash.fill(0)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
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

@njit
def placement_permission_numba(
        board: BoardData,
        temp: BoardTemp,
        current_player: int,
        x: int, y: int):

    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False

    if board.current_state[x, y] != EMPTY:
        return False

    temp.temp_state_1[:] = board.current_state

    temp.temp_state_1[x, y] = current_player

    any_capture = check_captures_local(x, y, temp.temp_state_1, temp.temp_marked_for_death, temp.temp_visited, temp.temp_positions_queue)
    if any_capture:
        remove_captured_stones(temp.temp_state_1, temp.temp_marked_for_death, current_player)

    # Проверка на самоубийственный ход (используем распаковку 3 значений из новой версии функции)
    temp.temp_visited.fill(0)
    size, liberties = find_group_and_liberties(x, y,temp.temp_state_1, temp.temp_positions_queue, temp.temp_visited)
    if (liberties == 0) and (size > 0):
        return False

    # Проверка правила Суперко (Зобристово хеширование)
    compute_zobrist_hash_numba(temp.temp_state_1, board.zobrist_table, temp.temp_hash_1)
    if not check_superko_numba(temp.temp_hash_1, board.history_hashes, board.history_hash_count[0]):
        return False

    return True

@njit(cache=True)
def position_permissions_numba(
        board: BoardData,
        temp: BoardTemp,
        current_player: int,
        x: int, y: int,
) -> Tuple[bool, bool]:
    # --- Быстрые проверки ---
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False, False

    if board.current_state[x, y] != EMPTY:
        return False, False

    # --- ШАГ 1: Симуляция постановки камня ---

    temp.temp_state_1[:] = board.current_state

    temp.temp_state_1[x, y] = current_player


    any_capture = check_captures_local(x,y, temp.temp_state_1, temp.temp_marked_for_death, temp.temp_visited, temp.temp_positions_queue)
    if any_capture:
        remove_captured_stones(temp.temp_state_1, temp.temp_marked_for_death, current_player)

    # Проверка на самоубийственный ход (используем распаковку 3 значений из новой версии функции)
    temp.temp_visited.fill(0)
    size, liberties = find_group_and_liberties(x, y, temp.temp_state_1, temp.temp_positions_queue, temp.temp_visited)
    if (liberties == 0) and (size > 0):
        return False, False

    # Проверка правила Суперко (Зобристово хеширование)
    compute_zobrist_hash_numba(temp.temp_state_1, board.zobrist_table, temp.temp_hash_1)
    if not check_superko_numba(temp.temp_hash_1, board.history_hashes, board.history_hash_count[0]):
        return False, False

    # --- ШАГ 2: Симуляция цикла жизни (Game of Life) ---
    game_of_life_with_captures(temp.temp_state_1, temp.temp_state_2, temp.temp_visited, temp.temp_marked_for_death, temp.temp_positions_queue, current_player)

    # Финальная проверка Суперко для второй стадии
    compute_zobrist_hash_numba(temp.temp_state_2, board.zobrist_table, temp.temp_hash_2)
    if not check_superko_numba(temp.temp_hash_2, board.history_hashes, board.history_hash_count[0]):
        return True, False

    # Проверка, не сцепился ли Game of Life с камнем, вернув доску в future_state_1
    if (temp.temp_hash_1[0] == temp.temp_hash_2[0] and temp.temp_hash_1[1] == temp.temp_hash_2[1] and
            temp.temp_hash_1[2] == temp.temp_hash_2[2] and temp.temp_hash_1[3] == temp.temp_hash_2[3]):
        return True, False

    # Если все проверки пройдены, ход полностью легален
    return True, True

@njit
def swap_colours(current_state: np.ndarray, out: np.ndarray):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            out[x,y] = get_opponent(current_state[x,y])

@njit
def get_legal_moves_mask_numba(
        board: BoardData,
        temp: BoardTemp,
        current_player: int,
        out: np.ndarray,
) -> None:

    # prange распараллеливает внешний цикл

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            placement, gol = position_permissions_numba(board, temp, current_player, x, y)

            idx_no_life = x * BOARD_SIZE + y
            if placement:
                out[idx_no_life] = 1.0
            else:
                out[idx_no_life] = 0

            idx_with_life = board_size_sqr + idx_no_life
            if gol:
                out[idx_with_life] = 1.0
            else:
                out[idx_with_life] = 0


#Истина, если клетка пустая
@njit
def get_legal_moves_mask_lazy_numba(board: BoardData, out: np.ndarray):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):

            empty = board.current_state[x,y] == EMPTY

            idx_no_life = x * BOARD_SIZE + y
            idx_with_life = board_size_sqr + idx_no_life

            if empty:
                out[idx_no_life] = LAZY_PERMIT
                out[idx_with_life] = LAZY_PERMIT
            else:
                out[idx_no_life] = 0
                out[idx_with_life] = 0


@njit
def copyto_numba(dst, src):
    dst[:] = src



@njit
def circle_add(val: int, add: int, max: int) -> int:
    return (val + max + add) % max

@njit
def apply_delta_board_numba(deltas: DeltaBoardArray, ptr: int, board: BoardData):

    if not deltas.valid[ptr]:
        return

    copyto_numba(board.history_stack[board.history_ptr[0]], board.current_state)
    copyto_numba(board.history_hashes[board.history_hash_ptr[0]], deltas.next_hash[ptr, 1])

    inc_history_ptr(board)

    copyto_numba(board.history_stack[board.history_ptr[0]], deltas.next_state[ptr, 1])
    copyto_numba(board.history_hashes[board.history_hash_ptr[0]], deltas.next_hash[ptr, 0])

    inc_history_ptr(board)

    copyto_numba(board.current_state, deltas.next_state[ptr, 0])

@njit
def inc_history_ptr(board: BoardData):
    board.history_ptr[0] = circle_add(board.history_ptr[0], 1, NN_HISTORY)
    board.history_count[0] = board.history_count[0] + 1

    board.history_hash_ptr[0] = circle_add(board.history_hash_ptr[0], 1, MAX_HISTORY)
    board.history_hash_count[0] = board.history_hash_count[0] + 1

@njit
def dec_history_ptr(board: BoardData):
    board.history_ptr[0] = circle_add(board.history_ptr[0], -1, NN_HISTORY)
    board.history_count[0] = board.history_count[0] + 1

    board.history_hash_ptr[0] = circle_add(board.history_hash_ptr[0], -1, MAX_HISTORY)
    board.history_hash_count[0] = board.history_hash_count[0] + 1

@njit
def push_to_history_numba(board: BoardData, temp: BoardTemp, state: np.ndarray):
    """
    Добавляет состояние в кольцевой буфер истории позиций и сохраняет его хеш.
    """
    # Считаем 256-битный хеш
    compute_zobrist_hash_numba(board.current_state, board.zobrist_table, temp.temp_hash_1)

    copyto_numba(board.history_stack[board.history_ptr[0]], state)
    copyto_numba(board.history_hashes[board.history_hash_ptr[0]], temp.temp_hash_1)

    inc_history_ptr(board)

@njit()
def is_legal_board_move_numba(board: BoardData, temp: BoardTemp, current_player: int, x:int, y:int, gol:int) -> Tuple[int, int]:

        if gol == 0:
            res = placement_permission_numba(board, temp, current_player, x, y)
            return PRECIESE_PERMIT if res else 0, LAZY_PERMIT

        elif gol == 1:
            placement, gol = position_permissions_numba(board, temp, current_player, x, y)
            return PRECIESE_PERMIT if placement else 0, PRECIESE_PERMIT if gol else 0

        else:
            return LAZY_PERMIT, LAZY_PERMIT
@njit()
def make_board_move_numba(delta: DeltaBoardArray, delta_ptr: int, board: BoardData, temp: BoardTemp, encoded_move: int, color: int):
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

    x,y,life_cycles,is_pass,is_swap = decode_move(encoded_move)

    is_pass = encoded_move == pass_code
    if is_pass:
        # Пас не меняет доску, просто переключаем игрока
        delta.valid[delta_ptr] = False
        return

    if not is_swap:
        copyto_numba(temp.temp_state_1, board.current_state)
        temp.temp_state_1[x,y] = color

        any_capture = check_captures_local(x, y, temp.temp_state_1, temp.temp_marked_for_death, temp.temp_visited, temp.temp_positions_queue)
        if any_capture:
            remove_captured_stones(temp.temp_state_1, temp.temp_marked_for_death, exclude_color=color)

        if life_cycles == 1:
            game_of_life_with_captures(temp.temp_state_1, temp.temp_state_2, temp.temp_visited, temp.temp_marked_for_death, temp.temp_positions_queue, color)
        else:
            copyto_numba(temp.temp_state_2, temp.temp_state_1)

    else:
        swap_colours(board.current_state, temp.temp_state_1)
        copyto_numba(temp.temp_state_2, temp.temp_state_1)




    # Каждый шаг записывается как отдельная позиция в историю
    push_to_history_numba(board, temp, board.current_state)
    copyto_numba(delta.next_hash[delta_ptr, 1], temp.temp_hash_1)



    push_to_history_numba(board, temp, temp.temp_state_1)
    copyto_numba(delta.next_hash[delta_ptr, 0], temp.temp_hash_1)

    # Обновляем текущее состояние
    copyto_numba(board.current_state, temp.temp_state_2)

    copyto_numba(delta.next_state[delta_ptr, 0], temp.temp_state_2)
    copyto_numba(delta.next_state[delta_ptr, 1], temp.temp_state_1)

    delta.valid[delta_ptr] = True

@njit
def write_eq_plane_chw(dst3, ch, src2, value):
    h = src2.shape[0]
    w = src2.shape[1]
    for y in range(h):
        for x in range(w):
            dst3[ch, y, x] = 1.0 if src2[y, x] == value else 0.0


@njit
def fill_plane_chw(dst3, ch, value):
    h = dst3.shape[1]
    w = dst3.shape[2]
    for y in range(h):
        for x in range(w):
            dst3[ch, y, x] = value

@njit()
def update_network_input_numba(board: BoardData, temp: BoardTemp, current_player_color: int, komi_norm: float):
    """
    Пишет вход сразу в CHW:
    temp.tensor_chw.shape == (IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    """
    half_states = STATES_TO_NN // 2
    opponent_color = get_opponent(current_player_color)

    temp.tensor.fill(0.0)

    # === 1. LOOKAHEAD ===
    game_of_life_with_captures(
        board.current_state,
        temp.temp_state_1,
        temp.temp_visited,
        temp.temp_marked_for_death,
        temp.temp_positions_queue,
        current_player_color
    )
    write_eq_plane_chw(temp.tensor, 0, temp.temp_state_1, current_player_color)

    game_of_life_with_captures(
        board.current_state,
        temp.temp_state_1,
        temp.temp_visited,
        temp.temp_marked_for_death,
        temp.temp_positions_queue,
        opponent_color
    )
    write_eq_plane_chw(temp.tensor, half_states, temp.temp_state_1, opponent_color)

    # === 2. CURRENT STATE ===
    if half_states > 1:
        write_eq_plane_chw(temp.tensor, 1, board.current_state, current_player_color)
        write_eq_plane_chw(temp.tensor, half_states + 1, board.current_state, opponent_color)

    # === 3. HISTORY ===
    history_needed = half_states - 2

    if history_needed > 0 and board.history_count[0] > 0:
        states_to_pull = history_needed
        if states_to_pull > board.history_count[0]:
            states_to_pull = board.history_count[0]

        for i in range(states_to_pull):
            idx = (board.history_ptr[0] - 1 - i) % NN_HISTORY
            state_layer = board.history_stack[idx]

            write_eq_plane_chw(temp.tensor, 2 + i, state_layer, current_player_color)
            write_eq_plane_chw(temp.tensor, half_states + 2 + i, state_layer, opponent_color)

    # === 4. KOMI CHANNEL ===
    fill_plane_chw(temp.tensor, STATES_TO_NN, komi_norm)


def board_with_permissions_as_text(board_state: np.ndarray, mask: np.ndarray) -> str:
    """
    Визуализировать текущее состояние доски с эмодзи.
    """

    lines = []

    for x in range(BOARD_SIZE):
        row_str = f"{x:2d}|"
        for y in range(BOARD_SIZE):
            if board_state[x,y] != EMPTY:
                row_str += symbols[board_state[x, y]] + ""
            else:
                stone, gol = mask[x * BOARD_SIZE + y] > 0, mask[board_size_sqr + x * BOARD_SIZE + y] > 0


                res = EMPTY + (100 if stone and gol else 0) + (10 if stone and not gol else 0)

                row_str += symbols[res] + ""


        lines.append(row_str)

    return "\n".join(lines)


@njit
def fast_territories_numba(board: BoardData, temp: BoardTemp) -> Tuple[int, int]:


    get_territories(board.current_state, temp.temp_visited, temp.temp_state_1, temp.temp_positions_queue)

    territory_black = 0
    territory_white = 0

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if temp.temp_state_1[x,y] == BLACK:
                territory_black +=1
            elif temp.temp_state_1[x,y] == WHITE:
                territory_white += 1

    return territory_black, territory_white

@njit
def clear_board_numba(board: BoardData):
    board.current_state.fill(0)
    board.history_stack.fill(0)
    board.history_hashes.fill(0)

    board.history_ptr[0] = 0
    board.history_hash_ptr[0] = 0
    board.history_count[0] = 0
    board.history_hash_count[0] = 0

@njit()
def undo_board_numba(board: BoardData):

    board.history_ptr[0] = circle_add(board.history_ptr[0], -2, NN_HISTORY)
    board.history_count[0] = board.history_count[0] - 2

    board.history_hash_ptr[0] = circle_add(board.history_hash_ptr[0], -2, MAX_HISTORY)
    board.history_hash_count[0] = board.history_hash_count[0] - 2

    copyto_numba(board.current_state, board.history_stack[board.history_ptr[0]])

@njit
def get_network_input_pytorch_numba(board: BoardData, temp: BoardTemp, current_player_color: int, komi: float) -> np.ndarray:
    """
    Получить входной тензор в формате PyTorch (C, H, W).
    """
    update_network_input_numba(board, temp, current_player_color, komi)

    return temp.tensor

def i32_scalar(v=0):
    return np.array([v], dtype=np.int32)

def build_board_data():
    return BoardData(
            current_state=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
        history_stack=np.zeros((NN_HISTORY, BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
        history_hashes=np.zeros((MAX_HISTORY, 4), dtype=np.uint64),
        zobrist_table=generate_zobrist_table(),
        history_ptr=i32_scalar(0),
        history_hash_ptr=i32_scalar(0),
        history_count=i32_scalar(0),
        history_hash_count=i32_scalar(0))

def build_board_temp():
    return BoardTemp(
            temp_state_1=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
        temp_state_2=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
        temp_hash_1=np.zeros(4, dtype=np.uint64),
        temp_hash_2=np.zeros(4, dtype=np.uint64),
        temp_positions_queue=np.empty((board_size_sqr, 2), dtype=np.int32),
        temp_marked_for_death=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
        temp_visited=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
        tensor=np.zeros(( IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32))

def build_delta_board_array(max_nodes):
    return DeltaBoardArray(next_state=np.zeros((max_nodes, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8),
                    next_hash=np.zeros((max_nodes, 2, 4), dtype=np.uint64),
                    valid=np.zeros(max_nodes, dtype=np.bool_))
