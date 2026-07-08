from collections import namedtuple
from typing import Tuple

import numpy as np
from numba import njit

from board import DeltaBoardArray, PRECIESE_PERMIT, LAZY_PERMIT, get_legal_moves_mask_lazy_numba, \
    build_delta_board_array
from config import possible_moves_total, pass_code, BOARD_SIZE, SMALL_TEMP
from game import DeltaGameArray, build_game_data, is_legal_game_move_numba, update_lazy_legal_mask, \
    clear_game_numba, undo_game_numba, get_current_player_numba, apply_delta_game_numba, try_make_game_move_numba, \
    build_delta_game_array
from helper_functions import get_expected_score, get_score_utility

# Структура, которая будет летать внутри @njit функций
TreeData = namedtuple("TreeData", [
    "size",  # [1] int32: аллокатор (сколько узлов занято)

    "root_index",  # np.array([0], dtype=np.int32)
    "free_pool",  # np.empty(max_nodes, dtype=np.int32)
    "free_count",  # np.array([0], dtype=np.int32)

    "parent",  # [max_nodes] int32: индекс родителя (-1 если корень)
    "parent_action",  # [max_nodes] int16: действие, приведшее в узел

    "prior_prob",  # [max_nodes] float32: P(s, a)
    "visit_count",  # [max_nodes] int32: N(s, a)
    "total_value",  # [max_nodes] float32: W(s, a)
    "mean_value",  # [max_nodes] float32: Q(s, a)
    "raw_value",  # [max_nodes] float32
    "u_score",  # [max_nodes] float32
    "u_d_score",  # [max_nodes] float32
    "expected_score",  # [max_nodes] float32

    "is_expanded",  # [max_nodes] bool_
    "is_terminal",  # [max_nodes] bool_

    "noise_added",  # [max_nodes] bool_

    "child_priors",  # [max_nodes, possible_moves] float32: вероятности детей
    "children",  # [max_nodes, possible_moves] int32: индексы детей (-1 если нет)

    "game_state",
    "noise_seed_base",
    "noise_alpha"
])

TreeTemp = namedtuple("TreeTemp", [
    "actions_value", #float32 size actions
    "actions_value_2",
    "actions_count", #int32 size actions
    "actions_mask", #bool size actions
    "path", #int32 size nodes
    "path_actions", #int32 size nodes
    "path_len" # int32 size 1
])

class MCTS_Tree:
    def __init__(self, max_nodes: int, komi_norm: float):
        self.max_nodes = max_nodes
        self.possible_moves = possible_moves_total

        # Выделяем память один раз за всё время жизни агента
        self.data = TreeData(
            size=np.array([0], dtype=np.int32),
            parent=np.zeros(max_nodes, dtype=np.int32),
            parent_action=np.zeros(max_nodes, dtype=np.int16),
            prior_prob=np.zeros(max_nodes, dtype=np.float32),
            visit_count=np.zeros(max_nodes, dtype=np.int32),
            total_value=np.zeros(max_nodes, dtype=np.float32),
            mean_value=np.zeros(max_nodes, dtype=np.float32),
            raw_value=np.zeros(max_nodes, dtype=np.float32),
            u_score=np.zeros(max_nodes, dtype=np.float32),
            u_d_score=np.zeros(max_nodes, dtype=np.float32),
            expected_score=np.zeros(max_nodes, dtype=np.float32),
            is_expanded=np.zeros(max_nodes, dtype=np.bool_),
            is_terminal=np.zeros(max_nodes, dtype=np.bool_),
            noise_added=np.zeros(max_nodes, dtype=np.bool_),
            child_priors=np.zeros((max_nodes, possible_moves_total), dtype=np.float32),
            children=np.zeros((max_nodes, possible_moves_total), dtype=np.int32),
            free_pool=np.zeros(max_nodes, dtype=np.int32),
            free_count=np.array([0], dtype=np.int32),
            root_index=np.array([-1], dtype=np.int32),
            game_state=build_game_data(komi_norm),
            noise_seed_base = np.array([42], dtype=np.int32),
            noise_alpha = np.array([0.25], dtype=np.float32)
        )


        self.delta_games = build_delta_game_array(max_nodes)
        self.delta_boards = build_delta_board_array(max_nodes)

        self.temp = TreeTemp(
            actions_value=np.zeros(possible_moves_total, dtype=np.float32),
            actions_value_2=np.zeros(possible_moves_total, dtype=np.float32),
            actions_count=np.zeros(possible_moves_total, dtype=np.int32),
            actions_mask=np.zeros(possible_moves_total, dtype=np.bool_),
            path = np.zeros(max_nodes, dtype=np.int32),
            path_actions=np.zeros(max_nodes, dtype=np.int32),
            path_len=np.array([0], dtype=np.int32)
        )

    def get_root(self) -> int:
        return self.data.root_index[0]

    @property
    def total_nodes(self) -> int:
        """Эквивалент старого count_nodes() — просто смотрим, сколько аллоцировано"""
        return self.data.size[0]

    @property
    def terminal_nodes(self) -> int:
        """Эквивалент count_terminal_nodes_in_tree()"""
        return count_terminal_numba(self.data)

@njit
def count_terminal_numba(data: TreeData) -> int:
    """Вместо рекурсии просто идем по плоскому массиву до size[0]"""
    size = data.size[0]
    if size == 0:
        return 0
    return np.sum(data.is_terminal[:size])

@njit
def add_dirichlet_noise_numba(
        data: TreeData,
        temp: TreeTemp,
        delta_games: DeltaGameArray,
        node_idx: int,
        frac: float,
        rng: np.random.Generator
) -> None:
    """
    Единый Numba-метод: подсчёт маски → Dirichlet-шум → наложение.
    Dirichlet(α) сэмплируется как Gamma(α,1) / sum(Gamma(α,1)).
    """
    if data.noise_added[node_idx]:
        return

    # --- 1. Строим булеву маску и считаем легальные ходы ---
    num_legal = 0
    for i in range(possible_moves_total):
        if delta_games.next_mask[node_idx, i] > 0:
            temp.actions_mask[i] = True
            num_legal += 1
        else:
            temp.actions_mask[i] = False

    if num_legal == 0:
        return

    # --- 2. Динамическая альфа ---
    alpha = 10.0 / num_legal

    # --- 3. Dirichlet через Gamma-сэмплирование ---
    # Gamma(alpha, 1) для каждого легального хода; остальные — 0
    gamma_sum = 0.0
    for i in range(possible_moves_total):
        if temp.actions_mask[i]:
            g = rng.standard_gamma(alpha)   # Gamma(alpha, scale=1)
            temp.actions_value[i] = g
            gamma_sum += g
        else:
            temp.actions_value[i] = 0.0

    # Нормируем → получаем Dirichlet
    if gamma_sum > 0.0:
        for i in range(possible_moves_total):
            if temp.actions_mask[i]:
                temp.actions_value[i] /= gamma_sum

    # --- 4. Накладываем шум на priors ---
    for i in range(possible_moves_total):
        if not temp.actions_mask[i]:
            continue

        old_p = data.child_priors[node_idx, i]
        if old_p < 0:
            continue

        new_p = (1.0 - frac) * old_p + frac * temp.actions_value[i]
        data.child_priors[node_idx, i] = new_p

        child_idx = data.children[node_idx, i]
        if child_idx != -1:
            data.prior_prob[child_idx] = new_p

    data.noise_added[node_idx] = True

@njit
def get_new_node_numba(data) -> int:
    """Берет индекс из пула, если есть, иначе сдвигает аллокатор."""
    if data.free_count[0] > 0:
        data.free_count[0] -= 1
        return data.free_pool[data.free_count[0]]
    else:
        idx = data.size[0]
        data.size[0] += 1
        return idx

@njit
def add_node_numba(data: TreeData, parent_idx: int, parent_action: int, prior_prob: float) -> int:
    """Создает новый узел и привязывает его к родителю. Эквивалент __init__."""
    idx = get_new_node_numba(data)

    # Инициализация скаляров
    data.parent[idx] = parent_idx
    data.parent_action[idx] = parent_action
    data.prior_prob[idx] = prior_prob

    data.visit_count[idx] = 0
    data.total_value[idx] = 0.0
    data.mean_value[idx] = 0.0
    data.raw_value[idx] = 0.0
    data.u_score[idx] = 0.0
    data.u_d_score[idx] = 0.0
    data.expected_score[idx] = 0.0

    data.is_expanded[idx] = False
    data.is_terminal[idx] = False
    data.noise_added[idx] = False

    # Инициализация векторных полей (срезы 2D массивов)
    data.child_priors[idx].fill(-1.0)
    data.children[idx].fill(-1)

    # Привязка к родителю (если это не корень)
    if parent_idx != -1:
        data.children[parent_idx, parent_action] = idx

    return idx

@njit
def recycle_branch_numba(data, root_to_recycle: int):
    """Итеративно возвращает всё поддерево в пул свободных индексов."""

    # Локальный стек для обхода в глубину (DFS).
    # Numba превращает np.empty внутри njit в сверхбыстрый C malloc.
    # Размер size[0] гарантирует, что стек не переполнится.
    stack = np.empty(data.size[0] + 1, dtype=np.int32)
    stack_ptr = 0

    # Кладем стартовый узел в стек
    stack[stack_ptr] = root_to_recycle
    stack_ptr += 1

    while stack_ptr > 0:
        # Pop
        stack_ptr -= 1
        curr = stack[stack_ptr]

        # 1. Кладем всех детей в стек для последующей очистки
        for a in range(possible_moves_total):
            child_idx = data.children[curr, a]
            if child_idx != -1:
                stack[stack_ptr] = child_idx
                stack_ptr += 1

                # Отвязываем от родителя (на всякий случай)
                data.children[curr, a] = -1

        # 2. Кидаем сам узел в пул свободных индексов
        data.free_pool[data.free_count[0]] = curr
        data.free_count[0] += 1


@njit
def set_new_root_numba(data: TreeData, new_root_idx: int):
    """
    Отсекает все ветки старого корня, кроме выбранной,
    переводит корень на новый индекс.
    """
    old_root_idx = data.root_index[0]
    # 1. Идем по всем детям старого корня
    for a in range(possible_moves_total):
        child_idx = data.children[old_root_idx, a]

        if child_idx == new_root_idx:
            # Наш победитель, его не трогаем
            continue
        elif child_idx != -1:
            # Отправляем в мясорубку всё поддерево
            recycle_branch_numba(data, child_idx)

    # 2. Убиваем сам старый корень
    data.free_pool[data.free_count[0]] = old_root_idx
    data.free_count[0] += 1

    # 3. Делаем new_root_idx новым корнем дерева
    data.root_index[0] = new_root_idx
    data.parent[new_root_idx] = -1
    data.parent_action[new_root_idx] = -1
    data.prior_prob[new_root_idx] = 0.0

@njit
def select_child_numba(data: TreeData, temp: TreeTemp, delta_games: DeltaGameArray, node_idx: int, c_puct: float, q_init_loss: float):
    """Вычисляет PUCT и возвращает лучший ход за O(N ходов) без создания Python-объектов."""

    temp.actions_value.fill(-np.inf)

    parent_visits_sqrt = np.sqrt(data.visit_count[node_idx] + 1.0)
    for a in range(possible_moves_total):
        prior = data.child_priors[node_idx, a]
        if prior < 0:
            continue


        # Легальность хода берем из маски (не дергая тяжелый game_state)
        if data.game_state.legal_mask[a] == 0:
            data.child_priors[node_idx, a] = -np.inf
            continue

        child_idx = data.children[node_idx, a]

        if child_idx != -1:
            # Узел существует - используем статистику
            q_value = -data.mean_value[child_idx]
            u_value = c_puct * prior * parent_visits_sqrt / (1.0 + data.visit_count[child_idx])
        else:
            # Виртуальный узел - Q берем из родительской сети
            q_value = data.raw_value[node_idx] - q_init_loss
            u_value = c_puct * prior * parent_visits_sqrt

        score = q_value + u_value

        temp.actions_value[a] = score

    best_action_idx = -1

    while True:
        # Находим индекс максимального элемента (O(N) на С)
        action_idx = np.argmax(temp.actions_value)

        # Если максимум -inf, кандидаты закончились
        if temp.actions_value[action_idx] == -np.inf:
            break

        # Проверяем легальность
        if is_legal_game_move_numba(data.game_state, action_idx, delta_games, node_idx):
            best_action_idx = action_idx
            break
        else:
            # Маскируем нелегальный ход, чтобы argmax его больше не нашел
            temp.actions_value[action_idx] = -np.inf
            data.child_priors[node_idx, action_idx] = -np.inf

    if best_action_idx == -1:
        print("No valid actions at all!")
        best_action_idx = pass_code

    #print(f"selected {best_action_idx} child of {node_idx}")

    return best_action_idx

@njit
def expand_node_from_network_result_numba(
    data: TreeData,
    temp: TreeTemp,
    delta_games: DeltaGameArray,
    node_idx: int,
    root_player: int,
    score_goal_factor: float,
    policy: np.ndarray,
    raw_value: float,
    score: np.ndarray,
):
    """
    NumPy/Numba-версия expand_node_from_network_result.

    Требования к temp:
        temp.masked_policy : float32[possible_moves_total]

    Предполагается, что:
        - get_legal_moves_mask_lazy_numba(...) лениво дополняет data.game_state.legal_mask
        - get_expected_score_numba(score) -> float
        - get_score_utility_numba(x) -> float
    """

    # 1) Лениво доуточняем legal mask для текущего game_state
    update_lazy_legal_mask(data.game_state)
    delta_games.next_mask[node_idx, :] = data.game_state.legal_mask

    # 2) Применяем маску к policy и считаем сумму / число легальных ходов
    policy_sum = 0.0
    legal_count = 0

    for a in range(possible_moves_total):
        if data.game_state.legal_mask[a] != 0:
            p = policy[a]
            if p < 0.0:
                p = 0.0
            temp.actions_value[a] = p
            policy_sum += p
            legal_count += 1
        else:
            temp.actions_value[a] = 0.0

    # 3) Нормализация
    if policy_sum > 0.0:
        inv_sum = 1.0 / policy_sum
        for a in range(possible_moves_total):
            temp.actions_value[a] *= inv_sum
    else:
        # fallback: равномерно по легальным
        if legal_count > 0:
            uniform_p = 1.0 / legal_count
            for a in range(possible_moves_total):
                if data.game_state.legal_mask[a] != 0:
                    temp.actions_value[a] = uniform_p
                else:
                    temp.actions_value[a] = 0.0
            any_legal = True
        else:
            # Вообще нет легальных ходов
            for a in range(possible_moves_total):
                temp.actions_value[a] = 0.0
            any_legal = False
        print(f"Resort to fallback on expansion where root {data.root_index[0]} at node {node_idx} w legals {legal_count}")
        #print(f"policy min={int(policy.min())} max={int(policy.max())} sum={int(policy.sum())} legal count {legal_count}")

    # 4) Сохраняем child priors
    for a in range(possible_moves_total):
        if data.game_state.legal_mask[a] != 0:
            data.child_priors[node_idx, a] = temp.actions_value[a]
        else:
            data.child_priors[node_idx, a] = -np.inf

    # 5) Value / score utility
    expected_score = get_expected_score(score)
    data.expected_score[node_idx] = expected_score

    u_score = get_score_utility(expected_score) * score_goal_factor

    root_expected_score = data.expected_score[data.root_index[0]]
    if get_current_player_numba(data.game_state) != root_player:
        root_score_from_current_perspective = -root_expected_score
    else:
        root_score_from_current_perspective = root_expected_score

    score_delta = expected_score - root_score_from_current_perspective
    u_d_score = get_score_utility(score_delta) * score_goal_factor

    utility = raw_value + u_score + u_d_score

    data.raw_value[node_idx] = raw_value
    data.u_score[node_idx] = u_score
    data.u_d_score[node_idx] = u_d_score
    data.is_expanded[node_idx] = True

    #print(f"Expansion ended of node {node_idx} {data.is_expanded[node_idx]}")

    return utility



@njit
def select_leaf_numba(data: TreeData, temp: TreeTemp, delta_games: DeltaGameArray, delta_boards: DeltaBoardArray, c_puct: float, q_init_loss: float) -> int:
    """
    Спускается по дереву до листа. Лениво создает узел, если мы выбрали виртуальный.
    Возвращает путь узлов и путь действий для синхронизации game_state.
    """
    # Стек для сохранения пути (максимум size[0], так что берем с запасом)
    path_nodes = temp.path
    path_actions = temp.path_actions

    path_nodes[0] = data.root_index[0]
    path_len = 1
    curr = data.root_index[0]

    # Спускаемся, пока узел раскрыт (is_leaf == False) и не терминальный
    #print(f"begin selection from {curr} with expanded {data.is_expanded[curr]} and terminal {data.is_terminal[curr]}")

    while data.is_expanded[curr] and not data.is_terminal[curr]:

        best_action = select_child_numba(data, temp, delta_games, curr, c_puct, q_init_loss)

        child_idx = data.children[curr, best_action]

        #print(f"{curr} has best action {best_action} leading to {child_idx}")

        # Ленивое создание (аналог node_pool.pop() + child.parent = node)
        if child_idx == -1:
            prior = data.child_priors[curr, best_action]

            child_idx = add_node_numba(data, curr, best_action, prior)
            data.children[curr, best_action] = child_idx

            try_make_game_move_numba(data.game_state, delta_boards, delta_games, child_idx, best_action, False)
        else:
            apply_delta_game_numba(data.game_state, delta_boards, delta_games, child_idx)

        # Записываем шаг
        path_actions[path_len - 1] = best_action
        curr = child_idx
        path_nodes[path_len] = curr
        path_len += 1

    temp.path_len[0] = path_len

    #print(f"selected {curr} leaf")

    return curr

@njit()
def backpropagate_numba(tree: TreeData, temp: TreeTemp, delta_games: DeltaGameArray, value: float):
    """
    Распространить value обратно по пути поиска.

    Args:
        search_path: Список узлов от корня до листа
        value: Value для обновления
    """


    skips = 1

    #print(f"backprop begin with path len {temp.path_len[0]}")

    for i in range(temp.path_len[0] - 1, -1, -1):

        tree.visit_count[temp.path[i]] += 1
        tree.total_value[temp.path[i]] += value
        tree.mean_value[temp.path[i]] = tree.total_value[temp.path[i]] / tree.visit_count[temp.path[i]]

        value = -value


        if skips <= 0:
            undo_game_numba(tree.game_state)
        skips -= 1

    tree.game_state.legal_mask[:] = delta_games.next_mask[tree.root_index[0], :]
    tree.game_state.has_mask[0] = True

@njit
def reset_tree_numba(data: TreeData):
    data.size[0] = 0
    data.free_count[0] = 0
    data.root_index[0] = -1
    clear_game_numba(data.game_state)

@njit
def begin_search_numba(
    data: TreeData,
    delta_games: DeltaGameArray,
    root_idx: int         # -1 => новый поиск с пустого дерева
):
    """
    Возвращает:
        root_idx, temperature, root_player, seed, need_noise
    """

    if root_idx == -1:

        # Новый поиск
        reset_tree_numba(data)

        root_idx = add_node_numba(data, -1, -1, 0.0)
        data.root_index[0] = root_idx

        update_lazy_legal_mask(data.game_state)

        for a in range(possible_moves_total):
            delta_games.next_mask[root_idx, a] = data.game_state.legal_mask[a]

        delta_games.move[root_idx] = -1

    else:
        # Реюз дерева
        set_new_root_numba(data, root_idx)



@njit
def select_action_numba(
        data: TreeData,
        temp: TreeTemp,
        temperature: float,
        rng: np.random.Generator
) -> Tuple[int, int]:
    """
    Выбирает действие из корня (root_index[0]).
    Возвращает:
        (action_idx, policy_distribution, best_node_idx)
    """

    root_idx = data.root_index[0]

    # Аллоцируем массивы: visit_counts (временный) и policy_distribution (для возврата в Python)
    visit_counts = temp.actions_count
    policy_distribution = temp.actions_value

    # 1. Собираем visit counts для всех дочерних узлов
    total_visits = 0.0
    for action_idx in range(possible_moves_total):
        visit_counts[action_idx] = 0
        policy_distribution[action_idx] = 0

        child_idx = data.children[root_idx, action_idx]
        if child_idx != -1:
            vc = data.visit_count[child_idx]
            visit_counts[action_idx] = vc
            total_visits += vc

    # 2. Нормализуем visit counts для возврата
    if total_visits > 0:
        for i in range(possible_moves_total):
            policy_distribution[i] = visit_counts[i] / total_visits
    else:
        for i in range(possible_moves_total):
            policy_distribution[i] = visit_counts[i]

    # 3. Применяем температуру для сэмплирования
    if temperature <= SMALL_TEMP:
        # Детерминированный выбор
        action_idx = np.argmax(visit_counts)
    else:
        # Стохастический выбор (Log-Sum-Exp trick)
        max_logit = -np.inf

        # Считаем логиты и находим максимум за один проход, сохраняем в temp.actions_value
        for i in range(possible_moves_total):
            if visit_counts[i] > 0:
                logit = np.log(visit_counts[i]) / temperature
                temp.actions_value_2[i] = logit
                if logit > max_logit:
                    max_logit = logit
            else:
                temp.actions_value_2[i] = -np.inf

        # Возводим в экспоненту и сразу считаем сумму
        sum_exp = 0.0
        for i in range(possible_moves_total):
            if visit_counts[i] > 0:
                # x - max_logit гарантирует, что e^x <= 1.0 (стабильно)
                exp_val = np.exp(temp.actions_value_2[i] - max_logit)
                temp.actions_value_2[i] = exp_val
                sum_exp += exp_val
            else:
                temp.actions_value_2[i] = 0.0

        # Нормализация и выбор рулеткой (Roulette Wheel Selection)
        # Умножаем случайное [0..1) на сумму, чтобы избежать деления всего массива
        rand_val = rng.random() * sum_exp

        action_idx = -1
        cumulative = 0.0
        for i in range(possible_moves_total):
            if temp.actions_value_2[i] > 0:
                cumulative += temp.actions_value_2[i]
                if rand_val <= cumulative:
                    action_idx = i
                    break

        # Fallback (защита от багов плавающей точки из-за округлений, если rand_val ~ sum_exp)
        if action_idx == -1:
            action_idx = np.argmax(visit_counts)

    # 4. Достаем индекс выбранного дочернего узла
    best_node_idx = data.children[root_idx, action_idx]

    #print("==========================")
    #print(f"Selected best action {action_idx} leading to node {best_node_idx} with VC {data.visit_count[best_node_idx]}")

    return action_idx, best_node_idx