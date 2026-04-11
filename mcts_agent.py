import numpy as np
import math
from typing import List, Tuple
import torch

from config import BOARD_SIZE, board_size_sqr, possible_moves_total, DEEP_DEPTH
from move import decode_move
from game import Game
import time

from noise import DirichletNoiseConfig
from random_network import NetworkBase

import torch.nn.functional as F

_SCORES_ARRAY = np.arange(-board_size_sqr, board_size_sqr + 1, dtype=np.float32)
def get_expected_score(score_logits: np.ndarray) -> float:
    """
    Вычисляет математическое ожидание счета на чистом NumPy (zero-allocation для массива очков).
    """
    # 1. Стабильный Softmax на NumPy
    # Вычитаем максимум для вычислительной стабильности (защита от переполнения exp)
    shifted_logits = score_logits - np.max(score_logits)
    exp_logits = np.exp(shifted_logits)
    score_probs = exp_logits / np.sum(exp_logits)

    # 2. Вычисление математического ожидания
    # np.dot(a, b) работает быстрее, чем np.sum(a * b), и не выделяет память под промежуточный массив
    expected_score = np.dot(score_probs, _SCORES_ARRAY)

    return float(expected_score)


def get_score_utility(expected_score: float, scale: float = 15, max_utility: float = 0.2) -> float:
    """
    Преобразует ожидаемый счет в очках в "Утилиту счета" для MCTS.

    Args:
        expected_score: Ожидаемый счет в очках (например, 5.5).
        scale: Масштаб (в очках), при котором утилита начинает "насыщаться".
               Обычно от 10 до 20 очков.
        max_utility: Максимально возможная добавка к Q-value.

    Returns:
        float: Значение в диапазоне [-max_utility, +max_utility]
    """
    # Используем tanh для плавного сжатия:
    # При счете 0 вернет 0.
    # При счете +10 (и scale=10) вернет 0.76 * max_utility.
    # При счете +50 вернет почти 1.0 * max_utility.
    score_utility = np.tanh(expected_score / scale) * max_utility

    return float(score_utility)


class MCTSNode:
    """
    Узел дерева MCTS.

    Хранит статистику по посещениям, оценкам и вероятностям для состояния игры.
    """

    def __init__(self, game_state: Game, parent=None, parent_action=None, prior_prob=0.0):

        self.game_state = game_state  # Храним только это состояние
        self.parent = parent
        self.parent_action = parent_action
        self.prior_prob = prior_prob

        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

        # ВИРТУАЛЬНЫЕ дочки - только prior probabilities
        self.child_priors = {}  # {action_idx: prior_probability}


        self.children = {}  # {action_idx: MCTSNode} - создаются лениво
        self.expected_score = 0

        self.is_expanded = False
        self.noise_added = False

        self.is_terminal = False

    def reset(self):

        self.parent = None
        self.parent_action = None
        self.prior_prob = 0.0

        self.visit_count = 0
        self.total_value = 0
        self.mean_value = 0.0

        self.child_priors.clear()
        self.children.clear()

        self.expected_score = 0
        self.is_expanded = False
        self.noise_added = False

        self.is_terminal = False

    def is_leaf(self) -> bool:
        """Проверить, является ли узел листом дерева."""
        return not self.is_expanded

    def is_root(self) -> bool:
        """Проверить, является ли узел корнем."""
        return self.parent is None

    def get_ucb_score(self, c_puct: float = 1.25, parent_visit_count: int = 1) -> float:
        """
        Вычислить PUCT (Predictor + UCB applied to Trees) для этого узла.

        Формула AlphaZero:
        U(s,a) = Q(s,a) + C(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Где C(s) = c_puct (упрощенная версия, без log)

        Args:
            c_puct: Коэффициент исследования (exploration constant)
            parent_visit_count: N(s) - количество посещений родительского узла

        Returns:
            UCB score для этого узла
        """
        # Q-value (exploitation)
        q_value = self.mean_value

        # U-value (exploration)
        u_value = c_puct * self.prior_prob * math.sqrt(parent_visit_count) / (1 + self.visit_count)

        return q_value + u_value

    def count_terminal_nodes_in_tree(self):
        sum = 0
        if self.is_terminal:
            return 1
        for i in self.children:
            sum += self.children[i].count_terminal_nodes_in_tree()
        return sum

    def count_nodes(self):
        sum = 1

        for i in self.children:
            sum += self.children[i].count_nodes()

        return sum

    def add_noise(self, noise_config: DirichletNoiseConfig):
        if self.noise_added or noise_config is None:
            return
        self.noise_added = True

        legal_mask = self.game_state.get_legal_moves_mask()
        policy_array = np.zeros(BOARD_SIZE * BOARD_SIZE * 2 + 1, dtype=np.float32)

        policy_array[list(self.child_priors.keys())] = list(self.child_priors.values())

        self.prior_prob = noise_config.add_noise(policy_array, legal_mask)



class MCTS_Agent:
    """
    Агент на основе Monte Carlo Tree Search с нейросетью (или заглушкой).

    Реализует алгоритм AlphaZero MCTS с PUCT для выбора действий.
    """

    def __init__(self, network : NetworkBase, c_puct: float = 1.25,
                 temperature: float = 1.0, noise_config: DirichletNoiseConfig = None, **kwargs):
        """
        Args:
            network: Нейросеть (или RandomNetwork) для оценки позиций
            c_puct: Коэффициент исследования в формуле PUCT
            temperature: Температура для сэмплирования финального хода (1.0 = стохастический, 0.0 = детерминированный)
        """
        self.network = network
        self.c_puct = c_puct
        self.temperature = temperature

        self.noise_config = noise_config or DirichletNoiseConfig.for_selfplay()

        self.root = MCTSNode(Game())
        self.first_root = self.root

        if 'score_goal_factor' in kwargs:
            self.score_goal_factor = kwargs['score_goal_factor']
        else:
            self.score_goal_factor = 1

        self.node_pool = []
        for i in range(DEEP_DEPTH + 1):
            self.node_pool.append(MCTSNode(Game()))


    def flush(self):
        self.recycle_branch(self.root)

        self.root.game_state.clear()

    def set_new_root(self, new_root: MCTSNode):
        if self.root is not None:
            for node in self.root.children.values():
                if node == new_root:
                    continue
                self.recycle_branch(node)

            old_root = self.root
            self.root = new_root

            self.root.parent = None
            self.root.parent_action = None
            self.root.prior_prob = 0.0

            old_root.reset()
            self.node_pool.append(old_root)

    def recycle_branch(self, node: MCTSNode):
        """Рекурсивно возвращает всё поддерево в пул."""
        for child in node.children.values():
            self.recycle_branch(child)

        node.reset()
        self.node_pool.append(node)


    def search(self, root: MCTSNode = None, num_simulations: int = 1600) -> Tuple[int,int,int,int,bool, np.ndarray, MCTSNode]:
        """
        Запустить MCTS поиск для текущего состояния игры.

        Args:
            game: Текущее состояние игры
            num_simulations: Глубина исследования дерева
            root: Опционально для реюза дерева

        Returns:
            (best_move, policy_distribution): Лучший ход и распределение вероятностей

        """

        start_time = time.time()

        # Создаем корневой узел

        if root is None:
            self.root = self.node_pool.pop()
            self.root.game_state.clear()
            self.first_root = self.root
        else:
            self.set_new_root(root)

        if self.root.is_expanded:
            self.root.add_noise(self.noise_config)

        adjusted_simulations = num_simulations - self.root.visit_count
        # Выполняем num_simulations итераций MCTS
        for _ in range(adjusted_simulations):
            node = self.root
            search_path = [node]

            # 1. Selection: спускаемся по дереву, выбирая лучшие действия по PUCT
            while not node.is_leaf() and not node.game_state.game_over:
                node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion: если узел не терминальный, разворачиваем его
            value = 0.0
            if not node.game_state.game_over:
                value = self._expand_node(node)
            else:
                # Терминальный узел - вычисляем реальный результат
                value = self._get_terminal_value(node)
                node.is_terminal = True

            # 3. Backpropagation: распространяем value обратно по пути
            self._backpropagate(search_path, value)

        # Выбираем лучший ход на основе visit counts
        x, y, life_cycle, color, is_pass, policy_distribution, best_node = self._select_action()

        elapsed_time = time.time() - start_time
        print(f"Время поиска лучшего хода search: {elapsed_time:.2f} сек")

        return x, y, life_cycle, color, is_pass, policy_distribution, best_node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Выбор дочернего узла с ленивым созданием.
        """
        best_score = -float('inf')
        best_action_idx = None

        # Вычисляем UCB для всех возможных действий (виртуальных и реальных)
        for action_idx, prior_prob in node.child_priors.items():
            if action_idx in node.children:
                # Реальный дочерний узел - используем его статистику
                child = node.children[action_idx]
                ucb_score = child.get_ucb_score(self.c_puct, node.visit_count)
            else:
                # Виртуальный узел (еще не создан)
                # Q(s,a) = 0 для непосещенного узла
                # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + 0)
                ucb_score = self.c_puct * prior_prob * math.sqrt(node.visit_count + 1)

            if ucb_score > best_score:
                best_score = ucb_score
                best_action_idx = action_idx

        # Если выбранный узел еще не создан - создаем его сейчас
        if best_action_idx not in node.children:
            # Декодируем действие
            x, y, life_cycle, color, is_pass = decode_move(best_action_idx, node.game_state.current_player)

            child_node = self.node_pool.pop()

            success = node.game_state.is_valid_move(x, y, life_cycle, color, is_pass, child_node.game_state)

            if not success:
                # Это не должно происходить, если child_priors правильно заполнен
                raise ValueError(f"Invalid move selected: {x}:{y}{' GoL' if life_cycle == 1 else ''} Pass:{is_pass}")

            child_node.parent = node
            child_node.parent_action = (x, y, life_cycle, color, is_pass)
            child_node.prior_prob = node.child_priors[best_action_idx]

            node.children[best_action_idx] = child_node

        return node.children[best_action_idx]


    def _expand_node(self, node: MCTSNode) -> float:
        """
        Развернуть узел: получить policy и value из нейросети, создать ВИРТУАЛЬНЫЕ дочерние узлы.

        Args:
            node: Узел для разворачивания

        Returns:
            Value оценка этой позиции от нейросети
        """

        start_time = time.time()

        # Получаем входной тензор для нейросети
        state_tensor = node.game_state.get_network_input_pytorch()

        # Предсказание от нейросети: policy (723,) и value (скаляр)
        policy, value, score = self.network.predict(state_tensor)


        # Получаем маску легальных ходов
        legal_mask = node.game_state.get_legal_moves_mask()

        # Применяем маску к policy (обнуляем нелегальные ходы)
        masked_policy = policy * legal_mask

        if node == self.root:
            node.add_noise(self.noise_config)

        # Нормализуем policy
        policy_sum = np.sum(masked_policy)
        if policy_sum > 0:
            masked_policy = masked_policy / policy_sum
        else:
            # Если все вероятности 0, делаем равномерное распределение по легальным ходам
            masked_policy = legal_mask / np.sum(legal_mask)

        # ЛЕНИВОЕ СОЗДАНИЕ: Сохраняем только prior probabilities и future states
        # Физические узлы создадутся в _select_child при необходимости
        for action_idx in range(possible_moves_total):
            if legal_mask[action_idx] > 0:
                node.child_priors[action_idx] = masked_policy[action_idx]


        utility = value

        expected_score = get_expected_score(score)
        node.expected_score = expected_score

        utility += get_score_utility(expected_score) * self.score_goal_factor

        root_score_from_current_perspective = -self.root.expected_score if node.game_state.current_player != self.root.game_state.current_player else self.root.expected_score
        score_delta = expected_score - root_score_from_current_perspective
        utility += get_score_utility(score_delta) * self.score_goal_factor

        node.is_expanded = True

        elapsed = time.time() - start_time
        # print(f"expansion took {elapsed}")

        # Value с точки зрения текущего игрока
        return utility

    def _get_terminal_value(self, node: MCTSNode) -> float:
        """
        Получить value для терминального узла (игра окончена).

        Args:
            node: Терминальный узел

        Returns:
            +1 если текущий игрок выиграл, -1 если проиграл, 0 при ничьей
        """
        score = node.game_state.get_score()
        winner = score['winner']
        current_player = node.game_state.current_player



        score = score['margin_no_komi'] # В пользу победителя всегда
        utility = get_score_utility(score) * self.score_goal_factor

        if winner is None:
            return 0.0  # Ничья
        elif winner == current_player:
            return 1.0 + utility  # Победа
        else:
            return -1.0 - utility # Поражение

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        Распространить value обратно по пути поиска.

        Args:
            search_path: Список узлов от корня до листа
            value: Value для обновления
        """
        for node in reversed(search_path):
            # Инвертируем value для родителя (zero-sum game)
            # Инверсия в начале, потому что кажется, что где-то еще есть другая инверсия, и все рушится. Кажется. Но по идее да.
            value = -value


            node.visit_count += 1
            node.total_value += value
            node.mean_value = node.total_value / node.visit_count

    def _select_action(self) -> Tuple[int,int,int,int,bool, np.ndarray, MCTSNode]:
        """
        Выбрать финальное действие на основе visit counts корневых детей.

        Returns:
            (best_move, policy_distribution): Лучший ход и нормализованное распределение визитов
        """
        # Собираем visit counts для всех дочерних узлов
        visit_counts = np.zeros(possible_moves_total, dtype=np.float32)

        for action_idx, child in self.root.children.items():
            visit_counts[action_idx] = child.visit_count

        # Применяем температуру для сэмплирования
        if self.temperature == 0:
            # Детерминированный выбор (максимум)
            action_idx = np.argmax(visit_counts)
        else:
            # Стохастический выбор с температурой
            visit_counts_temp = visit_counts ** (1.0 / self.temperature)
            policy_distribution = visit_counts_temp / np.sum(visit_counts_temp)
            action_idx = np.random.choice(possible_moves_total, p=policy_distribution)

        # Нормализуем visit counts для возврата (это MCTS-улучшенный policy)
        policy_distribution = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else visit_counts

        # Декодируем action в Move
        x, y, life_cycle, color, is_pass = decode_move(action_idx, self.root.game_state.current_player)
        best_node = self.root.children[action_idx]

        return x, y, life_cycle, color, is_pass, policy_distribution, best_node

    def get_policy(self) -> np.ndarray:
        """
        Получить распределение вероятностей действий после поиска (для обучения).

        Returns:
            Numpy массив (723,) с нормализованными visit counts
        """
        visit_counts = np.zeros(possible_moves_total, dtype=np.float32)

        for action_idx, child in self.root.children.items():
            visit_counts[action_idx] = child.visit_count

        # Нормализуем
        if np.sum(visit_counts) > 0:
            return visit_counts / np.sum(visit_counts)
        else:
            return visit_counts
