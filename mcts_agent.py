import numpy as np
import math
from typing import Optional, List, Tuple, Union
import torch
from IPython.lib.guisupport import start_event_loop_wx
from networkx import sedgewick_maze_graph
from torch.utils.checkpoint import set_checkpoint_debug_enabled
from torchgen.dest.ufunc import eligible_for_binary_scalar_specialization

from board import Board, Move, BOARD_SIZE, BLACK
from game import Game, decode_action
import time

from noise import DirichletNoiseConfig
from random_network import NetworkBase

board_size_sqr = BOARD_SIZE * BOARD_SIZE
possible_moves_total = board_size_sqr * 2 + 1

import torch.nn.functional as F
def get_expected_score(score_logits) -> float:
    score_probs = F.softmax(score_logits, dim=0).squeeze()
    scores = torch.arange(-board_size_sqr,  board_size_sqr + 1, device=score_logits.device, dtype=torch.float32)
    expected_score = torch.sum(score_probs * scores).item()
    return expected_score


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


class MCTS_Agent:
    """
    Агент на основе Monte Carlo Tree Search с нейросетью (или заглушкой).

    Реализует алгоритм AlphaZero MCTS с PUCT для выбора действий.
    """

    def __init__(self, network : NetworkBase, num_simulations: int = 1600, c_puct: float = 1.25,
                 temperature: float = 1.0, noise_config: DirichletNoiseConfig = None, **kwargs):
        """
        Args:
            network: Нейросеть (или RandomNetwork) для оценки позиций
            num_simulations: Количество симуляций MCTS на ход
            c_puct: Коэффициент исследования в формуле PUCT
            temperature: Температура для сэмплирования финального хода (1.0 = стохастический, 0.0 = детерминированный)
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        self.noise_config = noise_config or DirichletNoiseConfig.for_selfplay()

        self.root : Union[None, MCTSNode] = None
        self.first_root = None

        if 'score_goal_mod' in kwargs:
            self.score_goal_mod = kwargs['score_goal_mod']
        else:
            self.score_goal_mod = 1

    def search(self, game: Game, root: MCTSNode = None) -> Tuple[Move, np.ndarray, MCTSNode]:
        """
        Запустить MCTS поиск для текущего состояния игры.

        Args:
            game: Текущее состояние игры

        Returns:
            (best_move, policy_distribution): Лучший ход и распределение вероятностей
        """

        start_time = time.time()

        # Создаем корневой узел
        if root is None:
            self.root = MCTSNode(game)
            self.first_root = self.root
        else:
            self.root = root
            root.parent = None

        adjusted_simulations = self.num_simulations - self.root.visit_count
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
        best_move, policy_distribution, best_node = self._select_action()

        elapsed_time = time.time() - start_time
        print(f"Время поиска лучшего хода search: {elapsed_time:.2f} сек")

        return best_move, policy_distribution, best_node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Выбор дочернего узла с ЛЕНИВЫМ созданием.
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

        # Если выбранный узел еще не создан - создаем его СЕЙЧАС
        if best_action_idx not in node.children:
            # Декодируем действие
            move = decode_action(best_action_idx, node.game_state.current_player)

            # КРИТИЧНО: Клонируем и применяем ход только для ОДНОГО выбранного дочернего узла
            success, future_game = node.game_state.is_valid_move(move)

            if not success:
                # Это не должно происходить, если child_priors правильно заполнен
                raise ValueError(f"Invalid move selected: {move}")

            # Создаем физический узел
            child_node = MCTSNode(
                future_game,
                parent=node,
                parent_action=move,
                prior_prob=node.child_priors[best_action_idx]
            )
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
        state_tensor = node.game_state.get_network_input_pytorch(32)

        # Предсказание от нейросети: policy (723,) и value (скаляр)
        policy, value, score = self.network.predict(state_tensor)


        # Получаем маску легальных ходов
        legal_mask = node.game_state.get_legal_moves_mask_simple()

        # Применяем маску к policy (обнуляем нелегальные ходы)
        masked_policy = policy * legal_mask

        if node == self.first_root:
            masked_policy = self.noise_config.add_noise(masked_policy, legal_mask)

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

        utility += get_score_utility(expected_score) * self.score_goal_mod

        root_score_from_current_perspective = -self.root.expected_score if node.game_state.current_player != self.root.game_state.current_player else self.root.expected_score
        score_delta = expected_score - root_score_from_current_perspective
        utility += get_score_utility(score_delta) * self.score_goal_mod

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
        utility = get_score_utility(score) * self.score_goal_mod

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


    def _encode_move(self, move: Move) -> int:
        """
        Кодировать Move в индекс действия.

        Args:
            move: Объект Move

        Returns:
            Индекс действия (0-722)
        """
        if move.is_pass():
            return BOARD_SIZE * BOARD_SIZE * 2
        else:
            x, y = move.position
            position_idx = x * BOARD_SIZE + y
            return position_idx + move.life_cycles * BOARD_SIZE * BOARD_SIZE



    def _select_action(self) -> Tuple[Move, np.ndarray, MCTSNode]:
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
        best_move = decode_action(action_idx, self.root.game_state.current_player)
        best_node = self.root.children[action_idx]

        return best_move, policy_distribution, best_node

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
