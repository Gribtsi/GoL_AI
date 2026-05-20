import numpy as np
import math
from typing import List, Tuple

from board import get_opponent, PRECIESE_PERMIT
from config import BOARD_SIZE, board_size_sqr, possible_moves_total, DEEP_DEPTH, EMPTY, TEMP_DECAY_TURN, \
    TEMP_DECAY_COEFF, C_PUCT, SMALL_TEMP
from move import decode_move
from game import Game, DeltaGame
import time

from noise import DirichletNoiseConfig
from random_network import NetworkBase

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


def get_score_utility(expected_score: float, scale: float = 15, max_utility: float = 0.125) -> float:
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

    def __init__(self, parent=None, parent_action=None, prior_prob=0.0):

        #self.game_state = game_state  # Храним только это состояние

        self.delta_game = DeltaGame()

        self.parent = parent
        self.parent_action = parent_action
        self.prior_prob = prior_prob

        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

        self.raw_value = 0.0
        self.u_score = 0.0
        self.u_d_score = 0.0

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

        self.raw_value = 0.0
        self.u_score = 0.0
        self.u_d_score = 0.0

        self.child_priors.clear()
        self.children.clear()

        self.delta_game.next_mask.fill(0)

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

    def add_noise(self, noise_config: DirichletNoiseConfig, game_state: Game, seed_base : int):
        if self.noise_added or noise_config is None:
            return
        self.noise_added = True

        mask = np.zeros(possible_moves_total, dtype=np.float32)
        game_state.get_legal_moves_mask_preciese(mask)

        noise = noise_config.get_noise(mask, seed_base + game_state.current_move)
        frac = noise_config.exploration_fraction

        for action_idx in list(self.child_priors.keys()):
            old_p = self.child_priors[action_idx]
            new_p = (1.0 - frac) * old_p + frac * noise[action_idx]
            self.child_priors[action_idx] = float(new_p)

            if action_idx in self.children:
                self.children[action_idx].prior_prob = float(new_p)


class MCTS_Agent:
    """
    Агент на основе Monte Carlo Tree Search с нейросетью (или заглушкой).

    Реализует алгоритм AlphaZero MCTS с PUCT для выбора действий.
    """

    def set_network(self, network : NetworkBase):
        self.network = network

    def has_network(self):
        return self.network is not None

    def set_random_seed(self, seed : int):
        self.random_seed_base = seed

    def __init__(self, network : NetworkBase = None, c_puct: float = C_PUCT,
                 temperature: float = 1.0, noise_config: DirichletNoiseConfig = None, **kwargs):
        """
        Args:
            network: Нейросеть (или RandomNetwork) для оценки позиций
            c_puct: Коэффициент исследования в формуле PUCT
            temperature: Температура для сэмплирования финального хода (1.0 = стохастический, 0.0 = детерминированный)
        """
        self.network = network
        self.c_puct = c_puct

        self.initial_temperature = temperature
        self.temperature = temperature

        self.random_seed_base = 42

        self.game_state : Game = Game()

        self.noise_config = noise_config or DirichletNoiseConfig.for_selfplay()

        self.root = MCTSNode()
        self.first_root = self.root

        if 'score_goal_factor' in kwargs:
            self.score_goal_factor = kwargs['score_goal_factor']
        else:
            self.score_goal_factor = 1

        self.ucb_scores_cache = np.zeros(possible_moves_total, dtype=np.float32)
        self.legal_mask_cache = np.zeros(possible_moves_total, dtype=np.float32)
        self.masked_policy_cache = np.zeros(possible_moves_total, dtype=np.float32)

        self.node_pool = []
        for i in range(DEEP_DEPTH + 1):
            self.node_pool.append(MCTSNode(Game()))

        self.root_player = EMPTY


    def flush(self):
        self.recycle_branch(self.root)

        self.game_state.clear()

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


    def search(self, root: MCTSNode = None, num_simulations: int = DEEP_DEPTH) -> Tuple[int, np.ndarray, MCTSNode]:
        """
        Запустить MCTS поиск для текущего состояния игры.

        Args:
            game: Текущее состояние игры
            num_simulations: Глубина исследования дерева
            root: Опционально для реюза дерева

        Returns:
            (best_move, policy_distribution): Лучший ход и распределение вероятностей

        """

        #start_time = time.time()

        # Создаем корневой узел

        if root is None:
            self.root = self.node_pool.pop()
            self.game_state.clear()
            self.first_root = self.root

            self.game_state.update_legal_moves_mask()
            np.copyto(self.root.delta_game.next_mask, self.game_state.legal_mask)
        else:
            self.set_new_root(root)

        self.temperature = self.initial_temperature
        if self.game_state.current_move > TEMP_DECAY_TURN:
            for i in range(self.game_state.current_move - TEMP_DECAY_TURN):
                self.temperature *= TEMP_DECAY_COEFF

        self.root_player = self.game_state.current_player

        if self.root.is_expanded:
            self.root.add_noise(self.noise_config, self.game_state, self.random_seed_base)

        adjusted_simulations = num_simulations - self.root.visit_count
        # Выполняем num_simulations итераций MCTS
        for _ in range(adjusted_simulations):
            node = self.root
            search_path = [node]

            # 1. Selection: спускаемся по дереву, выбирая лучшие действия по PUCT
            while not node.is_leaf() and not self.game_state.game_over:
                node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion: если узел не терминальный, разворачиваем его
            if not self.game_state.game_over:
                value = self._expand_node(node)
            else:
                # Терминальный узел - вычисляем реальный результат
                value = self._get_terminal_value(node)
                node.is_terminal = True

            # 3. Backpropagation: распространяем value обратно по пути
            self._backpropagate(search_path, value)

        # Выбираем лучший ход на основе visit counts
        encoded_move, policy_distribution, best_node = self._select_action()



        #elapsed_time = time.time() - start_time
        #print(f"Время поиска лучшего хода search: {elapsed_time:.2f} сек")

        return encoded_move, policy_distribution, best_node

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Выбор дочернего узла с ленивым созданием.

        Предполагаем, что self.game_state находится в узле node

        """

        self.ucb_scores_cache.fill(-np.inf)

        # Вычисляем UCB для всех возможных действий (виртуальных и реальных)
        for action_idx, prior_prob in node.child_priors.items():

            if prior_prob <= 0:
                continue

            if action_idx in node.children:
                # Реальный дочерний узел - используем его статистику
                child = node.children[action_idx]
                q_value = -child.mean_value
                u_value = self.c_puct * child.prior_prob * math.sqrt(node.visit_count + 1) / (1 + child.visit_count)
                ucb_score = q_value + u_value

            else:
                # Виртуальный узел (еще не создан)
                # Q(s,a) = 0 для непосещенного узла
                # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + 0)
                ucb_score = self.c_puct * prior_prob * math.sqrt(node.visit_count + 1)

            self.ucb_scores_cache[action_idx] = ucb_score

        best_action_idx = None

        while True:
            # Находим индекс максимального элемента (O(N) на С)
            action_idx = np.argmax(self.ucb_scores_cache)

            # Если максимум -inf, кандидаты закончились
            if self.ucb_scores_cache[action_idx] == -np.inf:
                break

            # Проверяем легальность
            if self.game_state.is_valid_move(action_idx, node.delta_game.next_mask):
                best_action_idx = action_idx
                break
            else:
                # Маскируем нелегальный ход, чтобы argmax его больше не нашел
                self.ucb_scores_cache[action_idx] = -np.inf
                node.child_priors[action_idx] = -float('inf')

        if best_action_idx is None:
            raise ValueError(f"No valid moves in game!")

        # Если выбранный узел еще не создан - создаем его сейчас
        if best_action_idx not in node.children:
            # Декодируем действие

            child_node = self.node_pool.pop()

            success = self.game_state.make_move(best_action_idx, child_node.delta_game, False)

            #success = node.game_state.is_valid_move(x, y, life_cycle, color, is_pass, child_node.game_state)

            if not success:
                # Это не должно происходить, если child_priors правильно заполнен
                x,y,life_cycle,is_pass,is_swap = decode_move(best_action_idx)
                raise ValueError(f"Invalid move selected: {x}:{y}{' GoL' if life_cycle == 1 else ''}{' Pass' if is_pass else ''}{' Swap' if is_swap else ''}")

            child_node.parent = node
            child_node.parent_action = best_action_idx
            child_node.prior_prob = node.child_priors[best_action_idx]

            node.children[best_action_idx] = child_node
        else:
            self.game_state.apply_delta(node.children[best_action_idx].delta_game)

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
        state_tensor = self.game_state.get_network_input_pytorch()

        # Предсказание от нейросети: policy (723,) и value (скаляр)
        policy, raw_value, score = self.network.predict(state_tensor)

        # Получаем маску легальных ходов
        self.game_state.get_legal_moves_mask_lazy(self.legal_mask_cache)

        # Применяем маску к policy (обнуляем нелегальные ходы)
        np.multiply(policy, self.legal_mask_cache, out=self.masked_policy_cache)

        # Нормализуем policy
        policy_sum = np.sum(self.masked_policy_cache)
        if policy_sum > 0:
            masked_policy = self.masked_policy_cache / policy_sum
        else:
            # Если все вероятности 0, делаем равномерное распределение по легальным ходам
            np.copyto(self.masked_policy_cache, self.legal_mask_cache)
            self.masked_policy_cache /= np.sum(self.masked_policy_cache)

        # ЛЕНИВОЕ СОЗДАНИЕ: Сохраняем только prior probabilities и future states
        # Физические узлы создадутся в _select_child при необходимости
        for action_idx in range(possible_moves_total):
            if self.legal_mask_cache[action_idx] > 0:
                node.child_priors[action_idx] = self.masked_policy_cache[action_idx]

        if node == self.root:
            node.add_noise(self.noise_config, self.game_state, self.random_seed_base)

        utility = raw_value

        expected_score = get_expected_score(score)
        node.expected_score = expected_score


        utility += get_score_utility(expected_score) * self.score_goal_factor

        root_score_from_current_perspective = -self.root.expected_score if self.game_state.current_player != self.root_player else self.root.expected_score
        score_delta = expected_score - root_score_from_current_perspective
        utility += get_score_utility(score_delta) * self.score_goal_factor

        node.raw_value = raw_value
        node.u_score = get_score_utility(expected_score) * self.score_goal_factor
        node.u_d_score = get_score_utility(score_delta) * self.score_goal_factor

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

        winner, score = self.game_state.get_winner_and_margin_fast()

        utility = get_score_utility(score) * self.score_goal_factor  # В пользу победителя всегда

        if winner == EMPTY:
            return 0.0  # Ничья
        elif winner == self.game_state.current_player:
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

        skips = 1
        for node in reversed(search_path):

            node.visit_count += 1
            node.total_value += value
            node.mean_value = node.total_value / node.visit_count

            value = -value


            if skips <= 0:
                self.game_state.undo()
            skips -= 1

        np.copyto(self.game_state.legal_mask, self.root.delta_game.next_mask)
        self.game_state.has_mask = True


    def _select_action(self) -> Tuple[int, np.ndarray, MCTSNode]:
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
        if self.temperature <= SMALL_TEMP:
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
        best_node = self.root.children[action_idx]

        return action_idx, policy_distribution, best_node
