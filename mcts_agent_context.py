import time

import numpy as np

from config import DEEP_DEPTH, MEAN_KOMI, BAD_VALUES_COUNT, EMPTY, MAX_KOMI, possible_moves_total, C_PUCT, Q_INIT_LOSS
from game import try_make_game_move_numba, apply_delta_game_numba, get_network_input_from_game, \
    get_current_player_numba, get_winner_and_margin_numba
from game_rules import GameRules
from helper_functions import generate_komi, all_less_than_threshold, get_fast_finish_simulations_and_chance, \
    get_mean_value
from mcts_tree import MCTS_Tree, reset_tree_numba, add_node_numba, begin_search_numba, \
    expand_node_from_network_result_numba, backpropagate_numba, add_dirichlet_noise_numba, select_leaf_numba, \
    select_action_numba
from random_network import AsyncNetworkBase


class AgentContext:
    """Хранит состояние одной сессии self-play."""

    def __init__(self, process_id: int, agent_index: int, network_client: AsyncNetworkBase, rules: GameRules):

        self.index = agent_index
        self.client = network_client
        self.rules = rules

        self.tree = MCTS_Tree(self.rules.deep_depth + 10, MEAN_KOMI)

        self.history = []
        self.komi = 0
        self.till_the_end = False
        self.is_deep = False
        self.depth_target = 0
        self.values_cache = np.zeros(BAD_VALUES_COUNT * 2, dtype=np.float32)
        self.values_cache_ptr = 0

        self.current_seed_base = 0
        self.rng = None

        # Состояние MCTS
        self.node_to_expand = 0
        self.simulations_done = 0
        self.root_player = EMPTY
        self.temperature = 1.0
        self.best_node = -1
        self.waiting_for_network = False
        self.game_started_at = 0
        self.early_termination = False

    def init_new_game(self, komi_offset=0):
        self.game_started_at = time.time()

        reset_tree_numba(self.tree.data)

        self.rng = np.random.default_rng()
        self.current_seed_base = int(self.rng.integers(0, 2 ** 30))
        self.rng = np.random.default_rng(self.current_seed_base)

        if self.rules.fixed_komi is not None:
            # турнир: фиксированное коми
            self.komi = self.rules.fixed_komi
            flat_komi = False
        else:
            # self-play: случайное коми
            self.komi, flat_komi = generate_komi(komi_offset, self.rng)

        self.tree.data.game_state.komi_norm[0] = self.komi / MAX_KOMI

        #print(f"start game {self.tree.data.game_state.komi_norm[0]}")
        self.till_the_end = flat_komi or (
            self.rng.uniform(0, 1) <= self.rules.until_the_end_chance
        )

        self.temperature = self.rules.initial_temp
        self.history = []



        self.values_cache = np.zeros(BAD_VALUES_COUNT * 2, dtype=np.float32)
        self.values_cache_ptr = 0
        self.best_node = -1

        self.early_termination = False

        self.start_new_turn()

    def update_rules(self, rules: GameRules):
        """ТМ может передать новые правила перед следующим матчем."""
        self.rules = rules
        # пересоздаём дерево только если размер изменился
        if self.tree.data.visit_count.shape[0] != rules.deep_depth + 10:
            self.tree = MCTS_Tree(rules.deep_depth + 10, rules.mean_komi)

    def apply_a_move(self, encoded_move: int = -1):

        if not (0 <= encoded_move < possible_moves_total):
            raise Exception("Applying invalid move")

        self.write_current_to_history(encoded_move=encoded_move, deep_write=False)

        #Тут мы кароче смотрим, раскрыт ли наш будущий корень. Если да - просто идем туда, иначе - принудительно раскрываем и идем туда
        root_idx = self.tree.data.root_index[0]
        child = self.tree.data.children[root_idx, encoded_move]
        if child == -1:
            # Ребенка нет - форсим создание ребенка
            curr = self.tree.data.root_index[0]

            prior = self.tree.data.child_priors[curr, encoded_move]

            child_idx = add_node_numba(self.tree.data, curr, encoded_move, prior)
            self.tree.data.children[curr, encoded_move] = child_idx

            try_make_game_move_numba(self.tree.data.game_state, self.tree.delta_boards, self.tree.delta_games, child_idx, encoded_move, False)

            child = child_idx

        # Ребенок есть - делаем все то же, что и при обычном поиске - двигаем дерево вниз на 1
        else:
            apply_delta_game_numba(self.tree.data.game_state, self.tree.delta_boards, self.tree.delta_games, child)

        #print(f"move applied with {self.client.name}")
        self.start_new_turn(child)


    def write_current_to_history(self, encoded_move: int, deep_write: bool = True):
        rules = self.rules


        if deep_write:
            self.history.append({
                'state_tensor': get_network_input_from_game(self.tree.data.game_state).copy(),
                'mcts_policy': self.tree.temp.actions_value.copy(),
                'player': self.root_player,
                'move': encoded_move,
                'deep_search': self.is_deep
            })
        else:
            self.history.append(
                {
                    'move': encoded_move
                }
            )

        if deep_write:
            self.values_cache[self.values_cache_ptr] = self.tree.data.raw_value[self.tree.data.root_index[0]]
        else:
            self.values_cache[self.values_cache_ptr] = 0

        # print("checking concede")

        if (rules.use_early_resign
                and not self.till_the_end
                and (self.tree.data.game_state.current_move[0] > rules.min_turns_before_resign)
                and all_less_than_threshold(rules.concede_at, self.values_cache, self.values_cache_ptr)):
            self.tree.data.game_state.game_over[0] = True
        elif (rules.use_fast_finish
              and not self.till_the_end
              and all_less_than_threshold(rules.fast_finish_at, self.values_cache, self.values_cache_ptr)):
            self.early_termination = True

        self.values_cache_ptr = (self.values_cache_ptr + 1) % (BAD_VALUES_COUNT * 2)



    def start_new_turn(self, root_idx: int = -1):
        rules = self.rules

        if self.early_termination:
            deep_depth, deep_search_chance = get_fast_finish_simulations_and_chance(
                get_mean_value(self.values_cache, self.values_cache_ptr))
        else:
            deep_depth        = rules.deep_depth
            deep_search_chance = rules.deep_search_chance

        self.is_deep      = not rules.use_pcr or (self.rng.uniform(0, 1) <= deep_search_chance)
        self.depth_target = deep_depth if self.is_deep else rules.shallow_depth

        self.simulations_done = self.tree.data.visit_count[root_idx] if root_idx != -1 else 0
        begin_search_numba(self.tree.data, self.tree.delta_games, root_idx)

        self.waiting_for_network = False

        cur_move = self.tree.data.game_state.current_move[0]

        if cur_move > rules.temp_decay_turn:
            self.temperature = self.temperature * rules.temp_decay_coeff ** (
                    cur_move - rules.temp_decay_turn)
        else:
            self.temperature = rules.initial_temp

        self.root_player = get_current_player_numba(self.tree.data.game_state)

        #print(f"start turn {self.tree.data.game_state.komi_norm[0]}")


NO_PROGRESS = 0
TREE_PROGRESS = 1
MOVE_MADE = 2


def update_ctx(ctx: AgentContext) -> int:

    rules = ctx.rules

    if ctx.tree.data.game_state.game_over[0]:
        return NO_PROGRESS

    # 1. Если агент ждет ответа от сети — проверяем готовность
    if ctx.waiting_for_network:

        #print(f"[{ctx.index}] checking result_ready, pending={ctx.client._request_pending}")

        if ctx.client.result_ready():
            policy, value, score = ctx.client.get_result()

            utility = expand_node_from_network_result_numba(ctx.tree.data, ctx.tree.temp, ctx.tree.delta_games,
                                                            ctx.node_to_expand,
                                                            ctx.root_player, 1.0, policy, value, score)

            backpropagate_numba(ctx.tree.data, ctx.tree.temp, ctx.tree.delta_games, utility)

            ctx.simulations_done += 1
            ctx.waiting_for_network = False

            return TREE_PROGRESS

        return NO_PROGRESS # Переходим к следующему агенту

    # 2. Если агент не ждет сеть, проверяем, закончил ли он симуляции для хода
    if ctx.simulations_done < ctx.depth_target:

        root_idx = ctx.tree.data.root_index[0]
        #Если корень уже расширен. Добавляем шум
        if ctx.tree.data.is_expanded[root_idx] :
            add_dirichlet_noise_numba(ctx.tree.data, ctx.tree.temp, ctx.tree.delta_games, root_idx, rules.noise_frac, ctx.rng)

        node_idx = select_leaf_numba(ctx.tree.data, ctx.tree.temp, ctx.tree.delta_games, ctx.tree.delta_boards, C_PUCT, Q_INIT_LOSS)

        ctx.node_to_expand = node_idx

        if ctx.tree.data.game_state.game_over[0]:
            # Терминальный узел — справляемся без сети

            winner, margin = get_winner_and_margin_numba(ctx.tree.data.game_state)

            if winner == ctx.root_player:
                value = 1
            elif winner == EMPTY:
                value = 0
            else:
                value = -1

            ctx.tree.data.is_terminal[node_idx] = True

            backpropagate_numba(ctx.tree.data, ctx.tree.temp, ctx.tree.delta_games, value)

            ctx.simulations_done += 1
        else:
            # Нетерминальный узел — запрашиваем сеть
            state_tensor = get_network_input_from_game(ctx.tree.data.game_state)
            ctx.client.send_request(state_tensor)
            ctx.waiting_for_network = True

        return TREE_PROGRESS

    # 3. MCTS для хода завершен — делаем физический ход

    if rules.use_override_temp and ctx.rng.random() < rules.override_temp_chance:
        temperature = max(ctx.temperature, rules.override_temp)
    else:
        temperature = ctx.temperature

    encoded_move, best_node_idx = select_action_numba(ctx.tree.data, ctx.tree.temp, temperature,
                                                                    ctx.rng)
    ctx.best_node = best_node_idx

    ctx.write_current_to_history(encoded_move)

    #print(f"applying delta game with {best_node_idx}")
    apply_delta_game_numba(ctx.tree.data.game_state, ctx.tree.delta_boards, ctx.tree.delta_games, best_node_idx)

    #print("starting new turn")
    ctx.start_new_turn(best_node_idx)

    #print(f"move made with {ctx.client.name}" )
    return MOVE_MADE
