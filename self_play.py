
import time

from IPython.display import clear_output
from numpy.random import random, uniform

from game import Game
from mcts_agent import MCTS_Agent
from random_network import RandomNetwork
from train_data_handle import save_training_data
from board import BOARD_SIZE, BLACK, EMPTY
import numpy as np

def process_game_history(history: list, final_board_state: np.ndarray, score: dict) -> list:
    """Добавляет к каждой записи в истории результат игры."""
    processed = []
    index = 0
    for turn_data in history:
        # Результат с точки зрения текущего игрока


        winner = score['winner']
        current_player = turn_data['player']
        margin = int(score['margin_no_komi'])
        if current_player == winner:
            value = 1.0
            total_score = margin
        elif winner is None:  # Ничья
            value = 0.0
            total_score = 0
        else:  # Проигрыш
            value = -1.0
            total_score = -margin

        territories = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.float32)
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                territories[i,j] = 1 if final_board_state[i,j] == current_player else -1 if final_board_state[i,j] != EMPTY else 0

        move = turn_data['move']

        processed.append({
            'state_tensor': turn_data['state_tensor'],
            'mcts_policy': turn_data['mcts_policy'],
            'value': value,
            'score': total_score,
            'territories': territories,
            'turn': index,
            'move': move.to_dict()
        })

        index += 1
    return processed

def play_game_with_visualization(
        rl_agent: MCTS_Agent,
        random_turns: int = 10, random_agent: MCTS_Agent = None, delay: float = 1.0,
        max_moves: int = 300, play_until_end_chance = 0.1, concede_value = 0.9, extra_data: str = "") -> (Game, MCTS_Agent):
    """
    Запустить одну партию с визуализацией в Jupyter Notebook.

    Args:
        agent: MCTS агент для игры
        delay: Задержка между ходами в секундах (минимум 1.0)
        max_moves: Максимальное количество ходов (защита от зацикливания)

    Returns:
        game: Финальное состояние игры
    """
    # Создаем новую игру
    game = Game()
    move_count = 0


    till_the_end = (uniform(0,1) <= play_until_end_chance)
    concede_at = concede_value

    print("=" * 60)
    print("НАЧАЛО ИГРЫ: MCTS Agent (Self-Play)")
    print(f"Симуляций MCTS на ход: {rl_agent.num_simulations}")
    print(f"Температура: {rl_agent.temperature}")
    print("=" * 60)
    time.sleep(2)

    best_node = None

    turn_index = 0

    # Игровой цикл
    while (not game.game_over) and (move_count < max_moves):
        clear_output(wait=True)

        # Отображаем текущее состояние
        print("=" * 60)
        print(f"ХОД {move_count + 1} | {extra_data} | до конца {till_the_end}")
        print("=" * 60)
        game.print_board()
        print()

        if not (random_agent is None) and turn_index < random_turns:
            agent = random_agent
        else:
            agent = rl_agent

        # Запускаем MCTS поиск
        print(f"🔍 MCTS запущен ({agent.num_simulations} симуляций)...")
        start_time = time.time()

        if best_node is None:
            move, policy, best_node = agent.search(game)
        else:
            move, policy, best_node = agent.search(game, best_node)

        turn_index += 1

        search_time = time.time() - start_time
        print(f"✅ MCTS завершен за {search_time:.2f} сек")
        print(f"📍 Выбран ход: {move}")

        current_state_tensor = game.get_network_input_pytorch(32)
        game.history.append({
            'state_tensor': current_state_tensor,
            'mcts_policy': policy,
            'player': game.current_player,
            'move': move
        })

        # Применяем ход
        success = game.make_move(move)

        if not success:
            raise Exception(f"❌ ОШИБКА: Ход не был применен!")

        move_count += 1

        if not till_the_end:
            print(f"mean value {best_node.mean_value} player {best_node.game_state.current_player} move {best_node.game_state.current_move}")
            if (best_node.mean_value > concede_at) and (game.get_current_leader() != game.current_player): # У нас инверсия откуда-то яхз взялась, поэтому знак больше
                game.game_over = True

        if delay - search_time > 0:
            time.sleep(delay - search_time)

    # Финальное состояние
    clear_output(wait=True)
    print("=" * 60)
    print(f"ИГРА ОКОНЧЕНА! {max_moves}")
    print("=" * 60)
    game.print_board()
    print()

    # Подсчет очков
    game.print_score()
    print()
    print(f"Всего ходов: {move_count}")

    if move_count >= max_moves:
        print(f"⚠️  Достигнут лимит ходов ({max_moves})")

    return game, rl_agent


def generate_self_play_games(
        games_to_generate: int, games_batch: int,
        rl_agent: MCTS_Agent, max_moves: int,
        data_save_dir: str, random_moves : int = 0, random_mcts : int = 2, delay : float = 1, play_until_end_chance = 0.9, concede_value = 0.9):
    random = RandomNetwork()
    agent_random = MCTS_Agent(random, num_simulations=random_mcts, temperature=1.0)



    for k in range(int(games_to_generate / games_batch) if games_batch < games_to_generate else 1):
        all_training_data = []
        start_time = time.time()
        for i in range(games_batch if games_batch < games_to_generate else games_to_generate):
            text = f"ИГРА {i + 1 + k * games_batch}/{games_to_generate} | ПРОШЛО {time.time() - start_time}c."
            game_result, agent = play_game_with_visualization(rl_agent=rl_agent, random_agent=agent_random, random_turns=random_moves,
                                                       delay=delay, max_moves=max_moves, extra_data=text, play_until_end_chance=play_until_end_chance, concede_value=concede_value)




            # Обрабатываем историю, добавляя итоговый результат

            processed_history = process_game_history(game_result.history, game_result.board.get_territories(), game_result.get_score())



            all_training_data.extend(processed_history)

            # --- Блок сохранения данных ---

        print(f"\nЭтап {k} сбора данных завершен. Собрано {len(all_training_data)} состояний.")

        if not save_training_data(save_dir=data_save_dir, training_data=all_training_data):
            print(f"\n❌ Ошибка при сохранении батча игр. Генерация остановлена.")
            break

    return agent_random

def play_tournament_game(player1: MCTS_Agent, player2: MCTS_Agent, max_moves: int = 300, extra_text=""):
    """
    Запускает одну быструю партию между двумя агентами без визуализации.

    Args:
        player1 (MCTS_Agent): Агент, играющий за Черных (ходит первым).
        player2 (MCTS_Agent): Агент, играющий за Белых.
        max_moves (int): Максимальное количество ходов для предотвращения зацикливания.

    Returns:
        Game: Объект с финальным состоянием игры.
    """
    # 1. Создание новой игры
    game = Game()
    move_count = 0

    # Игровой цикл
    while not game.game_over and move_count < max_moves:

        clear_output(wait=True)

        # Отображаем текущее состояние
        print("=" * 60)
        print(f"| ХОД {move_count + 1} | {extra_text} |")
        print("=" * 60)
        game.print_board()
        print()

        # 2. Выбор текущего агента в зависимости от цвета игрока
        if game.current_player == game.board.BLACK:
            current_agent = player1
        else:
            current_agent = player2

        # Запускаем MCTS поиск
        print(f"🔍 MCTS запущен ({current_agent.num_simulations} симуляций)...")
        start_time = time.time()

        # 3. Запуск MCTS поиска для текущего агента
        # Мы не сохраняем историю и политику, так как в турнире нас интересует только результат
        move, _, _ = current_agent.search(game)

        search_time = time.time() - start_time
        print(f"✅ MCTS завершен за {search_time:.2f} сек")
        print(f"📍 Выбран ход: {move}")

        # 4. Применение хода
        success = game.make_move(move)

        if not success:
            print(f"❌ ОШИБКА: Агент {game.current_player} сделал нелегальный ход {move}!")
            # В этом случае нужно присвоить победу оппоненту
            game.winner = game.get_opponent(game.current_player)
            game.game_over = True
            break  # Прерываем игру

        move_count += 1

    # Если игра не завершилась по правилам (например, два паса),
    # но достигнут лимит ходов, победитель определяется по очкам.
    if not game.game_over:
        score = game.get_score()
        game.winner = score.get('winner')
        game.game_over = True

    return game
