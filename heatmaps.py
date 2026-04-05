from typing import List, Dict
import numpy as np

TurnRecord = Dict[str, np.ndarray]  # один ход
GameRecord = List[TurnRecord]  # одна партия


def make_emoji_mapper(
        thresholds=(1, 5, 10, 20, 50, 100),
        emojis=("🟢", "🟡", "🟠", "🟤", "🔴", "🟣", "🔵"),
):
    t1, t2, t3, t4, t5, t6 = thresholds

    def mapper(count: int) -> str:
        if count == 0:
            return "⚪"
        if count <= t1:
            return emojis[0]
        if count <= t2:
            return emojis[1]
        if count <= t3:
            return emojis[2]
        if count <= t4:
            return emojis[3]
        if count <= t5:
            return emojis[4]
        if count <= t6:
            return emojis[5]
        return emojis[6]

    return mapper


def print_board_heatmaps_for_move(
        move_index: int,
        history: int,
        states: List[TurnRecord],
        thresholds=(1, 5, 10, 20),
):
    """
    move_index – номер хода (0-based), по которому фильтруем.
    states     – единый список всех состояний всех игр.
    thresholds – пороги для градации частот.
    """

    # Берём только те записи, где turn == move_index
    filtered = [t for t in states if t["turn"] == move_index]

    if not filtered:
        print(f"Нет состояний с turn == {move_index}")
        return

    black_boards = []
    white_boards = []

    for turn in filtered:
        st = turn["state_tensor"]  # shape: (C, H, W)

        black_board = st[1 + history]  # только чёрные камни [H, W]
        white_board = st[17 + history]  # только белые камни  [H, W]

        if st[32, 0, 0] > 0:
            white_board, black_board = black_board, white_board

        black_boards.append(black_board)
        white_boards.append(white_board)

    # Все доски должны быть одного размера
    black_boards = np.stack(black_boards, axis=0)  # (N, H, W)
    white_boards = np.stack(white_boards, axis=0)  # (N, H, W)

    _, H, W = black_boards.shape

    black_counts = np.zeros((H, W), dtype=np.int32)
    white_counts = np.zeros((H, W), dtype=np.int32)

    # В каналах 1 и 8 считаем непустые клетки
    for b in black_boards:
        black_counts += (b != 0)
    for w in white_boards:
        white_counts += (w != 0)

    mapper = make_emoji_mapper(thresholds)

    def print_grid(counts: np.ndarray, title: str):
        print(title)
        for y in range(H):
            row = [mapper(int(counts[y, x] / len(filtered) * 100)) for x in range(W)]
            print("".join(row))
        print()

    print(f"Ход (turn) == {move_index}")
    print_grid(black_counts, "Чёрные камни:")
    print_grid(white_counts, "Белые камни:")
