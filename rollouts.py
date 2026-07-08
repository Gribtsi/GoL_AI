from numba import njit
import numpy as np

from board import copyto_numba, get_territories, get_opponent, fast_territories_numba, build_delta_board_array
from config import (
    BOARD_SIZE,
    board_size_sqr,
    EMPTY,
    BLACK,
    WHITE,
    pass_code, possible_moves_total,
)
from move import encode_move, decode_move
from game import (
    GameData,
    DeltaGameArray,
    DeltaBoardArray,
    try_make_game_move_numba,
    get_current_player_numba,
    get_winner_and_margin_numba, build_game_data, build_delta_game_array, undo_game_numba,
)


@njit()
def copy_game_state_numba(src: GameData, dst: GameData):
    copyto_numba(dst.move_history, src.move_history)
    copyto_numba(dst.current_move, src.current_move)
    copyto_numba(dst.game_over, src.game_over)
    copyto_numba(dst.has_mask, src.has_mask)
    copyto_numba(dst.legal_mask, src.legal_mask)

    copyto_numba(dst.komi_norm, src.komi_norm)
    copyto_numba(dst.max_moves, src.max_moves)

    copyto_numba(dst.board.current_state, src.board.current_state)
    copyto_numba(dst.board.history_stack, src.board.history_stack)
    copyto_numba(dst.board.history_hashes, src.board.history_hashes)
    copyto_numba(dst.board.zobrist_table, src.board.zobrist_table)
    copyto_numba(dst.board.history_ptr, src.board.history_ptr)
    copyto_numba(dst.board.history_hash_ptr, src.board.history_hash_ptr)
    copyto_numba(dst.board.history_count, src.board.history_count)
    copyto_numba(dst.board.history_hash_count, src.board.history_hash_count)


@njit()
def is_simple_eye_point(state: np.ndarray, x: int, y: int, color: int) -> bool:
    if state[x, y] != EMPTY:
        return False

    found_any_neighbor = False

    if x > 0:
        found_any_neighbor = True
        if state[x - 1, y] != color:
            return False
    if x + 1 < BOARD_SIZE:
        found_any_neighbor = True
        if state[x + 1, y] != color:
            return False
    if y > 0:
        found_any_neighbor = True
        if state[x, y - 1] != color:
            return False
    if y + 1 < BOARD_SIZE:
        found_any_neighbor = True
        if state[x, y + 1] != color:
            return False

    return found_any_neighbor


@njit()
def calc_territory_for_color_after_move(
    game: GameData,
    delta_boards: DeltaBoardArray,
    delta_games: DeltaGameArray,
    ptr: int,
    encoded_move: int,
    player_color: int
) -> int:
    ok = try_make_game_move_numba(game, delta_boards, delta_games, ptr, encoded_move, True)
    if not ok:
        return -1

    total = territory_for_color(game, player_color)

    undo_game_numba(game)

    return total


@njit()
def is_simple_eye_point(state: np.ndarray, x: int, y: int, color: int) -> bool:
    if state[x, y] != EMPTY:
        return False

    found_any_neighbor = False

    if x > 0:
        found_any_neighbor = True
        if state[x - 1, y] != color:
            return False
    if x + 1 < BOARD_SIZE:
        found_any_neighbor = True
        if state[x + 1, y] != color:
            return False
    if y > 0:
        found_any_neighbor = True
        if state[x, y - 1] != color:
            return False
    if y + 1 < BOARD_SIZE:
        found_any_neighbor = True
        if state[x, y + 1] != color:
            return False

    return found_any_neighbor

@njit()
def territory_for_color(game: GameData, color: int) -> int:
    white_terr, black_terr = fast_territories_numba(game.board, game.board_temp)

    return white_terr if color == WHITE else (black_terr if color == BLACK else -1)

@njit()
def collect_rollout_moves(
    game: GameData,
    moves_out: np.ndarray,
    forbid_eyes: bool,
    forbid_terr_dec: bool,
    delta_boards: DeltaBoardArray,
    delta_games: DeltaGameArray
) -> int:
    player = get_current_player_numba(game)
    count = 0

    current_territory = territory_for_color(game, player)

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if game.board.current_state[x, y] != EMPTY:
                continue

            if forbid_eyes and is_simple_eye_point(game.board.current_state, x, y, player):
                continue

            move0 = encode_move(x, y, 0, False, False)
            if try_make_game_move_numba(game, delta_boards, delta_games, 0, move0, True):
                if forbid_terr_dec:
                    new_territory = calc_territory_for_color_after_move(
                        game, delta_boards, delta_games, 0, move0, player
                    )
                    if new_territory < current_territory:
                        pass
                    else:
                        moves_out[count] = move0
                        count += 1
                else:
                    moves_out[count] = move0
                    count += 1

            move1 = encode_move(x, y, 1, False, False)
            if try_make_game_move_numba(game, delta_boards, delta_games, 0, move1, True):
                if forbid_terr_dec:
                    new_territory = calc_territory_for_color_after_move(
                        game, delta_boards, delta_games, 0, move1, player
                    )
                    if new_territory < current_territory:
                        pass
                    else:
                        moves_out[count] = move1
                        count += 1
                else:
                    moves_out[count] = move1
                    count += 1

    return count


@njit()
def rollout_impl(
    source_game: GameData,
    temp_game: GameData,
    delta_boards: DeltaBoardArray,
    delta_games: DeltaGameArray,
    moves_temp: np.ndarray,
    forbid_eyes: bool,
    forbid_terr_dec: bool,
    rng: np.random.Generator
) -> float:
    copy_game_state_numba(source_game, temp_game)


    source_player = get_current_player_numba(source_game)

    while not temp_game.game_over[0]:
        move_count = collect_rollout_moves(
            temp_game,
            moves_temp,
            forbid_eyes,
            forbid_terr_dec,
            delta_boards,
            delta_games
        )

        if move_count == 0:
            chosen_move = pass_code
        else:
            idx = rng.integers(0, move_count)
            chosen_move = moves_temp[idx]

        try_make_game_move_numba(temp_game, delta_boards, delta_games, 0, chosen_move, True)

    winner, _ = get_winner_and_margin_numba(temp_game)

    if winner == EMPTY:
        return 0.0
    elif winner == source_player:
        return 1.0
    else:
        return -1.0


@njit()
def try_move_on_temp_game_numba(
    source_game: GameData,
    temp_game: GameData,
    delta_boards,
    delta_games,
    move: int
) -> bool:
    copy_game_state_numba(source_game, temp_game)

    ok = try_make_game_move_numba(
        temp_game,
        delta_boards,
        delta_games,
        0,
        move,
        True
    )

    if ok:
        undo_game_numba(temp_game)

    return ok

@njit()
def build_rollout_policy_mask_numba(
    source_game: GameData,
    temp_game: GameData,
    delta_boards: DeltaBoardArray,
    delta_games: DeltaGameArray,
    out_mask: np.ndarray,
    no_eyes: bool,
    no_terr_dec: bool
):
    copy_game_state_numba(source_game, temp_game)

    out_mask.fill(0)

    player = get_current_player_numba(temp_game)

    current_territory = 0
    if no_terr_dec:
        current_territory = territory_for_color(temp_game, player)

    for move in range(possible_moves_total):
        if move == pass_code:
            out_mask[move] = 1.0
            continue

        x, y, gol, is_pass, is_swap = decode_move(move)

        if no_eyes and is_simple_eye_point(temp_game.board.current_state, x, y, player):
            continue

        ok = try_make_game_move_numba(
            temp_game,
            delta_boards,
            delta_games,
            0,
            move,
            True
        )
        if not ok:
            continue

        keep = True

        if no_terr_dec:
            new_territory = territory_for_color(temp_game, player)
            if new_territory < current_territory:
                keep = False

        if move != pass_code:
            undo_game_numba(temp_game)

        if keep:
            out_mask[move] = 1.0


class Rollout:
    def __init__(self):
        self.temp_game = build_game_data(0)
        self.delta_boards = build_delta_board_array(1)
        self.delta_games = build_delta_game_array(1)
        self.candidate_moves = np.empty(board_size_sqr * 2, dtype=np.int32)
        self.policy_mask = np.zeros(possible_moves_total, dtype=np.bool_)

    def rollout(self, source_game: GameData, no_eyes: bool, no_terr_dec: bool, rng: np.random.Generator) -> int:
        self.temp_game.komi_norm[0] = source_game.komi_norm[0]

        return int(rollout_impl(
            source_game,
            self.temp_game,
            self.delta_boards,
            self.delta_games,
            self.candidate_moves,
            no_eyes,
            no_terr_dec,
            rng
        ))

    def try_move(self, source_game: GameData, move: int) -> bool:
        self.temp_game.komi_norm[0] = source_game.komi_norm[0]
        return bool(try_move_on_temp_game_numba(
            source_game,
            self.temp_game,
            self.delta_boards,
            self.delta_games,
            move
        ))

    def build_policy_mask(self, source_game: GameData, no_eyes: bool, no_terr_dec: bool) -> np.ndarray:
        self.temp_game.komi_norm[0] = source_game.komi_norm[0]

        build_rollout_policy_mask_numba(
            source_game,
            self.temp_game,
            self.delta_boards,
            self.delta_games,
            self.policy_mask,
            no_eyes,
            no_terr_dec
        )
        return self.policy_mask