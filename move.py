from numba.experimental import jitclass
from numba import int64, boolean, njit
from typing import Dict, Any, Tuple

from config import BOARD_SIZE, board_size_sqr, pass_code, swap_code


def move_to_dict(x, y, life_cycles, color, is_pass) -> Dict[str, Any]:
    """Вызывается только из обычного Python-кода (не из @njit)"""
    return {
        "pos_x": x,
        "pos_y": y,
        "life_cycles": life_cycles,
        "color": color,
        "is_pass": is_pass
    }


def dict_to_move(data: Dict[str, Any]) -> Tuple[int,int,int,int,bool]:
    pos = data.get("position")
    is_pass = data.get("is_pass", False)

    if pos is None or is_pass:
        pos_x, pos_y = -1, -1
        is_pass = True
    else:
        pos_x, pos_y = pos[0], pos[1]

    return(
        pos_x,
        pos_y,
        data.get("life_cycles", 0),
        data.get("color", 0),
        is_pass
    )


@njit
def decode_move(action_idx: int) -> Tuple[int,int,int,bool, bool]:
    """
    Декодировать индекс действия в объект Move.

    Args:
        action_idx: Индекс от 0 до 722
        color: Цвет текущего игрока

    Returns:
        Объект Move
    """
    if action_idx == pass_code:
        # Пас
        return 0, 0, 0, True, False
    elif action_idx == swap_code:
        return 0,0,0, False, True
    else:
        # Обычный ход
        position_idx = action_idx % board_size_sqr
        life_cycle = action_idx // board_size_sqr

        x = position_idx // BOARD_SIZE
        y = position_idx % BOARD_SIZE

        return x, y, life_cycle, False, False


@njit
def encode_move(x:int, y:int, life_cycles:int, is_pass:bool, is_swap : bool) -> int:
    """
    Кодировать Move в индекс действия.

    Args:
        move: Объект Move

    Returns:
        Индекс действия (0-722)
    """
    if is_pass:
        return pass_code
    elif is_swap:
        return swap_code
    else:
        position_idx = x * BOARD_SIZE + y
        return position_idx + life_cycles * BOARD_SIZE * BOARD_SIZE
