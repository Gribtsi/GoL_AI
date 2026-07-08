from typing import Tuple
from numba import njit
import numpy as np

from config import MAX_KOMI, FLAT_RANDOM_KOMI_CHANCE, MEAN_KOMI, KOMI_STD_DEV, RANDOMISE_KOMI, \
    FAST_FINISH_AT, SHALLOW_DEPTH, DEEP_DEPTH, BAD_VALUES_COUNT, PIE_RULE_CHANCE, board_size_sqr, SCORE_SATURATION_AT, \
    SCORE_UTILITY_SCALE


_POSSIBLE_VALUES = np.arange(-MAX_KOMI, MAX_KOMI + 1, 1.0)

_PROBS = np.exp(-0.5 * ((_POSSIBLE_VALUES - MEAN_KOMI) / KOMI_STD_DEV) ** 2)
_PROBS /= np.sum(_PROBS)

_CDF = np.cumsum(_PROBS)
@njit
def _generate_komi_njit(
        possible_values: np.ndarray,
        cdf: np.ndarray,             # передаём _CDF явно
        rng: np.random.Generator
) -> Tuple[float, bool]:

    r = rng.random()

    if r < PIE_RULE_CHANCE:
        return 0.0, False

    elif r < PIE_RULE_CHANCE + FLAT_RANDOM_KOMI_CHANCE:
        # симметричный диапазон [-MAX_KOMI, MAX_KOMI]
        idx = rng.integers(0, MAX_KOMI * 2 + 1)
        base = float(idx - MAX_KOMI)
        is_flat = True

    else:
        # семплирование через CDF
        u = rng.random()
        idx = np.searchsorted(cdf, u)
        if idx >= len(possible_values):
            idx = len(possible_values) - 1
        base = float(possible_values[idx])
        is_flat = False

    # +0.5 только если flat не попал на границу ИЛИ всегда — зависит от логики
    if rng.random() < 0.5:
        base += 0.5

    # clamp с учётом половинок
    lo = float(-MAX_KOMI) + 0.5
    hi = float(MAX_KOMI)  - 0.5
    if base < lo:
        base = lo
    elif base > hi:
        base = hi

    return base, is_flat




# === 4. Обертка с нужной вам сигнатурой ===
def generate_komi(komi_offset : float, rng: np.random.Generator) -> Tuple[float, bool]:
    """
    Генерирует коми. С вероятностью flat_chance выбирает равномерно из всего диапазона,
    иначе использует нормальное распределение вокруг mean_komi.
    """
    # Просто передаем заранее высчитанные глобальные массивы в njit-метод

    if RANDOMISE_KOMI:
        return _generate_komi_njit(_POSSIBLE_VALUES, _CDF, rng)
    else:
        return 0.0, False



@njit()
def get_fast_finish_simulations_and_chance(value: float) -> Tuple[float, float]:

    value = min(value, FAST_FINISH_AT)
    p = (value + 1.0) / 2.0
    p_max = (FAST_FINISH_AT + 1.0) / 2.0
    la = p / p_max

    max_depth = (1 - la) * SHALLOW_DEPTH + la * DEEP_DEPTH
    deep_search_chance = 0.1 + 0.9 * (1 - la)

    return max_depth, deep_search_chance


@njit()
def all_less_than_threshold(threshold : float, values: np.ndarray, current_ptr : int):
    return get_mean_value(values, current_ptr) < threshold


@njit()
def get_mean_value(values: np.ndarray, current_ptr : int) -> float:
    values_sum = 0
    index = current_ptr
    for i in range(BAD_VALUES_COUNT):
        values_sum += values[index]
        index = (index + 2) % (BAD_VALUES_COUNT * 2)
    return values_sum / BAD_VALUES_COUNT

_SCORES_ARRAY = np.arange(-board_size_sqr, board_size_sqr + 1, dtype=np.float32)

@njit()
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


@njit()
def get_score_utility(expected_score: float) -> float:
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
    score_utility = np.tanh(expected_score / SCORE_SATURATION_AT) * SCORE_UTILITY_SCALE

    return float(score_utility)
