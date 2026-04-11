from typing import Tuple
from numba import njit
import numpy as np

from config import MAX_KOMI, FLAT_RANDOM_KOMI_CHANCE, MEAN_KOMI, KOMI_STD_DEV

_POSSIBLE_VALUES = np.arange(-MAX_KOMI + 0.5, MAX_KOMI, 1.0)

_PROBS = np.exp(-0.5 * ((_POSSIBLE_VALUES - MEAN_KOMI) / KOMI_STD_DEV) ** 2)
_PROBS /= np.sum(_PROBS)

_CDF = np.cumsum(_PROBS)


# === 3. Быстрая Numba-функция ===
@njit
def _generate_komi_njit(
        possible_values: np.ndarray,
        cdf: np.ndarray,
        flat_chance: float
) -> Tuple[float, bool]:
    # 1. Плоское распределение
    if np.random.random() < flat_chance:
        # В Numba быстрее взять случайный индекс через randint, чем дергать choice
        idx = np.random.randint(0, len(possible_values))
        return possible_values[idx], True

    # 2. Нормальное распределение (через CDF)
    r = np.random.random()
    # searchsorted за O(log N) находит индекс, где вероятность попадает в нужный интервал
    idx = np.searchsorted(cdf, r)
    return possible_values[idx], False


# === 4. Обертка с нужной вам сигнатурой ===
def generate_komi() -> Tuple[float, bool]:
    """
    Генерирует коми. С вероятностью flat_chance выбирает равномерно из всего диапазона,
    иначе использует нормальное распределение вокруг mean_komi.
    """
    # Просто передаем заранее высчитанные глобальные массивы в njit-метод
    return _generate_komi_njit(_POSSIBLE_VALUES, _CDF, FLAT_RANDOM_KOMI_CHANCE)
