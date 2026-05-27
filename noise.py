import numpy as np

from config import possible_moves_total


class DirichletNoiseConfig:
    """
    Конфигурация шума Дирихле для добавления разнообразия в корневой узел MCTS.

    Используется для увеличения exploration и создания уникальных траекторий игр
    даже при детерминированном выборе ходов (temperature=0).
    """

    def __init__(self, epsilon: float = 0.25, alpha: float = 0.3, enabled: bool = True):
        """
        Args:
            epsilon: Доля шума в итоговой policy (0.25 в оригинальном AlphaZero)
            alpha: Параметр концентрации распределения Дирихле (0.3 для Go)
            enabled: Включен ли шум (False для self-play обучения, True для турниров)
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.enabled = enabled

    @staticmethod
    def for_selfplay() -> 'DirichletNoiseConfig':
        """Конфигурация для self-play обучения"""
        return DirichletNoiseConfig(epsilon=0.25, alpha=0.15, enabled=True)

    @staticmethod
    def for_tournament() -> 'DirichletNoiseConfig':
        """Конфигурация для турнира (шум включен для разнообразия)."""
        return DirichletNoiseConfig(epsilon=0.25, alpha=0.3, enabled=True)

    @staticmethod
    def for_human_play() -> 'DirichletNoiseConfig':
        """Конфигурация для игры с человеком (шум включен, но слабее)."""
        return DirichletNoiseConfig(epsilon=0.15, alpha=0.3, enabled=True)

    @property
    def exploration_fraction(self) -> float:
        return self.epsilon

    def get_noise(self, rng: np.random.Generator, dim: int) -> np.ndarray:

        if not self.enabled:
            return np.zeros(dim, dtype=np.float32)

        dirichlet = rng.dirichlet(
            [self.alpha] * dim
        ).astype(np.float32)
        return dirichlet
