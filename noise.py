import numpy as np


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
        """Конфигурация для self-play обучения (шум отключен)."""
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

    def get_noise(self, legal_mask: np.ndarray) -> np.ndarray:
        """
        Возвращает вектор шума длины legal_mask.shape[0].
        Нелегальные ходы получают 0.
        Легальные ходы получают распределение Dirichlet.
        """
        legal_mask = np.asarray(legal_mask)
        if legal_mask.ndim != 1:
            raise ValueError("legal_mask must be a 1D array")

        noise = np.zeros_like(legal_mask, dtype=np.float32)

        if not self.enabled:
            return noise

        legal_indices = np.flatnonzero(legal_mask > 0)
        if legal_indices.size == 0:
            return noise

        dirichlet = np.random.dirichlet(
            [self.alpha] * legal_indices.size
        ).astype(np.float32)

        noise[legal_indices] = dirichlet
        return noise