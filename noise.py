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

    def add_noise(self, policy: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """
        Добавить шум Дирихле к policy.

        Args:
            policy: Исходная policy от нейросети (уже нормализованная)
            legal_mask: Маска легальных ходов

        Returns:
            Policy с добавленным шумом
        """
        if not self.enabled:
            return policy

        # Количество легальных ходов
        num_legal = int(np.sum(legal_mask))

        if num_legal == 0:
            return policy

        # Генерируем шум Дирихле только для легальных ходов
        noise = np.zeros_like(policy)
        legal_indices = np.where(legal_mask > 0)[0]

        # Дирихле для легальных ходов
        dirichlet_noise = np.random.dirichlet([self.alpha] * num_legal)
        noise[legal_indices] = dirichlet_noise

        # Смешиваем policy с шумом
        noisy_policy = (1 - self.epsilon) * policy + self.epsilon * noise

        # Нормализуем на всякий случай
        noisy_policy = noisy_policy / np.sum(noisy_policy)

        return noisy_policy

    @staticmethod
    def for_selfplay() -> 'DirichletNoiseConfig':
        """Конфигурация для self-play обучения (шум отключен)."""
        return DirichletNoiseConfig(epsilon=0.5, alpha=0.6, enabled=True)

    @staticmethod
    def for_tournament() -> 'DirichletNoiseConfig':
        """Конфигурация для турнира (шум включен для разнообразия)."""
        return DirichletNoiseConfig(epsilon=0.25, alpha=0.3, enabled=True)

    @staticmethod
    def for_human_play() -> 'DirichletNoiseConfig':
        """Конфигурация для игры с человеком (шум включен, но слабее)."""
        return DirichletNoiseConfig(epsilon=0.15, alpha=0.3, enabled=True)
