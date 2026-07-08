
from dataclasses import dataclass, field
from config import (
    DEEP_DEPTH, SHALLOW_DEPTH, DEEP_SEARCH_CHANCE,
    UNTIL_THE_END_CHANCE, INITIAL_TEMP, TEMP_DECAY_TURN, TEMP_DECAY_COEFF,
    USE_PCR, USE_EARLY_RESIGN, USE_FAST_FINISH, FAST_FINISH_AT, CONCEDE_AT,
    MIN_TURNS_BEFORE_RESIGN, NOISE_FRAC,
    USE_OVERRIDE_TEMP, OVERRIDE_TEMP, OVERRIDE_TEMP_CHANCE,
    MEAN_KOMI,
)


@dataclass
class GameRules:
    """
    Все параметры, которые раньше были захардкожены константами внутри
    AgentContext / update_ctx / init_new_game.

    Self-play: GameRules()            — дефолты совпадают с config.py
    Турнир:   GameRules.tournament()  — фиксированное коми, бюджет, без шума
    """

    # ── Бюджет симуляций ──────────────────────────────────────────────
    deep_depth:         int   = DEEP_DEPTH
    shallow_depth:      int   = SHALLOW_DEPTH
    deep_search_chance: float = DEEP_SEARCH_CHANCE
    use_pcr:            bool  = USE_PCR          # Partial Completion Rate

    # ── Коми ─────────────────────────────────────────────────────────
    fixed_komi:         float | None = None      # None = случайное (self-play)
    mean_komi:          float        = MEAN_KOMI
    until_the_end_chance: float      = UNTIL_THE_END_CHANCE

    # ── Температура ───────────────────────────────────────────────────
    initial_temp:       float = INITIAL_TEMP
    temp_decay_turn:    int   = TEMP_DECAY_TURN
    temp_decay_coeff:   float = TEMP_DECAY_COEFF
    use_override_temp:  bool  = USE_OVERRIDE_TEMP
    override_temp:      float = OVERRIDE_TEMP
    override_temp_chance: float = OVERRIDE_TEMP_CHANCE

    # ── Ранний выход / быстрое завершение ─────────────────────────────
    use_early_resign:   bool  = USE_EARLY_RESIGN
    use_fast_finish:    bool  = USE_FAST_FINISH
    fast_finish_at:     float = FAST_FINISH_AT
    concede_at:         float = CONCEDE_AT
    min_turns_before_resign: int = MIN_TURNS_BEFORE_RESIGN

    # ── Шум Дирихле ───────────────────────────────────────────────────
    noise_frac:         float = NOISE_FRAC

    # ── Произвольные дополнительные поля от ТМ ───────────────────────
    extra: dict = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def for_self_play(cls) -> "GameRules":
        """Дефолтный self-play: всё из config.py."""
        return cls()

    @classmethod
    def tournament(
        cls,
        komi: float = 0.0,
        simulation_budget: int = DEEP_DEPTH,
        temperature: float = 0.15,
        **kwargs,
    ) -> "GameRules":
        """
        Турнирный пресет:
          - фиксированное коми
          - фиксированный бюджет симуляций (deep_depth = simulation_budget, pcr отключён)
          - низкая температура, без затухания
          - ранний выход и быстрое завершение отключены
          - шум Дирихле отключён
        """
        return cls(
            fixed_komi=komi,
            until_the_end_chance=1.0,       # всегда играем до конца
            deep_depth=simulation_budget,
            shallow_depth=simulation_budget,
            deep_search_chance=1.0,
            use_pcr=False,
            initial_temp=temperature,
            temp_decay_turn=10 ** 9,        # никогда не затухает
            temp_decay_coeff=1.0,
            use_override_temp=False,
            use_early_resign=False,
            use_fast_finish=False,
            noise_frac=0.1,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "GameRules":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra  = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)