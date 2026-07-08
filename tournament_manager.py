"""
tournament_manager.py

TournamentManager — центральный сервис турнира.

Роли:
  1. Подменяет ModelProvider: держит REP-сокет MODEL_PROVIDER_ADDRESS,
     раздаёт разные модели чётным (агент A) и нечётным (агент B) инференс-сервисам.
  2. Подменяет DataWriter: держит PULL-сокет WRITER_ADDRESS,
     принимает game_data / game_log / tournament_result от воркеров.
  3. Читает расписание турниров из JSONL-файла TOURNAMENT_SCHEDULE_PATH.
  4. Управляет воркер-процессами: раздаёт задания через PUSH-сокеты.
  5. Пишет итоги в JSONL-файл TOURNAMENT_RESULTS_PATH.

Схема инференс-сервисов:
  Инференсы с чётным  process_id → модель A (претендент А)
  Инференсы с нечётным process_id → модель B (претендент Б)

  Маппинг воркер → инференс (при INFERENCE_SERVICES_COUNT=4, PROCESSES_PER_WORKER=20):
    workers 0..9  → inf_a=0, inf_b=1
    workers 10..19 → inf_a=2, inf_b=3
"""

import os
import json
import time
import threading
import zmq
import pickle
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from addresses_config import MODEL_PROVIDER_ADDRESS, WRITER_ADDRESS, TOURNAMENT_ACK_ADDRESS
from config import (
    INFERENCE_SERVICES_COUNT,
    PROCESSES_PER_WORKER, GAMES_PER_TOURNAMENT_PROCESS,
)
from game_rules import GameRules
from tournament_service import (
    MatchAssignment,
    get_worker_control_address,
    CMD_START, CMD_STOP,
)

# ──────────────────────────────────────────────
# Пути и константы
# ──────────────────────────────────────────────
MODELS_DIR   = os.environ.get("MODELS_DIR",  "/models")
DATASETS_DIR = os.environ.get("DATASETS_DIR", "/data")

TOURNAMENT_SCHEDULE_PATH = os.path.join(DATASETS_DIR, "tournament_schedule.jsonl")
TOURNAMENT_RESULTS_PATH  = os.path.join(DATASETS_DIR, "tournament_results.jsonl")

# Пауза после смены модели — ждём, пока все инференс-сервисы
# сами подтянут новую версию через check_for_updates (интервал ~10 сек)
MODEL_SWITCH_DELAY_SEC = 20

# Число воркеров на одну пару инференс-сервисов
# (PROCESSES_PER_WORKER // (INFERENCE_SERVICES_COUNT // 2))
def workers_per_inf_pair() -> int:
    res = PROCESSES_PER_WORKER // (INFERENCE_SERVICES_COUNT // 2)

    return res if res > 0 else 1

# ──────────────────────────────────────────────
# Формат расписания (одна строка JSONL)
# ──────────────────────────────────────────────
@dataclass
class TournamentEntry:
    """
    Одна запись расписания турниров.

    Пример JSONL-строки:
    {
        "komi": 6.5,
        "agent_a": "model_v10.pth",
        "agent_b": "model_v7.pth",
        "total_games": 200,
        "rules": {"simulation_budget": 400, "temperature": 0.05}
    }
    """
    agent_a: str
    agent_b: str
    total_games: int
    rules_a: GameRules  # будет передан в TournamentRules.from_dict
    rules_b: GameRules

    @classmethod
    def from_dict(cls, d: dict) -> "TournamentEntry":
        _rules = d.get("rules_a", {})        # komi всегда из верхнего уровня
        rules_a = GameRules.from_dict(_rules)
        _rules = d.get("rules_b", {})        # komi всегда из верхнего уровня
        rules_b = GameRules.from_dict(_rules)
        return cls(
            agent_a=d["agent_a"],
            agent_b=d["agent_b"],
            total_games=d["total_games"],
            rules_a=rules_a,
            rules_b=rules_b
        )


# ──────────────────────────────────────────────
# WorkerController — отправка заданий воркерам
# ──────────────────────────────────────────────
class WorkerController:
    """
    Держит PUSH-сокеты к каждому воркер-процессу.
    """

    def __init__(self, zmq_context: zmq.Context):
        self.context = zmq_context
        self._sockets: dict[int, zmq.Socket] = {}
        for i in range(PROCESSES_PER_WORKER):
            s = self.context.socket(zmq.PUSH)
            addr = get_worker_control_address(i)
            s.connect(addr)
            self._sockets[i] = s
            print(f"[WorkerController] connected to worker {i} at {addr}")

    def send_assignment(self, process_id: int, assignment: MatchAssignment):
        payload = {"command": CMD_START, "assignment": assignment.to_dict()}
        self._sockets[process_id].send_json(payload)

    def stop_worker(self, process_id: int):
        self._sockets[process_id].send_json({"command": CMD_STOP})

    def stop_all(self):
        for pid in range(PROCESSES_PER_WORKER):
            self.stop_worker(pid)


# ──────────────────────────────────────────────
# Вспомогательная функция: маппинг воркер → инференс-пара
# ──────────────────────────────────────────────
def get_inference_pair_for_worker(process_id: int) -> tuple[int, int]:
    """
    Возвращает (inf_a_index, inf_b_index) для данного воркер-процесса.

    При INFERENCE_SERVICES_COUNT=4 и PROCESSES_PER_WORKER=20:
      workers 0..9  → (0, 1)
      workers 10..19 → (2, 3)

    Формула:
      pair_idx = process_id // workers_per_pair
      inf_a = pair_idx * 2
      inf_b = pair_idx * 2 + 1
    """
    wpair = workers_per_inf_pair()
    pair_idx = process_id // wpair
    # Ограничиваем на случай нечётного деления
    max_pair = INFERENCE_SERVICES_COUNT // 2 - 1
    pair_idx = min(pair_idx, max_pair)
    return pair_idx * 2, pair_idx * 2 + 1


# ──────────────────────────────────────────────
# TournamentManager — главный класс
# ──────────────────────────────────────────────
class TournamentManager:

    def __init__(self):
        self.context = zmq.Context()

        self.worker_ctrl    = WorkerController(self.context)

        self.ack_socket = self.context.socket(zmq.PULL)
        self.ack_socket.bind(TOURNAMENT_ACK_ADDRESS)

    def _load_schedule(self) -> list[TournamentEntry]:
        entries = []
        with open(TOURNAMENT_SCHEDULE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(TournamentEntry.from_dict(json.loads(line)))
        print(f"[TM] Loaded {len(entries)} tournament entries from schedule.")
        return entries

    def _distribute_match(self, entry: TournamentEntry) -> list[MatchAssignment]:
        """
        Раздаёт total_games по PROCESSES_PER_WORKER воркерам равномерно.
        Остаток игр добавляется к последнему воркеру.
        """
        games = entry.total_games // PROCESSES_PER_WORKER

        assignments = []
        for pid in range(PROCESSES_PER_WORKER):
            inf_a, inf_b = get_inference_pair_for_worker(pid)
            if games == 0:
                continue
            assignments.append(MatchAssignment(
                inference_a_index=inf_a,
                inference_b_index=inf_b,
                agent_a_name=entry.agent_a,
                agent_b_name=entry.agent_b,
                games_to_generate=games,
                rules_a=entry.rules_a,
                rules_b=entry.rules_b
            ))
        return assignments

    def _wait_for_games(self, total_games: int):
        received = 0
        while received < total_games:
            self.ack_socket.recv()  # блокируемся до следующего подтверждения
            received += 1
            print(f"[TM] {received}/{total_games} games", end="\r")
        print(f"\n[TM] All {total_games} games done.")

    def _print_match_summary(self, entry: TournamentEntry, results: list[dict]):
        a_score = sum(r["result_for_a"] for r in results)
        b_score = len(results) - a_score
        print(
            f"\n[TM] === Match summary: {entry.agent_a} vs {entry.agent_b} ===\n"
            f"     {entry.agent_a}: {a_score:.1f}   {entry.agent_b}: {b_score:.1f}   "
            f"({len(results)} games, komi={entry.rules_a.fixed_komi})"
        )

    def _set_tournament_models(self, name_a: str, name_b: str):
        """Блокирующе отправляет SET_TOURNAMENT_MODELS и ждёт OK."""
        socket = self.context.socket(zmq.REQ)
        socket.connect(MODEL_PROVIDER_ADDRESS)
        socket.send_json({
            "command": "SET_TOURNAMENT_MODELS",
            "model_a": name_a,
            "model_b": name_b,
        })
        reply = socket.recv_string()
        socket.close()

        if reply != "OK":
            raise RuntimeError(f"[TM] ModelProvider вернул ошибку: {reply}")

        print(f"[TM] Models set: A={name_a}, B={name_b}")

    def run(self):

        os.makedirs(DATASETS_DIR, exist_ok=True)

        # Запускаем сервисные потоки
        time.sleep(1)  # даём потокам забиндиться

        schedule = self._load_schedule()

        for idx, entry in enumerate(schedule):
            print(f"\n[TM] ══════════════════════════════════════")
            print(f"[TM] Tournament {idx+1}/{len(schedule)}: "
                  f"{entry.agent_a} vs {entry.agent_b}, "
                  f"komi={entry.rules_a.fixed_komi}, games={entry.total_games}")
            print(f"[TM] ══════════════════════════════════════")

            # 1. Загружаем модели в провайдер
            self._set_tournament_models(entry.agent_a, entry.agent_b)

            # 2. Ждём, пока все инференс-сервисы подтянут новые модели
            print(f"[TM] Waiting {MODEL_SWITCH_DELAY_SEC}s for inference services to reload...")
            time.sleep(MODEL_SWITCH_DELAY_SEC)

            # 4. Раздаём задания воркерам
            assignments = self._distribute_match(entry)
            total_assigned = sum(a.games_to_generate for a in assignments)
            print(f"[TM] Distributing {total_assigned} games across {len(assignments)} workers")

            for idx, assignment in enumerate(assignments):
                self.worker_ctrl.send_assignment(idx, assignment)

            # 5. Ждём завершения всех игр
            self._wait_for_games(total_assigned)


        # Останавливаем всё
        print("\n[TM] Schedule complete. Stopping workers...")
        self.worker_ctrl.stop_all()
        print("[TM] Done.")

if __name__ == "__main__":
    tm = TournamentManager()
    tm.run()
