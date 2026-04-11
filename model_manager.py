import os
from typing import Union

import torch
import torch.nn as nn

# Подтягиваем конфигурацию, чтобы Manager знал, как инициализировать модель
from config import NUM_RES_BLOCKS


def init_weights(model: nn.Module):
    """
    Применяет Kaiming и Xavier инициализацию к модели.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Инициализация Kaiming для сверточных слоев
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Инициализация Xavier для линейных слоев
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # Стандартная инициализация для BatchNorm
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ModelManager:
    """
    Управляет созданием, сохранением и загрузкой моделей и их чекпоинтов.
    """

    def __init__(self, model_class, save_dir="checkpoints/", device="cpu"):
        """
        Args:
            model_class: Класс вашей нейросети (например, RLAgent).
            save_dir: Директория для сохранения чекпоинтов.
            device: Устройство для вычислений.
        """
        self.model_class = model_class
        self.save_dir = save_dir
        self.device = device
        os.makedirs(self.save_dir, exist_ok=True)

    def create_new_model(self) -> nn.Module:
        """
        Создает новый экземпляр модели с правильной случайной инициализацией.
        """
        print("Creating a new model with random weights...")
        model = self.model_class().to(self.device)
        init_weights(model)

        # Сразу переводим в channels_last (полезно для MCTS/обучения)
        model = model.to(memory_format=torch.channels_last)
        return model

    def save_checkpoint(self, model: nn.Module, optimizer: Union[torch.optim.Optimizer, None], metadata: dict,
                        filename: str):
        """
        Сохраняет полный чекпоинт для возобновления обучения.
        """
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filename: str, model: nn.Module, optimizer: torch.optim.Optimizer = None):
        """
        Загружает чекпоинт в существующие модель и оптимизатор (для дообучения).
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # map_location гарантирует, что тензоры загрузятся на нужный девайс
        checkpoint = torch.load(filepath, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('metadata', {})

    def load_model_weights(self, filename: str) -> nn.Module:
        """
        Создает новую модель и загружает в нее только веса (для инференса/MCTS).
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Создаем пустую модель и переносим на девайс в нужном формате
        model = self.model_class().to(self.device, memory_format=torch.channels_last)

        # Загружаем чекпоинт
        checkpoint = torch.load(filepath, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # ВАЖНО: Для инференса MCTS обязательно нужно перевести модель в eval().
        # Иначе BatchNorm будет менять свою статистику при каждом вызове предсказания,
        # что полностью сломает предсказания агента!
        model.eval()

        print(f"Model weights loaded from {filepath}")
        return model


def create_new_model(model_manager: ModelManager, name: str = "agent_v0.pth"):
    # Вызов создает модель уже на нужном device и применяет инициализацию
    challenger_model = model_manager.create_new_model()

    # Сохраняем чекпоинт с метаданными (пока без оптимизатора)
    initial_metadata = {'games_played': 0, 'version': 0}

    # Рекомендуется использовать расширение .pth или .pt для файлов чекпоинтов
    if not name.endswith('.pth') and not name.endswith('.pt'):
        name += '.pth'

    model_manager.save_checkpoint(challenger_model, None, initial_metadata, name)

    print(f"Начальная модель '{name}' создана и сохранена.")
    return challenger_model