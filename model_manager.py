import os
from typing import Union

import torch
import torch.nn as nn

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

    def __init__(self, model_class, save_dir="checkpoints/", device="cpu", NUM_RES_BLOCKS=15):
        """
        Args:
            model_class: Класс вашей нейросети (например, RLAgent).
            save_dir: Директория для сохранения чекпоинтов.
            device: Устройство для вычислений.
            **model_kwargs: Аргументы для конструктора модели (например, num_res_blocks=7).
        """
        self.model_class = model_class
        self.NUM_RES_BLOCKS = NUM_RES_BLOCKS
        self.save_dir = save_dir
        self.device = device
        os.makedirs(self.save_dir, exist_ok=True)

    def create_new_model(self) -> nn.Module:
        """
        Создает новый экземпляр модели с правильной случайной инициализацией.
        """
        print("Creating a new model with random weights...")
        model = self.model_class(self.NUM_RES_BLOCKS).to(self.device)
        init_weights(model)  # Применяем нашу функцию инициализации
        return model

    def save_checkpoint(self, model: nn.Module, optimizer: Union[ torch.optim.Optimizer, None], metadata: dict, filename: str):
        """
        Сохраняет полный чекпоинт для возобновления обучения.

        Args:
            model: Экземпляр модели.
            optimizer: Экземпляр оптимизатора.
            metadata: Словарь с метаданными (например, {'games_played': 1000}).
            filename: Имя файла (например, 'agent_v1_g1000.pth').
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
        Загружает чекпоинт в существующие модель и оптимизатор.

        Args:
            filename: Имя файла для загрузки.
            model: Экземпляр модели, куда будут загружены веса.
            optimizer: (Опционально) Экземпляр оптимизатора.

        Returns:
            metadata: Загруженные метаданные.
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('metadata', {})

    def load_model_weights(self, filename: str) -> nn.Module:
        """
        Создает новую модель и загружает в нее только веса (для инференса).
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        model = self.model_class(self.NUM_RES_BLOCKS).to(self.device)
        checkpoint = torch.load(filepath, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {filepath}")
        return model


def create_new_model(model_manager: ModelManager, name: str = "agent_v0"):
    current_best_model_id = name
    challenger_model = model_manager.create_new_model()  # create_new_model уже вызывает init_weights

    # Сохраняем чекпоинт с метаданными (пока без оптимизатора)
    initial_metadata = {'games_played': 0, 'version': 0}
    model_manager.save_checkpoint(challenger_model, None, initial_metadata, f"{name}")

    print(f"Начальная модель '{name}' создана и сохранена.")
