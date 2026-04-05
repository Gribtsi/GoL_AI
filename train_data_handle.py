from typing import List, Dict, Any
import json
import numpy as np
import os
import glob
from datetime import datetime
from game import NumpyEncoder


def save_training_data(save_dir: str, training_data: List) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)


    print(f"Попытка сохранить {len(training_data)} записей в файл: {filepath}")

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Используем наш кастомный энкодер с помощью аргумента `cls`
            json.dump(training_data, f, cls=NumpyEncoder, indent=2)

        print(f"✅ Данные успешно сохранены.")
        return True

    except Exception as e:
        print(f"❌ Произошла ошибка при сохранении файла: {e}")
        return False



def load_and_combine_data(filepaths: List[str]) -> List[Dict[str, Any]]:
    """
    Загружает тренировочные данные из нескольких JSON-файлов,
    объединяет их в один список и преобразует списки обратно в массивы NumPy.

    Args:
        filepaths: Список путей к JSON-файлам с данными.

    Returns:
        Единый список словарей с тренировочными данными.
    """
    combined_training_data = []
    total_records = 0
    successful_files = 0

    print(f"Начинаем загрузку данных из {len(filepaths)} файлов...")

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"⚠️ Файл не найден, пропущен: {filepath}")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Загружаем данные из одного файла
                data_from_file = json.load(f)

                # Проходим по каждой записи и конвертируем нужные поля в NumPy
                for record in data_from_file:
                    record['state_tensor'] = np.array(record['state_tensor'], dtype=np.float32)
                    record['mcts_policy'] = np.array(record['mcts_policy'], dtype=np.float32)
                    record['territories'] = np.array(record['territories'], dtype=np.float32)
                    # Поле 'value' уже является числом (float), его преобразовывать не нужно

                # Добавляем обработанные данные в общий список
                combined_training_data.extend(data_from_file)

                # Статистика
                num_records = len(data_from_file)
                total_records += num_records
                successful_files += 1
                print(f"  ✅ Файл '{os.path.basename(filepath)}' успешно загружен ({num_records} записей).")

        except json.JSONDecodeError as e:
            print(f"❌ Ошибка декодирования JSON в файле '{filepath}': {e}")
        except Exception as e:
            print(f"❌ Произошла непредвиденная ошибка при обработке файла '{filepath}': {e}")

    print("-" * 60)
    print(f"Загрузка завершена.")
    print(f"Успешно обработано файлов: {successful_files} из {len(filepaths)}")
    print(f"Общее количество тренировочных записей: {total_records}")
    print("-" * 60)

    return combined_training_data


def load_and_combine_data_by_count(directory: str, target_count: int) -> List[Dict[str, Any]]:
    """
    Загружает тренировочные данные из JSON-файлов в указанной директории,
    начиная с самых новых, пока не наберет необходимое количество записей.

    Args:
        directory: Путь к директории с файлами training_data_*.json.
        target_count: Необходимое количество записей (может быть превышено
                      за счет размера последнего добавленного файла).

    Returns:
        Единый список словарей с тренировочными данными (NumPy массивы).
    """
    combined_training_data = []

    # Ищем все файлы, соответствующие шаблону в папке
    search_pattern = os.path.join(directory, "training_data_*.json")
    filepaths = glob.glob(search_pattern)

    # Сортируем файлы по времени модификации (mtime) ПО УБЫВАНИЮ (от новых к старым)
    # Знак минус перед os.path.getmtime делает сортировку descending
    filepaths.sort(key=lambda x: -os.path.getmtime(x))

    print(f"Найдено {len(filepaths)} файлов. Цель: {target_count} записей.")

    successful_files = 0
    total_records = 0

    for filepath in filepaths:
        # Если уже набрали нужное количество, прекращаем чтение новых файлов
        if total_records >= target_count:
            break

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data_from_file = json.load(f)

                # Конвертируем нужные поля в NumPy массивы
                for record in data_from_file:
                    record['state_tensor'] = np.array(record['state_tensor'], dtype=np.float32)
                    record['mcts_policy'] = np.array(record['mcts_policy'], dtype=np.float32)
                    record['territories'] = np.array(record['territories'], dtype=np.float32)
                    # 'value' остается float

                combined_training_data.extend(data_from_file)

                num_records = len(data_from_file)
                total_records += num_records
                successful_files += 1

                print(f"  ✅ Загружен: '{os.path.basename(filepath)}' | "
                      f"Записей в файле: {num_records} | "
                      f"Всего собрано: {total_records}/{target_count}")

        except json.JSONDecodeError as e:
            print(f"❌ Ошибка декодирования JSON в файле '{os.path.basename(filepath)}': {e}")
        except Exception as e:
            print(f"❌ Непредвиденная ошибка файла '{os.path.basename(filepath)}': {e}")

    print("-" * 60)
    print(f"Загрузка завершена.")
    print(f"Прочитано файлов: {successful_files}")
    print(f"Собрано записей: {total_records} (Цель была: {target_count})")
    if total_records < target_count:
        print(f"⚠️ ВНИМАНИЕ: Во всех файлах не хватило записей для достижения цели!")
    print("-" * 60)

    return combined_training_data