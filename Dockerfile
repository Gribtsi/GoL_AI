# Используем официальный образ PyTorch с поддержкой CUDA
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь твой код в контейнер
COPY . .

# По умолчанию контейнер ничего не делает, команду запуска задаст docker-compose
CMD ["python"]