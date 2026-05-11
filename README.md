# LR5: DQN (Deep Q-Network) — Обучение с подкреплением

Задача: CartPole-v1 (балансировка шеста на тележке)

Студент: Папин А.В., Группа: ИУ5Ц-21М

---

## Описание

Реализация алгоритма Deep Q-Network (DQN) для обучения агента в среде CartPole-v1.
Исследованы 4 архитектуры нейронных сетей.

---

## Архитектуры

| # | Архитектура | Параметры | Описание |
|---|-------------|-----------|----------|
| 1 | **QNetwork** | 17 410 | Базовый DQN (3 слоя, ReLU) |
| 2 | **DuelingQNetwork** | 33 859 | Разделение V и A |
| 3 | **QNetworkBN** | 17 922 | С BatchNorm и Dropout |
| 4 | **QNetworkLSTM** | 68 866 | С LSTM слоем |

---

## Результаты (100 эпизодов)

| Архитектура | Средняя награда | Максимум |
|-------------|----------------|----------|
| **QNetwork** | **500.0** | 500.0 |
| QNetworkLSTM | 235.5 | 252.0 |
| DuelingQNetwork | 176.8 | 193.0 |
| QNetworkBN | 20.1 | 31.0 |

**Лучший результат:** QNetwork — достигает максимума 500 за 100 эпизодов

---

## Запуск

### Установка зависимостей
```bash
make install
```

### Обучение
```bash
# Базовая QNetwork (50 эпизодов)
make train

# Конкретная архитектура
make train-qnetwork
make train-dueling
make train-bn
make train-lstm

# Все архитектуры (100 эпизодов)
make train-all EPISODES=100

# Свои параметры
python main.py --network qnetwork --episodes 100 --lr 0.003
```

### Демонстрация
```bash
# После обучения модель сохраняется в mlruns/
make demo MODEL=mlruns/.../artifacts/data/model.pth NETWORK=qnetwork
```

### MLflow UI
```bash
make mlflow
```

---

## Параметры обучения

| Параметр | Значение |
|----------|----------|
| Hidden dim | 128 |
| Learning rate | 0.003 |
| Gamma | 0.99 |
| Batch size | 32 |
| Replay buffer | 10 000 |
| Target update | 50 шагов |
| Epsilon decay | 0.98 |

---

## Структура проекта

```
├── main.py              # Основная программа обучения
├── demo.py              # Визуализация агента
├── dqn/
│   ├── agent.py         # Класс DQNAgent
│   ├── network.py       # 4 архитектуры сетей
│   └── replay_buffer.py # Буфер опыта
├── docs/
│   └── RESULTS.md       # Результаты экспериментов
└── Makefile            # Команды запуска
```

---

## Зависимости

- Python 3.12+
- PyTorch 2.10.0
- Gymnasium 0.29.1
- MLflow 2.8.0
- NumPy, Matplotlib