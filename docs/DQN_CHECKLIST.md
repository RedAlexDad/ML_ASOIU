# LR5: DQN (Deep Q-Network)

## Чек-лист выполнения

### 1. Теоретическая часть
- [x] Изучить основы обучения с подкреплением (RL)
- [x] Понять Q-learning и табличный метод
- [x] Изучить архитектуру DQN
- [x] Разобраться с Experience Replay
- [x] Понять Target Network

### 2. Окружение (Environment)
- [x] Выбрать задачу (CartPole-v1)
- [x] Установить gymnasium (бывший OpenAI Gym)
- [x] Проверить работу окружения

### 3. Реализация DQN
- [x] Создать класс нейросети (Q-network) - 3 архитектуры
- [x] Реализовать Experience Replay Buffer
- [x] Реализовать epsilon-greedy исследование
- [x] Написать алгоритм обучения
- [x] Добавить Target Network обновление

### 4. Обучение
- [x] Запустить обучение (3 архитектуры)
- [x] Сохранить в MLflow
- [x] Настроить гиперпараметры

### 5. Тестирование и визуализация
- [x] Оценить агента на тесте (eval_episodes)
- [x] Построить графики (training_analysis, training_progress, detailed_analysis)
- [ ] Сохранить видео игры агента

### 6. Отчет
- [x] Сравнить архитектуры (QNetwork vs Dueling vs BN)
- [x] Показать результаты (RESULTS.md)
- [x] Оформить отчет (README + RESULTS.md)

---

## Гперпараметры (финальные)

| Параметр | Значение |
|----------|----------|
| Batch size | 32 |
| Learning rate | 0.003 |
| Gamma (discount) | 0.99 |
| Epsilon decay | 0.98 |
| Target update | каждые 50 шагов |
| Episodes | 50-100 |
| Hidden dim | 128 |
| Buffer capacity | 10000 |

---

## Результаты обучения

| Архитектура | Средняя награда (оценка) | Время |
|-------------|--------------------------|-------|
| **QNetwork** | **500.0** (максимум!) | 32.9 сек |
| QNetworkBN | 403.7 | 27.4 сек |
| DuelingQNetwork | 126.4 | 19.7 сек |

**Лучшая модель:** QNetwork (базовый DQN)

---

## Структура проекта

```
ML_ASOIU/
├── main.py              # Основной код обучения
├── Makefile             # Команды make train-*
├── demo.py              # Демонстрация агента
├── requirements.txt     # Зависимости
├── RESULTS.md           # Результаты сравнения
├── dqn/                 # Модуль DQN
│   ├── __init__.py
│   ├── network.py       # QNetwork, DuelingQNetwork, QNetworkBN
│   ├── buffer.py        # ReplayBuffer
│   └── agent.py         # DQNAgent
├── src/                 # Вспомогательные модули
│   ├── train.py         # Обучение
│   ├── evaluation.py    # Оценка агента
│   └── visualization.py # Построение графиков
└── docs/                # Документация
    └── DQN_CHECKLIST.md # Этот чеклист
```

---

## Дополнительные улучшения (реализовано)

- [x] Dueling DQN
- [ ] Double DQN
- [ ] Prioritized Experience Replay
- [ ] Rainbow DQN

---

## Запуск

```bash
# Установка зависимостей
make install

# Обучение базового DQN
make train-qnetwork EPISODES=50

# Обучение Dueling DQN
make train-dueling EPISODES=50

# Обучение с BatchNorm
make train-bn EPISODES=50

# Обучение всех архитектур
make train-all

# Запуск MLflow UI
make mlflow
```