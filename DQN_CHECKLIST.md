# LR5: DQN (Deep Q-Network)

## Чек-лист выполнения

### 1. Теоретическая часть
- [ ] Изучить основы обучения с подкреплением (RL)
- [ ] Понять Q-learning и табличный метод
- [ ] Изучить архитектуру DQN
- [ ] Разобраться с Experience Replay
- [ ] Понять Target Network

### 2. Окружение (Environment)
- [ ] Выбрать задачу (CartPole, MountainCar или свою)
- [ ] Установить gymnasium (бывший OpenAI Gym)
- [ ] Проверить работу окружения

### 3. Реализация DQN
- [ ] Создать класс нейросети (Q-network)
- [ ] Реализовать Experience Replay Buffer
- [ ] Реализовать epsilon-greedy исследование
- [ ] Написать алгоритм обучения
- [ ] Добавить Target Network обновление

### 4. Обучение
- [ ] Запустить обучение
- [ ] Сохранить чекпоинты
- [ ] Настроить гиперпараметры

### 5. Тестирование и визуализация
- [ ] Оценить агента на тесте
- [ ] Построить графики (reward per episode)
- [ ] Сохранить видео игры агента

### 6. Отчет
- [ ] Сравнить с baseline (random, tabular Q)
- [ ] Показать результаты
- [ ] Оформить отчет

---

## Гиперпараметры (примерные)

| Параметр | Значение |
|----------|----------|
| Batch size | 64 |
| Learning rate | 0.001 |
| Gamma (discount) | 0.99 |
| Epsilon decay | 0.995 |
| Target update | каждые 1000 шагов |
| Episodes | 500-1000 |

---

## Структура проекта

```
lr5-dqn/
├── main.py           # Основной код
├── dqn/              # Модуль DQN
│   ├── __init__.py
│   ├── network.py    # Q-network
│   ├── buffer.py     # Replay buffer
│   └── agent.py      # Агент
├── plots/            # Графики
└── README.md
```

---

## Дополнительные улучшения (опционально)

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Rainbow DQN