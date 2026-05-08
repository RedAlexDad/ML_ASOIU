# LR5: DQN (Deep Q-Network)

Задача: CartPole-v1 (балансировка шеста)

Вариант 2

Студент: Папин А.В., ИУ5Ц-21М

## Запуск

```bash
pip install gymnasium torch matplotlib
python main.py
```

## Архитектура

- Q-network: 3 слоя (state_dim → 128 → 128 → action_dim)
- Replay Buffer: 10000
- Target Network: обновление каждые 100 шагов
- Epsilon-greedy: decay от 1.0 до 0.01

## Результаты

(в процессе...)