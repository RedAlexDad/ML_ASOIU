# Методы машинного обучения (ММО)

Репозиторий с заданиями по курсу "Методы машинного обучения в АСОИУ", ИУ5Ц-21М, МГТУ им. Баумана.

Все работы выполнены в виде Jupyter Notebook (Python).

## Структура

```
├── main/                               # Этот README
├── lr1-feature-processing/             # LR1 - Обработка признаков (ч.1)
├── lr2-feature-processing-ev/          # LR2 - Обработка признаков (ч.2)
├── lr3-multidimensional-search/        # LR3 - Многомерный поиск
├── lr4-constrained-optimization/       # LR4 - Условная оптимизация
├── lr5-dqn-reinforcement-learning/     # LR5 - Глубокое Q-обучение (DQN)
└── rk1-data-processing/                # РК1
```

## Лабораторные работы

| # | Тема | Ветка |
|---|------|-------|
| LR1 | Обработка признаков (пропуски, кодирование, нормализация) | [lr1-feature-processing](https://github.com/RedAlexDad/ML_ASOIU/tree/lr1-feature-processing) |
| LR2 | Обработка признаков (масштабирование, выбросы, отбор признаков) | [lr2-feature-processing-ev](https://github.com/RedAlexDad/ML_ASOIU/tree/lr2-feature-processing-ev) |
| LR3 | Методы многомерного поиска (FR, PR, DFP, BFGS, L-BFGS) | [lr3-multidimensional-search](https://github.com/RedAlexDad/ML_ASOIU/tree/lr3-multidimensional-search) |
| LR4 | Методы поиска условного экстремума | [lr4-constrained-optimization](https://github.com/RedAlexDad/ML_ASOIU/tree/lr4-constrained-optimization) |
| LR5 | Глубокое Q-обучение (DQN) — обучение с подкреплением | [lr5-dqn-reinforcement-learning](https://github.com/RedAlexDad/ML_ASOIU/tree/lr5-dqn-reinforcement-learning) |

## Рубежные контроли

| # | Тема | Ветка |
|---|------|-------|
| РК1 | Предобработка данных (EDA, выбросы, Target Encoding, MaxAbs Scaling) | [rk1-data-processing](https://github.com/RedAlexDad/ML_ASOIU/tree/rk1-data-processing) |

## Переключение между ветками

```bash
git checkout lr1-feature-processing         # LR1
git checkout lr2-feature-processing-ev      # LR2
git checkout lr3-multidimensional-search    # LR3
git checkout lr4-constrained-optimization   # LR4
git checkout lr5-dqn-reinforcement-learning # LR5
git checkout rk1-data-processing            # РК1
```

## GitHub Pages

Сайт с РК1: **https://redalexdad.github.io/ML_ASOIU/**

## Зависимости

```
numpy pandas matplotlib seaborn scikit-learn scipy pytorch
```