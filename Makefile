.PHONY: help install train train-qnetwork train-dueling train-bn train-all demo test clean mlflow

GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
RESET := $(shell tput sgr0)

help:
	@echo "$(BLUE)LR5: DQN (Deep Q-Network)$(RESET) - Обучение с подкреплением"
	@echo ""
	@echo "$(GREEN)Основные команды:$(RESET)"
	@echo "  $(YELLOW)make install$(RESET)                - Установить зависимости"
	@echo "  $(YELLOW)make train$(RESET)                  - Обучить агента (базовый DQN, 50 эпизодов)"
	@echo "  $(YELLOW)make mlflow$(RESET)                 - Запустить MLflow UI"
	@echo "  $(YELLOW)make demo$(RESET)                   - Запустить визуализацию"
	@echo "  $(YELLOW)make test$(RESET)                   - Быстрый тест агента"
	@echo "  $(YELLOW)make clean$(RESET)                  - Очистить временные файлы"
	@echo ""
	@echo "$(GREEN)Архитектуры нейросетей:$(RESET)"
	@echo "  $(YELLOW)make train-qnetwork$(RESET)         - Базовая DQN (3 слоя, ReLU)"
	@echo "  $(YELLOW)make train-dueling$(RESET)          - Dueling DQN (разделение V и A)"
	@echo "  $(YELLOW)make train-bn$(RESET)               - DQN с BatchNorm и Dropout"
	@echo "  $(YELLOW)make train-lstm$(RESET)             - DQN с LSTM (память)"
	@echo "  $(YELLOW)make train-all$(RESET)              - Обучить все 4 модели для сравнения"
	@echo ""
	@echo "$(GREEN)Параметры обучения:$(RESET)"
	@echo "  EPISODES, BATCH_SIZE, LR, GAMMA, EPSILON_DECAY, EPSILON_MIN"
	@echo "  HIDDEN_DIM, TARGET_UPDATE, BUFFER_CAPACITY, WARMUP_STEPS, NETWORK"

install:
	pip install -r requirements.txt

train:
	python3 main.py --network $(NETWORK) --episodes $(EPISODES) --batch-size $(BATCH_SIZE) --lr $(LR) --gamma $(GAMMA) --hidden-dim $(HIDDEN_DIM) --warmup-steps $(WARMUP_STEPS) $(if $(NO_MLFLOW),--no-mlflow,)

train-qnetwork:
	@echo "$(GREEN)Обучение базовой DQN сети...$(RESET)"
	python3 main.py --network qnetwork --episodes $(EPISODES) --batch-size $(BATCH_SIZE) --lr $(LR) --gamma $(GAMMA) --hidden-dim $(HIDDEN_DIM)

train-dueling:
	@echo "$(GREEN)Обучение Dueling DQN сети...$(RESET)"
	python3 main.py --network dueling --episodes $(EPISODES) --batch-size $(BATCH_SIZE) --lr $(LR) --gamma $(GAMMA) --hidden-dim $(HIDDEN_DIM)

train-bn:
	@echo "$(GREEN)Обучение DQN с BatchNorm...$(RESET)"
	python3 main.py --network bn --episodes $(EPISODES) --batch-size $(BATCH_SIZE) --lr $(LR) --gamma $(GAMMA) --hidden-dim $(HIDDEN_DIM)

train-lstm:
	@echo "$(GREEN)Обучение DQN с LSTM...$(RESET)"
	python3 main.py --network lstm --episodes $(EPISODES) --batch-size $(BATCH_SIZE) --lr $(LR) --gamma $(GAMMA) --hidden-dim $(HIDDEN_DIM)

train-all:
	@echo "$(GREEN)Обучение всех архитектур для сравнения...$(RESET)"
	@echo ""
	@echo "$(YELLOW)=== 1. QNetwork (базовый DQN) ===$(RESET)"
	python3 main.py --network qnetwork --episodes $(EPISODES) --batch-size 32 --lr 0.003 --gamma 0.99 --hidden-dim 128
	@echo ""
	@echo "$(YELLOW)=== 2. DuelingQNetwork ===$(RESET)"
	python3 main.py --network dueling --episodes $(EPISODES) --batch-size 32 --lr 0.003 --gamma 0.99 --hidden-dim 128
	@echo ""
	@echo "$(YELLOW)=== 3. QNetwork с BatchNorm ===$(RESET)"
	python3 main.py --network bn --episodes $(EPISODES) --batch-size 32 --lr 0.003 --gamma 0.99 --hidden-dim 128
	@echo ""
	@echo "$(YELLOW)=== 4. QNetwork с LSTM ===$(RESET)"
	python3 main.py --network lstm --episodes $(EPISODES) --batch-size 32 --lr 0.003 --gamma 0.99 --hidden-dim 128
	@echo ""
	@echo "$(GREEN)Обучение всех моделей завершено!$(RESET)"

demo:
	python3 demo.py --model $(MODEL) --episodes $(EPISODES) --delay $(DELAY)

test:
	@python3 -c "from dqn import DQNAgent; agent = DQNAgent(state_dim=4, action_dim=2); agent.load('$(MODEL)'); print('$(GREEN)Модель загружена OK$(RESET)')"

mlflow:
	@echo "$(GREEN)Запуск MLflow UI...$(RESET)"
	@echo "$(YELLOW)Откройте: http://127.0.0.1:5000/#/experiments/1/runs?columns=metrics.episode_reward,metrics.episode_loss,metrics.episode_q,metrics.epsilon$(RESET)"
	mlflow ui --host 127.0.0.1 --port 5000

clean:
	rm -rf plots/*.png dqn_model.pth __pycache__ dqn/__pycache__ src/__pycache__ .pytest_cache
	@echo "$(GREEN)Очистка завершена$(RESET)"

EPISODES ?= 50
BATCH_SIZE ?= 32
LR ?= 0.003
GAMMA ?= 0.99
EPSILON_DECAY ?= 0.98
EPSILON_MIN ?= 0.01
HIDDEN_DIM ?= 128
TARGET_UPDATE ?= 50
BUFFER_CAPACITY ?= 10000
WARMUP_STEPS ?= 1000
NETWORK ?= qnetwork
MODEL ?= dqn_model.pth
DELAY ?= 0.02