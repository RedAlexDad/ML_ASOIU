.PHONY: help install train demo test clean

GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
RESET := $(shell tput sgr0)

help:
	@echo "$(BLUE)LR5: DQN (Deep Q-Network)$(RESET) - Обучение с подкреплением"
	@echo ""
	@echo "$(GREEN)Доступные команды:$(RESET)"
	@echo "  $(YELLOW)make install$(RESET)                - Установить зависимости"
	@echo "  $(YELLOW)make train$(RESET)                  - Обучить агента (50 эпизодов)"
	@echo "  $(YELLOW)make train EPISODES=100$(RESET)     - Обучить с кастомными параметрами"
	@echo "  $(YELLOW)make train MLFLOW=--mlflow$(RESET)  - Обучить с MLflow логгированием"
	@echo "  $(YELLOW)make mlflow$(RESET)                 - Запустить MLflow UI"
	@echo "  $(YELLOW)make demo$(RESET)                   - Запустить визуализацию"
	@echo "  $(YELLOW)make test$(RESET)                   - Быстрый тест агента"
	@echo "  $(YELLOW)make clean$(RESET)                  - Очистить временные файлы"
	@echo ""
	@echo "$(GREEN)Параметры обучения:$(RESET)"
	@echo "  EPISODES, BATCH_SIZE, LR, GAMMA, EPSILON_DECAY, EPSILON_MIN"
	@echo "  HIDDEN_DIM, TARGET_UPDATE, BUFFER_CAPACITY, WARMUP_STEPS"

install:
	pip install -r requirements.txt

train:
	python3 main.py --episodes $(EPISODES) --batch-size $(BATCH_SIZE) --lr $(LR) --gamma $(GAMMA) --hidden-dim $(HIDDEN_DIM) --warmup-steps $(WARMUP_STEPS) $(if $(NO_MLFLOW),--no-mlflow,)

demo:
	python3 demo.py --model $(MODEL) --episodes $(EPISODES) --delay $(DELAY)

test:
	@python3 -c "from dqn import DQNAgent; agent = DQNAgent(state_dim=4, action_dim=2); agent.load('dqn_model.pth'); print('$(GREEN)Модель загружена OK$(RESET)')"

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
HIDDEN_DIM ?= 128
WARMUP_STEPS ?= 1000
MODEL ?= dqn_model.pth
DELAY ?= 0.02