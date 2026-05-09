.PHONY: help install train demo test clean

GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
RESET := $(shell tput sgr0)

help:
	@echo "$(BLUE)LR5: DQN (Deep Q-Network)$(RESET) - Обучение с подкреплением"
	@echo ""
	@echo "$(GREEN)Доступные команды:$(RESET)"
	@echo "  $(YELLOW)make install$(RESET)   - Установить зависимости"
	@echo "  $(YELLOW)make train$(RESET)     - Обучить агента"
	@echo "  $(YELLOW)make demo$(RESET)      - Запустить визуализацию"
	@echo "  $(YELLOW)make test$(RESET)      - Быстрый тест агента"
	@echo "  $(YELLOW)make clean$(RESET)     - Очистить временные файлы"

install:
	pip install -r requirements.txt

train:
	python3 main.py

demo:
	python3 demo.py

test:
	@python3 -c "from dqn import DQNAgent; agent = DQNAgent(state_dim=4, action_dim=2); agent.load('dqn_model.pth'); print('$(GREEN)Модель загружена OK$(RESET)')"

clean:
	rm -rf plots/*.png dqn_model.pth __pycache__ dqn/__pycache__ src/__pycache__ .pytest_cache
	@echo "$(GREEN)Очистка завершена$(RESET)"