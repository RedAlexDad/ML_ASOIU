.PHONY: help install train demo test clean

help:
	@echo "LR5: DQN (Deep Q-Network) - Обучение с подкреплением"
	@echo ""
	@echo "Доступные команды:"
	@echo "  make install    - Установить зависимости"
	@echo "  make train      - Обучить агента"
	@echo "  make demo       - Запустить визуализацию"
	@echo "  make test       - Быстрый тест агента"
	@echo "  make clean      - Очистить временные файлы"

install:
	pip install gymnasium torch matplotlib numpy

train:
	python3 main.py

demo:
	python3 demo.py

test:
	python3 -c "from dqn import DQNAgent; agent = DQNAgent(state_dim=4, action_dim=2); agent.load('dqn_model.pth'); print('Model OK')"

clean:
	rm -rf plots/*.png dqn_model.pth __pycache__ dqn/__pycache__ src/__pycache__ .pytest_cache