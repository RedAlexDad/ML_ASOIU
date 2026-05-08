"""
Пример использования L-BFGS в PyTorch

Выполнил: Студент группы ИУ5Ц-21М Папин А.В.
Дата: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time


# ============================================================================
# Пример 1: Простая линейная регрессия с L-BFGS
# ============================================================================
def example_linear_regression():
    """Пример линейной регрессии с L-BFGS."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 1: Линейная регрессия с L-BFGS")
    print("=" * 60)
    
    # Данные
    torch.manual_seed(42)
    X = torch.randn(100, 1) * 2
    y = 3 * X + 2 + torch.randn(100, 1) * 0.5
    
    # Модель
    model = nn.Linear(1, 1)
    
    # L-BFGS оптимизатор
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=10
    )
    criterion = nn.MSELoss()
    
    # Обучение
    losses = []
    for epoch in range(100):
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}: Loss = {loss:.6f}')
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Сходимость L-BFGS')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X.numpy(), y.numpy(), label='Данные', alpha=0.6)
    axes[1].plot(X.numpy(), model(X).detach().numpy(), 'r-', label='Предсказание')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].set_title('Результат регрессии')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/pytorch_lbfgs_regression.png', dpi=150)
    print("График сохранён в 'plots/pytorch_lbfgs_regression.png'")
    plt.show()
    
    return model, losses


# ============================================================================
# Пример 2: Сравнение L-BFGS и Adam
# ============================================================================
def compare_lbfgs_adam():
    """Сравнение L-BFGS и Adam на одной задаче."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Сравнение L-BFGS и Adam")
    print("=" * 60)
    
    # Данные
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # Функция для создания модели
    def create_model():
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    # L-BFGS
    print("\nОбучение с L-BFGS...")
    model_lbfgs = create_model()
    optimizer_lbfgs = optim.LBFGS(model_lbfgs.parameters(), lr=1.0, max_iter=20)
    criterion = nn.MSELoss()
    
    def closure():
        optimizer_lbfgs.zero_grad()
        output = model_lbfgs(X)
        loss = criterion(output, y)
        loss.backward()
        return loss
    
    losses_lbfgs = []
    start = time.time()
    for epoch in range(50):
        loss = optimizer_lbfgs.step(closure)
        losses_lbfgs.append(loss.item())
    time_lbfgs = time.time() - start
    
    # Adam
    print("Обучение с Adam...")
    model_adam = create_model()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)
    
    losses_adam = []
    start = time.time()
    for epoch in range(50):
        optimizer_adam.zero_grad()
        output = model_adam(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer_adam.step()
        losses_adam.append(loss.item())
    time_adam = time.time() - start
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)
    print(f"L-BFGS: {time_lbfgs:.4f} сек, Final Loss = {losses_lbfgs[-1]:.6f}")
    print(f"Adam:   {time_adam:.4f} сек, Final Loss = {losses_adam[-1]:.6f}")
    print("=" * 60)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses_lbfgs, 'b-', label='L-BFGS', linewidth=2)
    axes[0].plot(losses_adam, 'r-', label='Adam', linewidth=2)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Сравнение сходимости')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(['L-BFGS', 'Adam'], [time_lbfgs, time_adam], color=['blue', 'red'])
    axes[1].set_ylabel('Время (сек)')
    axes[1].set_title('Время обучения')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/pytorch_lbfgs_vs_adam.png', dpi=150)
    print("График сохранён в 'plots/pytorch_lbfgs_vs_adam.png'")
    plt.show()
    
    return losses_lbfgs, losses_adam


# ============================================================================
# Пример 3: Fine-tuning с L-BFGS
# ============================================================================
def example_fine_tuning():
    """Пример fine-tuning с L-BFGS."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Fine-tuning с L-BFGS")
    print("=" * 60)
    
    # Простая задача: регрессия с предобучением
    torch.manual_seed(42)
    
    # "Предобученная" модель (просто линейная)
    pretrained_model = nn.Linear(5, 1)
    
    # Предобучение на одних данных
    X_pretrain = torch.randn(500, 5)
    y_pretrain = 2 * X_pretrain[:, 0:1] + 0.5 * X_pretrain[:, 1:2] + torch.randn(500, 1) * 0.1
    
    optimizer_pretrain = optim.Adam(pretrained_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Предобучение модели...")
    for epoch in range(100):
        optimizer_pretrain.zero_grad()
        output = pretrained_model(X_pretrain)
        loss = criterion(output, y_pretrain)
        loss.backward()
        optimizer_pretrain.step()
    
    print(f"Loss после предобучения: {loss.item():.6f}")  # type: ignore
    
    # Fine-tuning на новых данных (другое распределение)
    X_finetune = torch.randn(100, 5)
    y_finetune = 3 * X_finetune[:, 0:1] + 0.3 * X_finetune[:, 1:2] + torch.randn(100, 1) * 0.1
    
    # L-BFGS для fine-tuning
    optimizer = optim.LBFGS(
        pretrained_model.parameters(),
        lr=0.1,
        max_iter=50
    )
    
    # Обучение
    losses = []
    print("\nFine-tuning с L-BFGS...")
    for epoch in range(20):
        def closure():
            optimizer.zero_grad()
            output = pretrained_model(X_finetune)
            loss = criterion(output, y_finetune)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}: Loss = {loss:.6f}')
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Fine-tuning с L-BFGS')
    axes[0].grid(True, alpha=0.3)
    
    # Предсказания до и после
    with torch.no_grad():
        # Создаём новую модель для сравнения
        new_model = nn.Linear(5, 1)
        optimizer_new = optim.LBFGS(new_model.parameters(), lr=0.1, max_iter=50)
        
        for epoch in range(100):
            def closure():
                optimizer_new.zero_grad()
                output = new_model(X_finetune)
                loss = criterion(output, y_finetune)
                loss.backward()
                return loss
            optimizer_new.step(closure)
        
        pred_finetuned = pretrained_model(X_finetune)
        pred_from_scratch = new_model(X_finetune)
        
        axes[1].scatter(y_finetune.numpy(), pred_finetuned.numpy(), 
                       alpha=0.6, label='Fine-tuning', color='blue')
        axes[1].scatter(y_finetune.numpy(), pred_from_scratch.numpy(), 
                       alpha=0.6, label='С нуля', color='red')
        axes[1].plot([y_finetune.min(), y_finetune.max()], 
                    [y_finetune.min(), y_finetune.max()], 'k--', label='Идеал')
        axes[1].set_xlabel('Истинное значение')
        axes[1].set_ylabel('Предсказание')
        axes[1].set_title('Сравнение предсказаний')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/pytorch_fine_tuning.png', dpi=150)
    print("График сохранён в 'plots/pytorch_fine_tuning.png'")
    plt.show()
    
    return losses


# ============================================================================
# Основная функция
# ============================================================================
def main():
    """Запуск всех примеров."""
    print("=" * 60)
    print("L-BFGS В PYTORCH: ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ")
    print("=" * 60)
    
    # Создаём директорию для графиков
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Пример 1
    model, losses = example_linear_regression()
    
    # Пример 2
    losses_lbfgs, losses_adam = compare_lbfgs_adam()
    
    # Пример 3
    losses_ft = example_fine_tuning()
    
    print("\n" + "=" * 60)
    print("ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()
