"""
Сравнение L-BFGS с разными значениями параметра m (размер памяти)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from optimization import create_rosenbrock_function, lbfgs


# Параметры варианта 2
A, B, F0, N = 150, 2, 100, 3
f, grad_f, _ = create_rosenbrock_function(A, B, F0)

# Начальная точка
x0 = np.array([0.0, 0.0, 0.0])

# Тестируем разные значения m
m_values = [3, 5, 10, 15, 20, 30, 50, 100]
results = {}

print("=" * 70)
print("СРАВНЕНИЕ L-BFGS С РАЗНЫМИ ЗНАЧЕНИЯМИ ПАРАМЕТРА m")
print("=" * 70)
print(f"\nФункция Розенброка: n={N}, a={A}, b={B}, f0={F0}")
print(f"Начальная точка: x0 = {x0}")
print("=" * 70)

for m in m_values:
    start_time = time.time()
    result = lbfgs(f, grad_f, x0, m=m, max_iter=1000)
    elapsed = time.time() - start_time

    grad_norm = np.linalg.norm(grad_f(result["x_star"]))

    results[m] = {
        "iterations": result["iterations"],
        "time": elapsed,
        "f_star": result["f_star"],
        "grad_norm": grad_norm,
        "x_star": result["x_star"],
    }

    print(f"\nm = {m:3d}:")
    print(f"  Итераций: {result['iterations']:4d}")
    print(f"  Время:    {elapsed:.4f} сек")
    print(f"  f(x*):    {result['f_star']:.10f}")
    print(f"  ||∇f||:   {grad_norm:.2e}")

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Итерации от m
ax = axes[0, 0]
m_list = list(results.keys())
iterations = [results[m]["iterations"] for m in m_list]
ax.plot(m_list, iterations, "bo-", linewidth=2, markersize=8)
ax.set_xlabel("m (размер памяти)")
ax.set_ylabel("Число итераций")
ax.set_title("Зависимость числа итераций от m")
ax.grid(True, alpha=0.3)
ax.set_xticks(m_list)

# 2. Время от m
ax = axes[0, 1]
times = [results[m]["time"] for m in m_list]
ax.plot(m_list, times, "ro-", linewidth=2, markersize=8)
ax.set_xlabel("m (размер памяти)")
ax.set_ylabel("Время (сек)")
ax.set_title("Зависимость времени выполнения от m")
ax.grid(True, alpha=0.3)
ax.set_xticks(m_list)

# 3. Точность от m
ax = axes[1, 0]
grad_norms = [results[m]["grad_norm"] for m in m_list]
ax.semilogy(m_list, grad_norms, "go-", linewidth=2, markersize=8)
ax.set_xlabel("m (размер памяти)")
ax.set_ylabel("||∇f(x*)|| (лог)")
ax.set_title("Зависимость точности от m")
ax.grid(True, alpha=0.3)
ax.set_xticks(m_list)

# 4. Сводный график (нормализованный)
ax = axes[1, 1]
iter_norm = [i / max(iterations) for i in iterations]
time_norm = [t / max(times) for t in times]

ax.plot(m_list, iter_norm, "bo-", label="Итерации (норм.)", linewidth=2, markersize=8)
ax.plot(m_list, time_norm, "ro-", label="Время (норм.)", linewidth=2, markersize=8)
ax.set_xlabel("m (размер памяти)")
ax.set_ylabel("Значение (норм.)")
ax.set_title("Сравнение итераций и времени")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(m_list)

plt.tight_layout()
plt.savefig("plots/lbfgs_m_comparison.png", dpi=150, bbox_inches="tight")
print("\n" + "=" * 70)
print("График сохранён в 'plots/lbfgs_m_comparison.png'")
print("=" * 70)
plt.show()

# Выводы
print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)

best_iter = min(results.keys(), key=lambda m: results[m]["iterations"])
best_time = min(results.keys(), key=lambda m: results[m]["time"])
best_acc = min(results.keys(), key=lambda m: results[m]["grad_norm"])

print(
    f"\n✓ Лучший по итерациям: m={best_iter} ({results[best_iter]['iterations']} итераций)"
)
print(f"✓ Лучший по времени:   m={best_time} ({results[best_time]['time']:.4f} сек)")
print(
    f"✓ Лучший по точности:  m={best_acc} (||∇f||={results[best_acc]['grad_norm']:.2e})"
)

print("\n📌 Рекомендация:")
print("   • Для небольших задач (n < 100): m = 10-20")
print("   • Для средних задач (100 < n < 1000): m = 15-30")
print("   • Для больших задач (n > 1000): m = 5-15")
print("   • Золотая середина: m = 10-20")
print("=" * 70)


"""
Для этой задачи (n=3):
    - Разница между m=5 и m=100 минимальна (все сходятся за 18 итераций)
    - Это потому, что задача очень маленькая (всего 3 переменные)
    - После m≈5-10 дополнительная память не даёт преимущества

Когда m важно:
    - Для больших задач (n > 1000) m влияет на сходимость
    - m=10-20 — оптимальный выбор для большинства задач
    - m=30-50 — для очень сложных задач с плохой обусловленностью

Почему m=10 по умолчанию:
    - Хороший баланс между памятью и точностью
    - Работает хорошо для большинства задач
    - Не требует настройки
"""
