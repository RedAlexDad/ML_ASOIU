"""
Тест: Метод барьерных функций для задачи с оптимумом ВНУТРИ области

Проблема варианта 2: оптимум (1,1,1) лежит НА ГРАНИЦЕ g1(x) = 0
Решение: изменим ограничения так, чтобы оптимум был ВНУТРИ области
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from constrained_optimization import (
    create_rosenbrock_function,
    check_constraints,
    barrier_method,
    modified_lagrange_method,
    penalty_method
)


def test_barrier_with_interior_optimum():
    """
    Тест: Метод барьерных функций с оптимумом ВНУТРИ области
    
    Создадим ограничения где (1,1,1) лежит ВНУТРИ, а не на границе:
    g1(x) = x1² + x2² - x3 - 0.5 ≤ 0  (вместо -1)
    
    При x = (1,1,1): g1 = 1 + 1 - 1 - 0.5 = 0.5 > 0  -- всё ещё на границе!
    
    Нужно: g1(x) = x1² + x2² - x3 - 1.5 ≤ 0
    При x = (1,1,1): g1 = 1 + 1 - 1 - 1.5 = -0.5 < 0  -- ВНУТРИ!
    """
    print("\n" + "="*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ С ОПТИМУМОМ ВНУТРИ ОБЛАСТИ")
    print("="*70)
    
    # Параметры варианта 2
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    x0 = np.array([0.5, 0.5, 0.5])
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    # Изменяем g1 так, чтобы (1,1,1) было ВНУТРИ области
    # g1(x) = x1² + x2² - x3 - 1.5 ≤ 0
    # При x=(1,1,1): g1 = 1+1-1-1.5 = -0.5 < 0  ✓ ВНУТРИ!
    def g1_interior(x):
        return x[0]**2 + x[1]**2 - x[2] - 1.5
    
    constraints_interior = [
        g1_interior,
        lambda x: -x[0],  # g2: x1 ≥ 0
        lambda x: -x[1],  # g3: x2 ≥ 0
        lambda x: -x[2]   # g4: x3 ≥ 0
    ]
    
    # Проверка что (1,1,1) внутри области
    x_opt = np.array([1.0, 1.0, 1.0])
    print(f"\nПроверка точки (1,1,1):")
    for i, g in enumerate(constraints_interior):
        g_val = g(x_opt)
        status = "✓ ВНУТРИ" if g_val < 0 else "✗ НА ГРАНИЦЕ/СНАРУЖИ"
        print(f"  g{i+1}(1,1,1) = {g_val:.4f}  {status}")
    
    # Scipy решение
    print("\n" + "-"*70)
    print("SCIPY SLSQP РЕШЕНИЕ")
    print("-"*70)
    
    scipy_constraints = [
        NonlinearConstraint(lambda x: x[0]**2 + x[1]**2 - x[2] - 1.5, -np.inf, 0),
        NonlinearConstraint(lambda x: -x[0], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[1], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[2], -np.inf, 0)
    ]
    
    result_scipy = minimize(f, x0, method='SLSQP', jac=grad_f, 
                           constraints=scipy_constraints,
                           options={'ftol': 1e-8, 'maxiter': 1000})
    
    print(f"\nScipy решение:")
    print(f"  x* = [{result_scipy.x[0]:.8f}, {result_scipy.x[1]:.8f}, {result_scipy.x[2]:.8f}]")
    print(f"  f(x*) = {result_scipy.fun:.10f}")
    print(f"  Успех: {result_scipy.success}")
    
    # Наш метод барьерных функций
    print("\n" + "-"*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ")
    print("-"*70)
    
    result_barrier = barrier_method(
        f, grad_f, x0, constraints_interior,
        r0=1.0, C=10.0, eps=1e-6, max_outer_iter=50,
        barrier_type='log'
    )
    
    x_barrier = result_barrier['x_star']
    f_barrier = result_barrier['f_star']
    
    print(f"\nМетод барьерных функций:")
    print(f"  x* = [{x_barrier[0]:.8f}, {x_barrier[1]:.8f}, {x_barrier[2]:.8f}]")
    print(f"  f(x*) = {f_barrier:.10f}")
    print(f"  Итераций: {result_barrier['iterations']}")
    print(f"  Допустимо: {result_barrier['feasible']}")
    
    # Сравнение
    print("\n" + "-"*70)
    print("СРАВНЕНИЕ")
    print("-"*70)
    
    x_error = np.linalg.norm(x_barrier - result_scipy.x)
    f_error = abs(f_barrier - result_scipy.fun)
    
    print(f"  ||x_barrier - x_scipy|| = {x_error:.2e}")
    print(f"  |f_barrier - f_scipy| = {f_error:.2e}")
    
    # Проверка
    print("\n" + "="*70)
    if f_error < 1e-4:
        print("✅ МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ РАБОТАЕТ КОГДА ОПТИМУМ ВНУТРИ!")
    else:
        print("❌ МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ ВСЁ ЕЩЁ НЕ РАБОТАЕТ")
    print("="*70)
    
    # Для сравнения: метод Лагранжа
    print("\n" + "-"*70)
    print("МЕТОД МНОЖИТЕЛЕЙ ЛАГРАНЖА (для сравнения)")
    print("-"*70)
    
    result_lagrange = modified_lagrange_method(
        f, grad_f, x0, constraints_interior,
        r0=1.0, C=2.0, eps=1e-6, max_outer_iter=100
    )
    
    x_lagrange = result_lagrange['x_star']
    f_lagrange = result_lagrange['f_star']
    
    print(f"\nМетод Лагранжа:")
    print(f"  x* = [{x_lagrange[0]:.8f}, {x_lagrange[1]:.8f}, {x_lagrange[2]:.8f}]")
    print(f"  f(x*) = {f_lagrange:.10f}")
    
    f_error_lagrange = abs(f_lagrange - result_scipy.fun)
    print(f"  |f_lagrange - f_scipy| = {f_error_lagrange:.2e}")
    
    return f_error < 1e-4


def test_barrier_with_original_constraints():
    """
    Тест: Метод барьерных функций с оригинальными ограничениями
    (оптимум на границе)
    """
    print("\n" + "="*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ С ОПТИМУМОМ НА ГРАНИЦЕ")
    print("(Оригинальный вариант 2)")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = [
        lambda x: x[0]**2 + x[1]**2 - x[2] - 1,  # g1: оригинальный
        lambda x: -x[0],
        lambda x: -x[1],
        lambda x: -x[2]
    ]
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    x0 = np.array([0.5, 0.5, 0.5])
    
    # Scipy решение
    scipy_constraints = [
        NonlinearConstraint(lambda x: x[0]**2 + x[1]**2 - x[2] - 1, -np.inf, 0),
        NonlinearConstraint(lambda x: -x[0], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[1], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[2], -np.inf, 0)
    ]
    
    result_scipy = minimize(f, x0, method='SLSQP', jac=grad_f,
                           constraints=scipy_constraints,
                           options={'ftol': 1e-8, 'maxiter': 1000})
    
    print(f"\nScipy решение:")
    print(f"  x* = [{result_scipy.x[0]:.8f}, {result_scipy.x[1]:.8f}, {result_scipy.x[2]:.8f}]")
    print(f"  f(x*) = {result_scipy.fun:.10f}")
    
    # Метод барьерных функций
    result_barrier = barrier_method(
        f, grad_f, x0, constraints,
        r0=1.0, C=10.0, eps=1e-6, max_outer_iter=50,
        barrier_type='log'
    )
    
    x_barrier = result_barrier['x_star']
    f_barrier = result_barrier['f_star']
    
    print(f"\nМетод барьерных функций:")
    print(f"  x* = [{x_barrier[0]:.8f}, {x_barrier[1]:.8f}, {x_barrier[2]:.8f}]")
    print(f"  f(x*) = {f_barrier:.10f}")
    
    f_error = abs(f_barrier - result_scipy.fun)
    
    print(f"\n  |f_barrier - f_scipy| = {f_error:.2e}")
    
    print("\n" + "="*70)
    if f_error < 1e-4:
        print("✅ РАБОТАЕТ")
    else:
        print(f"❌ НЕ РАБОТАЕТ (погрешность {f_error:.2f})")
        print("   Оптимум на границе - метод барьерных функций не может достичь")
    print("="*70)
    
    return f_error < 1e-4


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ПРОВЕРКА: ЗАВИСИТ ЛИ МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ ОТ ПОЛОЖЕНИЯ ОПТИМУМА")
    print("="*70)
    
    # Тест 1: Оптимум на границе (оригинальный вариант 2)
    works_boundary = test_barrier_with_original_constraints()
    
    # Тест 2: Оптимум внутри области
    works_interior = test_barrier_with_interior_optimum()
    
    # Итог
    print("\n" + "="*70)
    print("ИТОГ")
    print("="*70)
    print(f"Оптимум НА ГРАНИЦЕ:     {'✅ РАБОТАЕТ' if works_boundary else '❌ НЕ РАБОТАЕТ'}")
    print(f"Оптимум ВНУТРИ области: {'✅ РАБОТАЕТ' if works_interior else '❌ НЕ РАБОТАЕТ'}")
    print("="*70)
    
    if not works_boundary and works_interior:
        print("\n✅ ПОДТВЕРЖДЕНО: Метод барьерных функций работает ТОЛЬКО когда")
        print("   оптимум лежит ВНУТРИ допустимой области, а не на границе!")
    elif works_boundary and works_interior:
        print("\n✅ Метод барьерных функций работает в обоих случаях")
    else:
        print("\n⚠️  Неожиданный результат - требуется анализ")
