"""
Тест: Метод барьерных функций с ДРУГОЙ функцией (не Розенброка)

Выбираем функцию где:
1. Оптимум лежит ВНУТРИ допустимой области
2. Градиент небольшой
3. Функция гладкая и выпуклая
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from constrained_optimization import (
    check_constraints,
    barrier_method,
    modified_lagrange_method,
    penalty_method,
    projected_gradient_method
)


# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ 1: Простая квадратичная
# ============================================================================

def quadratic_f(x):
    """
    f(x) = (x1-0.5)² + (x2-0.5)² + (x3-0.5)²
    
    Оптимум: x* = (0.5, 0.5, 0.5), f(x*) = 0
    """
    return (x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2

def quadratic_grad(x):
    """Градиент квадратичной функции"""
    return np.array([2*(x[0]-0.5), 2*(x[1]-0.5), 2*(x[2]-0.5)])


# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ 2: Квадратичная с перекрестными членами
# ============================================================================

def quadratic_cross_f(x):
    """
    f(x) = x1² + x2² + x3² + x1*x2 + x2*x3
    
    Оптимум (безусловный): x* = (0, 0, 0), f(x*) = 0
    """
    return x[0]**2 + x[1]**2 + x[2]**2 + x[0]*x[1] + x[1]*x[2]

def quadratic_cross_grad(x):
    """Градиент"""
    return np.array([
        2*x[0] + x[1],
        2*x[1] + x[0] + x[2],
        2*x[2] + x[1]
    ])


# ============================================================================
# ТЕСТОВАЯ ФУНКЦИЯ 3: Экспоненциальная
# ============================================================================

def exp_f(x):
    """
    f(x) = exp(x1) + exp(x2) + exp(x3)
    
    Оптимум стремится к x* = (-∞, -∞, -∞), но с ограничениями будет внутри
    """
    return np.exp(x[0]) + np.exp(x[1]) + np.exp(x[2])

def exp_grad(x):
    """Градиент"""
    return np.array([np.exp(x[0]), np.exp(x[1]), np.exp(x[2])])


def test_barrier_with_quadratic():
    """
    Тест: Метод барьерных функций с квадратичной функцией
    """
    print("\n" + "="*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ: КВАДРАТИЧНАЯ ФУНКЦИЯ")
    print("="*70)
    
    # Ограничения: куб [0, 1] × [0, 1] × [0, 1]
    # Оптимум (0.5, 0.5, 0.5) ВНУТРИ
    constraints = [
        lambda x: x[0] - 1,      # x1 ≤ 1
        lambda x: x[1] - 1,      # x2 ≤ 1
        lambda x: x[2] - 1,      # x3 ≤ 1
        lambda x: -x[0],         # x1 ≥ 0
        lambda x: -x[1],         # x2 ≥ 0
        lambda x: -x[2]          # x3 ≥ 0
    ]
    
    x0 = np.array([0.2, 0.2, 0.2])
    
    print(f"\nФункция: f(x) = (x1-0.5)² + (x2-0.5)² + (x3-0.5)²")
    print(f"Оптимум: x* = (0.5, 0.5, 0.5), f(x*) = 0")
    print(f"Ограничения: 0 ≤ xi ≤ 1 (оптимум ВНУТРИ)")
    
    # Scipy решение
    print("\n" + "-"*70)
    print("SCIPY SLSQP")
    print("-"*70)
    
    scipy_constraints = [
        NonlinearConstraint(lambda x: x[0] - 1, -np.inf, 0),
        NonlinearConstraint(lambda x: x[1] - 1, -np.inf, 0),
        NonlinearConstraint(lambda x: x[2] - 1, -np.inf, 0),
        NonlinearConstraint(lambda x: -x[0], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[1], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[2], -np.inf, 0)
    ]
    
    result_scipy = minimize(quadratic_f, x0, method='SLSQP', jac=quadratic_grad,
                           constraints=scipy_constraints,
                           options={'ftol': 1e-10, 'maxiter': 1000})
    
    print(f"  x* = [{result_scipy.x[0]:.8f}, {result_scipy.x[1]:.8f}, {result_scipy.x[2]:.8f}]")
    print(f"  f(x*) = {result_scipy.fun:.10e}")
    print(f"  Успех: {result_scipy.success}")
    
    # Метод барьерных функций
    print("\n" + "-"*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ")
    print("-"*70)
    
    result_barrier = barrier_method(
        quadratic_f, quadratic_grad, x0, constraints,
        r0=1.0, C=10.0, eps=1e-8, max_outer_iter=100,
        barrier_type='log'
    )
    
    x_barrier = result_barrier['x_star']
    f_barrier = result_barrier['f_star']
    
    print(f"  x* = [{x_barrier[0]:.8f}, {x_barrier[1]:.8f}, {x_barrier[2]:.8f}]")
    print(f"  f(x*) = {f_barrier:.10e}")
    print(f"  Итераций: {result_barrier['iterations']}")
    print(f"  Допустимо: {result_barrier['feasible']}")
    
    # Сравнение
    f_error = abs(f_barrier - result_scipy.fun)
    print(f"\n  |f_barrier - f_scipy| = {f_error:.2e}")
    
    works_barrier = f_error < 1e-4
    
    # Метод Лагранжа
    print("\n" + "-"*70)
    print("МЕТОД МНОЖИТЕЛЕЙ ЛАГРАНЖА")
    print("-"*70)
    
    result_lagrange = modified_lagrange_method(
        quadratic_f, quadratic_grad, x0, constraints,
        r0=1.0, C=2.0, eps=1e-8, max_outer_iter=100
    )
    
    print(f"  x* = [{result_lagrange['x_star'][0]:.8f}, ...]")
    print(f"  f(x*) = {result_lagrange['f_star']:.10e}")
    
    f_error_lagrange = abs(result_lagrange['f_star'] - result_scipy.fun)
    print(f"  |f_lagrange - f_scipy| = {f_error_lagrange:.2e}")
    
    works_lagrange = f_error_lagrange < 1e-4
    
    # Итог
    print("\n" + "="*70)
    print("ИТОГ")
    print("="*70)
    print(f"Метод барьерных функций:  {'✅ РАБОТАЕТ' if works_barrier else '❌ НЕ РАБОТАЕТ'}")
    print(f"Метод Лагранжа:           {'✅ РАБОТАЕТ' if works_lagrange else '❌ НЕ РАБОТАЕТ'}")
    
    return works_barrier, works_lagrange


def test_barrier_with_exp():
    """
    Тест: Метод барьерных функций с экспоненциальной функцией
    """
    print("\n" + "="*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ: ЭКСПОНЕНЦИАЛЬНАЯ ФУНКЦИЯ")
    print("="*70)
    
    # Ограничения: [-2, 0] × [-2, 0] × [-2, 0]
    # Оптимум стремится к (-∞, -∞, -∞), будет на границе (-2, -2, -2)
    constraints = [
        lambda x: x[0],      # x1 ≤ 0
        lambda x: x[1],      # x2 ≤ 0
        lambda x: x[2],      # x3 ≤ 0
        lambda x: -x[0] - 2, # x1 ≥ -2
        lambda x: -x[1] - 2, # x2 ≥ -2
        lambda x: -x[2] - 2  # x3 ≥ -2
    ]
    
    x0 = np.array([-1.0, -1.0, -1.0])
    
    print(f"\nФункция: f(x) = exp(x1) + exp(x2) + exp(x3)")
    print(f"Оптимум: стремится к (-∞, -∞, -∞)")
    print(f"Ограничения: -2 ≤ xi ≤ 0 (оптимум НА ГРАНИЦЕ x=-2)")
    
    # Scipy решение
    print("\n" + "-"*70)
    print("SCIPY SLSQP")
    print("-"*70)
    
    scipy_constraints = [
        NonlinearConstraint(lambda x: x[0], -np.inf, 0),
        NonlinearConstraint(lambda x: x[1], -np.inf, 0),
        NonlinearConstraint(lambda x: x[2], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[0] - 2, -np.inf, 0),
        NonlinearConstraint(lambda x: -x[1] - 2, -np.inf, 0),
        NonlinearConstraint(lambda x: -x[2] - 2, -np.inf, 0)
    ]
    
    result_scipy = minimize(exp_f, x0, method='SLSQP', jac=exp_grad,
                           constraints=scipy_constraints,
                           options={'ftol': 1e-10, 'maxiter': 1000})
    
    print(f"  x* = [{result_scipy.x[0]:.8f}, {result_scipy.x[1]:.8f}, {result_scipy.x[2]:.8f}]")
    print(f"  f(x*) = {result_scipy.fun:.10e}")
    
    # Метод барьерных функций
    print("\n" + "-"*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ")
    print("-"*70)
    
    result_barrier = barrier_method(
        exp_f, exp_grad, x0, constraints,
        r0=1.0, C=10.0, eps=1e-8, max_outer_iter=100,
        barrier_type='log'
    )
    
    f_error = abs(result_barrier['f_star'] - result_scipy.fun)
    print(f"  f(x*) = {result_barrier['f_star']:.10e}")
    print(f"  |f_barrier - f_scipy| = {f_error:.2e}")
    
    works = f_error < 1e-4
    
    print("\n" + "="*70)
    print(f"Метод барьерных функций: {'✅ РАБОТАЕТ' if works else '❌ НЕ РАБОТАЕТ'}")
    print("="*70)
    
    return works


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ МЕТОДА БАРЬЕРНЫХ ФУНКЦИЙ НА ДРУГИХ ФУНКЦИЯХ")
    print("="*70)
    
    # Тест 1: Квадратичная функция (оптимум внутри)
    works_quad_barrier, works_quad_lagrange = test_barrier_with_quadratic()
    
    # Тест 2: Экспоненциальная функция (оптимум на границе)
    works_exp = test_barrier_with_exp()
    
    # Итог
    print("\n" + "="*70)
    print("ОБЩИЙ ИТОГ")
    print("="*70)
    print(f"Квадратичная (внутри) + Барьер:  {'✅' if works_quad_barrier else '❌'}")
    print(f"Квадратичная (внутри) + Лагранж: {'✅' if works_quad_lagrange else '❌'}")
    print(f"Экспоненциальная (граница) + Барьер: {'✅' if works_exp else '❌'}")
    print("="*70)
    
    if works_quad_barrier:
        print("\n✅ МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ РАБОТАЕТ для квадратичных функций")
        print("   когда оптимум лежит ВНУТРИ допустимой области!")
    else:
        print("\n⚠️  Метод барьерных функций требует дополнительной настройки")
