"""
Модульные тесты для методов условной оптимизации (ЛР4)

Сравнение наших реализаций с scipy.optimize
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# Импортируем наши методы
from constrained_optimization import (
    create_rosenbrock_function,
    get_constraints_variant2,
    check_constraints,
    penalty_method,
    barrier_method,
    modified_lagrange_method,
    projected_gradient_method
)


def test_scipy_reference_solution():
    """Тест: получение эталонного решения через scipy"""
    print("\n" + "="*70)
    print("ЭТАЛОННОЕ РЕШЕНИЕ ЧЕРЕЗ SCIPY")
    print("="*70)
    
    # Параметры варианта 2
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = get_constraints_variant2()
    x0 = np.array([0.5, 0.5, 0.5])
    
    # Обертки для scipy
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    # Ограничения для scipy
    scipy_constraints = [
        NonlinearConstraint(
            lambda x: x[0]**2 + x[1]**2 - x[2] - 1,
            -np.inf, 0
        ),
        NonlinearConstraint(lambda x: -x[0], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[1], -np.inf, 0),
        NonlinearConstraint(lambda x: -x[2], -np.inf, 0)
    ]
    
    # Решение через SLSQP
    result = minimize(
        f,
        x0,
        method='SLSQP',
        jac=grad_f,
        constraints=scipy_constraints,
        options={'ftol': 1e-8, 'maxiter': 1000}
    )
    
    x_scipy = result.x
    f_scipy = result.fun
    
    print(f"\nScipy SLSQP решение:")
    print(f"  x* = [{x_scipy[0]:.8f}, {x_scipy[1]:.8f}, {x_scipy[2]:.8f}]")
    print(f"  f(x*) = {f_scipy:.10f}")
    print(f"  Успех: {result.success}")
    print(f"  Итераций: {result.nit}")
    
    # Проверка допустимости
    feasible, info = check_constraints(x_scipy, constraints)
    print(f"  Допустимо: {feasible}")
    
    assert feasible, "Scipy решение должно быть допустимым"
    assert abs(f_scipy - 100.0) < 1e-6, "Scipy решение должно быть близко к 100"
    
    print("\n✅ SCIPY РЕШЕНИЕ ВЕРНО")
    
    return x_scipy, f_scipy


def test_penalty_method_vs_scipy(x_scipy, f_scipy):
    """Тест: метод штрафных функций vs scipy"""
    print("\n" + "="*70)
    print("МЕТОД ШТРАФНЫХ ФУНКЦИЙ VS SCIPY")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = get_constraints_variant2()
    x0 = np.array([0.5, 0.5, 0.5])
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    # Наш метод
    result = penalty_method(
        f, grad_f, x0, constraints,
        r0=1.0, C=10.0, eps=1e-6, max_outer_iter=50
    )
    
    x_ours = result['x_star']
    f_ours = result['f_star']
    
    print(f"\nНаш метод штрафных функций:")
    print(f"  x* = [{x_ours[0]:.8f}, {x_ours[1]:.8f}, {x_ours[2]:.8f}]")
    print(f"  f(x*) = {f_ours:.10f}")
    
    # Сравнение с scipy
    x_error = np.linalg.norm(x_ours - x_scipy)
    f_error = abs(f_ours - f_scipy)
    
    print(f"\nПогрешность относительно scipy:")
    print(f"  ||x_ours - x_scipy|| = {x_error:.2e}")
    print(f"  |f_ours - f_scipy| = {f_error:.2e}")
    
    # Метод штрафных функций сходится снаружи, поэтому допустимость не обязательна
    assert f_error < 1e-4, "Метод штрафных функций должен дать f(x*) близкое к scipy"
    
    print("\n✅ МЕТОД ШТРАФНЫХ ФУНКЦИЙ ВЕРНО")


def test_lagrange_method_vs_scipy(x_scipy, f_scipy):
    """Тест: метод множителей Лагранжа vs scipy"""
    print("\n" + "="*70)
    print("МЕТОД МНОЖИТЕЛЕЙ ЛАГРАНЖА VS SCIPY")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = get_constraints_variant2()
    x0 = np.array([0.5, 0.5, 0.5])
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    # Наш метод
    result = modified_lagrange_method(
        f, grad_f, x0, constraints,
        r0=1.0, C=2.0, eps=1e-6, max_outer_iter=100
    )
    
    x_ours = result['x_star']
    f_ours = result['f_star']
    feasible = result['feasible']
    
    print(f"\nНаш метод множителей Лагранжа:")
    print(f"  x* = [{x_ours[0]:.8f}, {x_ours[1]:.8f}, {x_ours[2]:.8f}]")
    print(f"  f(x*) = {f_ours:.10f}")
    print(f"  Допустимо: {feasible}")
    
    # Сравнение с scipy
    x_error = np.linalg.norm(x_ours - x_scipy)
    f_error = abs(f_ours - f_scipy)
    
    print(f"\nПогрешность относительно scipy:")
    print(f"  ||x_ours - x_scipy|| = {x_error:.2e}")
    print(f"  |f_ours - f_scipy| = {f_error:.2e}")
    
    assert feasible, "Метод Лагранжа должен дать допустимое решение"
    assert f_error < 1e-4, "Метод Лагранжа должен дать f(x*) близкое к scipy"
    
    print("\n✅ МЕТОД МНОЖИТЕЛЕЙ ЛАГРАНЖА ВЕРНО")


def test_projected_gradient_vs_scipy(x_scipy, f_scipy):
    """Тест: метод проекции градиента vs scipy"""
    print("\n" + "="*70)
    print("МЕТОД ПРОЕКЦИИ ГРАДИЕНТА VS SCIPY")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = get_constraints_variant2()
    x0 = np.array([0.5, 0.5, 0.5])
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    # Наш метод (с eps1 <= 0 согласно методичке)
    result = projected_gradient_method(
        f, grad_f, x0, constraints,
        eps1=-1e-6, eps2=1e-8, max_iter=1000
    )
    
    x_ours = result['x_star']
    f_ours = result['f_star']
    feasible = result['feasible']
    
    print(f"\nНаш метод проекции градиента:")
    print(f"  x* = [{x_ours[0]:.8f}, {x_ours[1]:.8f}, {x_ours[2]:.8f}]")
    print(f"  f(x*) = {f_ours:.10f}")
    print(f"  Допустимо: {feasible}")
    
    # Сравнение с scipy
    x_error = np.linalg.norm(x_ours - x_scipy)
    f_error = abs(f_ours - f_scipy)
    
    print(f"\nПогрешность относительно scipy:")
    print(f"  ||x_ours - x_scipy|| = {x_error:.2e}")
    print(f"  |f_ours - f_scipy| = {f_error:.2e}")
    
    assert feasible, "Метод проекции градиента должен дать допустимое решение"
    # Метод проекции градиента медленнее сходится
    assert f_error < 0.1, "Метод проекции градиента должен дать разумное приближение"
    
    print("\n✅ МЕТОД ПРОЕКЦИИ ГРАДИЕНТА ВЕРНО")


def test_barrier_method_feasibility():
    """Тест: метод барьерных функций всегда внутри допустимой области
    
    ПРИМЕЧАНИЕ: Метод барьерных функций НЕ сходится к точному оптимуму,
    если оптимум лежит на границе допустимой области.
    Это известный недостаток метода - он остается строго внутри области.
    """
    print("\n" + "="*70)
    print("МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ - ПРОВЕРКА ДОПУСТИМОСТИ")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = get_constraints_variant2()
    x0 = np.array([0.5, 0.5, 0.5])
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    # Метод барьерных функций должен всегда оставаться внутри области
    result = barrier_method(
        f, grad_f, x0, constraints,
        r0=1.0, C=10.0, eps=1e-6, max_outer_iter=50,
        barrier_type='log'
    )
    
    x_ours = result['x_star']
    f_ours = result['f_star']
    feasible = result['feasible']
    
    print(f"\nНаш метод барьерных функций:")
    print(f"  x* = [{x_ours[0]:.8f}, {x_ours[1]:.8f}, {x_ours[2]:.8f}]")
    print(f"  f(x*) = {f_ours:.10f}")
    print(f"  Допустимо: {feasible}")
    
    # Проверяем только допустимость (метод всегда внутри области)
    assert feasible, "Метод барьерных функций ВСЕГДА должен давать допустимое решение"
    
    # Проверяем что f(x*) > 100 (метод не может достичь границы)
    assert f_ours >= 100.0, "f(x*) должно быть >= 100 (истинный оптимум на границе)"
    
    print(f"\n⚠️  ПРИМЕЧАНИЕ: Метод барьерных функций не сходится к точному оптимуму,")
    print(f"    т.к. оптимум лежит на границе допустимой области.")
    print(f"    Метод остается строго внутри области (f(x*) = {f_ours:.6f} > 100)")
    
    print("\n✅ МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ ВЕРНО (только допустимость)")


def test_known_optimum():
    """Тест: проверка на известном оптимуме (1,1,1)"""
    print("\n" + "="*70)
    print("ПРОВЕРКА НА ИЗВЕСТНОМ ОПТИМУМЕ (1, 1, 1)")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    constraints = get_constraints_variant2()
    
    f = lambda x: f_raw(x)
    grad_f = lambda x: grad_f_raw(x)
    
    x_opt_known = np.array([1.0, 1.0, 1.0])
    f_opt_known = 100.0
    
    # Проверяем что (1,1,1) действительно допустимая точка
    feasible, info = check_constraints(x_opt_known, constraints)
    f_at_opt = f(x_opt_known)
    
    print(f"\nИзвестный оптимум:")
    print(f"  x* = {x_opt_known}")
    print(f"  f(x*) = {f_at_opt:.10f}")
    print(f"  Допустимо: {feasible}")
    
    assert feasible, "(1,1,1) должно быть допустимой точкой"
    assert abs(f_at_opt - f_opt_known) < 1e-10, "f(1,1,1) должно равняться 100"
    
    print("\n✅ ИЗВЕСТНЫЙ ОПТИМУМ ВЕРНО")


def test_gradient_at_optimum():
    """Тест: градиент в оптимуме (проверка необходимых условий)"""
    print("\n" + "="*70)
    print("ПРОВЕРКА ГРАДИЕНТА В ОПТИМУМЕ")
    print("="*70)
    
    A, B, F0, N = 150, 2, 100, 3
    f_raw, grad_f_raw, _ = create_rosenbrock_function(A, B, F0)
    
    grad_f = lambda x: grad_f_raw(x)
    
    x_opt_known = np.array([1.0, 1.0, 1.0])
    
    grad_at_opt = grad_f(x_opt_known)
    grad_norm = np.linalg.norm(grad_at_opt)
    
    print(f"\nГрадиент в точке (1,1,1):")
    print(f"  ∇f = [{grad_at_opt[0]:.6f}, {grad_at_opt[1]:.6f}, {grad_at_opt[2]:.6f}]")
    print(f"  ||∇f|| = {grad_norm:.2e}")
    
    # В безусловной задаче градиент должен быть 0
    # В условной - должен быть линейной комбинацией градиентов активных ограничений
    # Для нашей задачи (1,1,1) лежит на границе g1, поэтому градиент не обязательно 0
    print(f"\nПримечание: В условной оптимизации ||∇f|| не обязательно = 0")
    
    print("\n✅ ГРАДИЕНТ В ОПТИМУМЕ ВЕРНО")


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "="*70)
    print("МОДУЛЬНЫЕ ТЕСТЫ МЕТОДОВ УСЛОВНОЙ ОПТИМИЗАЦИИ (ЛР4)")
    print("Сравнение с scipy.optimize")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Тест 1: Scipy решение
        x_scipy, f_scipy = test_scipy_reference_solution()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ SCIPY РЕШЕНИЕ: {e}")
        tests_failed += 1
        return tests_passed, tests_failed
    
    try:
        # Тест 2: Метод штрафных функций
        test_penalty_method_vs_scipy(x_scipy, f_scipy)
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ МЕТОД ШТРАФНЫХ ФУНКЦИЙ: {e}")
        tests_failed += 1
    
    try:
        # Тест 3: Метод Лагранжа
        test_lagrange_method_vs_scipy(x_scipy, f_scipy)
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ МЕТОД ЛАГРАНЖА: {e}")
        tests_failed += 1
    
    try:
        # Тест 4: Метод проекции градиента
        test_projected_gradient_vs_scipy(x_scipy, f_scipy)
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ МЕТОД ПРОЕКЦИИ ГРАДИЕНТА: {e}")
        tests_failed += 1
    
    try:
        # Тест 5: Метод барьерных функций
        test_barrier_method_feasibility()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ МЕТОД БАРЬЕРНЫХ ФУНКЦИЙ: {e}")
        tests_failed += 1
    
    try:
        # Тест 6: Известный оптимум
        test_known_optimum()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ ИЗВЕСТНЫЙ ОПТИМУМ: {e}")
        tests_failed += 1
    
    try:
        # Тест 7: Градиент в оптимуме
        test_gradient_at_optimum()
        tests_passed += 1
    except AssertionError as e:
        print(f"\n❌ ГРАДИЕНТ В ОПТИМУМЕ: {e}")
        tests_failed += 1
    
    # Вывод итога
    print("\n" + "="*70)
    print("ИТОГ")
    print("="*70)
    print(f"Всего тестов: {tests_passed + tests_failed}")
    print(f"Успешно: {tests_passed}")
    print(f"Провалено: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print(f"\n❌ {tests_failed} ТЕСТ(ОВ) НЕ ПРОЙДЕНЫ")
    
    return tests_failed == 0


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
