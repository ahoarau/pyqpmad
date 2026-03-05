import pyqpmad
import numpy as np
import pytest

def test_unconstrained():
    solver = pyqpmad.Solver()
    H = np.array([[2.0, 0.0], [0.0, 2.0]], order='F')
    h = np.array([-2.0, -4.0])
    primal = np.zeros(2)
    
    # Minimize 1/2 x'Hx + h'x  =>  x'x + [-2, -4]'x
    # Gradient: 2x + [-2, -4]' = 0  => x = [1, 2]
    
    status = solver.solve(primal, H, h)
    
    assert status == pyqpmad.ReturnStatus.OK
    np.testing.assert_allclose(primal, [1.0, 2.0])

def test_constrained():
    solver = pyqpmad.Solver()
    # Minimize x^2 + y^2 subject to x + y = 1
    H = np.array([[2.0, 0.0], [0.0, 2.0]], order='F')
    h = np.array([0.0, 0.0])
    A = np.array([[1.0, 1.0]], order='F')
    Alb = np.array([1.0])
    Aub = np.array([1.0])
    primal = np.zeros(2)
    
    status = solver.solve(primal, H, h, A=A, Alb=Alb, Aub=Aub)
    
    assert status == pyqpmad.ReturnStatus.OK
    # Optimum is at x=0.5, y=0.5
    np.testing.assert_allclose(primal, [0.5, 0.5])

def test_bounds():
    solver = pyqpmad.Solver()
    # Minimize x^2 + y^2 subject to x >= 1, y >= 1
    H = np.array([[2.0, 0.0], [0.0, 2.0]], order='F')
    h = np.array([0.0, 0.0])
    lb = np.array([1.0, 1.0])
    ub = np.array([10.0, 10.0])
    primal = np.zeros(2)
    
    status = solver.solve(primal, H, h, lb=lb, ub=ub)
    
    assert status == pyqpmad.ReturnStatus.OK
    np.testing.assert_allclose(primal, [1.0, 1.0])

@pytest.mark.parametrize("size", [4, 10, 50, 100, 500])
def test_random_problems(size):
    np.random.seed(42)
    solver = pyqpmad.Solver()
    
    # Generate random positive definite Hessian
    Q, _ = np.linalg.qr(np.random.randn(size, size))
    D = np.diag(np.random.rand(size) + 0.1)
    H = (Q @ D @ Q.T)
    
    # Random linear part
    h = np.random.randn(size)
    
    # 1. Unconstrained
    primal = np.zeros(size)
    status = solver.solve(primal, H, h)
    assert status == pyqpmad.ReturnStatus.OK
    # Reference solution: x = -H^-1 h
    ref_sol = np.linalg.solve(H, -h)
    np.testing.assert_allclose(primal, ref_sol, atol=1e-8)
    
    # 2. Bounded
    # Add tight bounds around a random point
    target = np.random.randn(size)
    lb = target - 0.1
    ub = target + 0.1
    primal = np.zeros(size)
    status = solver.solve(primal, H, h, lb=lb, ub=ub)
    assert status == pyqpmad.ReturnStatus.OK
    # Check if bounds are satisfied
    assert np.all(primal >= lb - 1e-10)
    assert np.all(primal <= ub + 1e-10)

    # 3. Constrained
    # Add random linear constraints
    n_cons = size // 2 if size > 2 else 1
    A = np.random.randn(n_cons, size)
    # Make constraints feasible by picking a point and calculating bounds
    x_feat = np.random.randn(size)
    Ax_feat = A @ x_feat
    Alb = Ax_feat - 0.1
    Aub = Ax_feat + 0.1
    
    primal = np.zeros(size)
    status = solver.solve(primal, H, h, A=A, Alb=Alb, Aub=Aub)
    assert status == pyqpmad.ReturnStatus.OK
    # Check if constraints are satisfied
    Ax = A @ primal
    assert np.all(Ax >= Alb - 1e-10)
    assert np.all(Ax <= Aub + 1e-10)

if __name__ == "__main__":
    pytest.main([__file__])
