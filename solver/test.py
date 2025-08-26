import numpy as np

def cgls(A, b, x0=None, max_iter=100, tol=1e-6, verbose=False):
    """
    Conjugate Gradient Least Squares (CGLS)
    Solves min_x ||Ax - b||^2
    
    Parameters:
        A: m x n matrix (NumPy array or scipy.sparse matrix)
        b: m vector (right-hand side)
        x0: optional initial guess (default zero)
        max_iter: maximum number of iterations
        tol: relative residual tolerance for convergence
        verbose: print progress if True
    
    Returns:
        x: n-vector solution
        residuals: list of residual norms
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = b - A @ x
    s = A.T @ r
    p = s.copy()
    norm_s_old = np.dot(s, s)

    residuals = [np.linalg.norm(r)]

    for k in range(max_iter):
        q = A @ p
        norm_q2 = np.dot(q, q)
        if norm_q2 == 0:
            break

        alpha = norm_s_old / norm_q2
        x += alpha * p
        r -= alpha * q
        s = A.T @ r
        norm_s_new = np.dot(s, s)

        if verbose:
            print(f"Iter {k+1:2d}: residual = {np.linalg.norm(r):.3e}, alpha = {alpha:.3e}")

        if np.sqrt(norm_s_new) < tol:
            break

        beta = norm_s_new / norm_s_old
        p = s + beta * p
        norm_s_old = norm_s_new
        residuals.append(np.linalg.norm(r))
    
    return x, residuals


A = np.random.randn(20, 10)
b = np.random.randn(20)

x0 = np.random.randn(10)

x, residuals = cgls(A, b, x0=x0, max_iter=100, tol=1e-6, verbose=True)
