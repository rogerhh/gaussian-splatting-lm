import math
from utils.general_utils import safe_interact

def conjugate_gradient(
    matvec,        # function(x) -> A @ x
    vec_add,       # function(x, y) -> x + y
    vec_sub,       # function(x, y) -> x - y
    dot,           # function(x, y) -> dot product
    scalar_mul,    # function(a, x) -> a * x
    b,             # right-hand side vector
    x0,            # initial guess
    tol=1e-10,     # relative tolerance on residual norm
    atol=0.0,      # absolute tolerance on residual norm
    max_iter=1000, # max number of iterations
    callback=None, # optional: function(x, r, iter)
    verbose=False  # if True, print residual norm
):
    x = x0
    r = vec_sub(b, matvec(x))          # r0 = b - A x0
    p = r
    rs_old = dot(r, r)

    norm_r0 = rs_old ** 0.5
    if norm_r0 < atol:
        return x

    for k in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / dot(p, Ap)

        x = vec_add(x, scalar_mul(alpha, p))     # x_{k+1} = x_k + alpha * p_k
        r = vec_sub(r, scalar_mul(alpha, Ap))    # r_{k+1} = r_k - alpha * A p_k

        rs_new = dot(r, r)
        norm_r = rs_new ** 0.5

        if verbose:
            print(f"[Iter {k+1}] Residual norm: {norm_r:.2e}")

        if callback:
            callback(x, r, k+1)

        if norm_r < max(tol * norm_r0, atol):
            break

        beta = rs_new / rs_old
        p = vec_add(r, scalar_mul(beta, p))       # p_{k+1} = r_{k+1} + beta * p_k
        rs_old = rs_new

    return x

def cg_damped(
    Ax,             # A @ x, here A is assumed to be SPD
    dot,            # dot(x, y)
    saxpy,          # ax + y
    b,              # right-hand side
    x0,             # initial guess
    M=None,         # preconditioner M ~ A^-1
    tol=1e-10,
    atol=0.0,
    max_iter=1000,
    restart_iter=5, # restart every `restart_iter` iterations
    callback=None,
    verbose=False
):
    x = x0
    iter_total = 0

    break_flag = False

    while iter_total < max_iter:
        print(f"Restarting CG iteration {iter_total + 1}...") if verbose else None

        r = saxpy(-1.0, Ax(x), b)         # r = b - A x
        res = dot(r, r)
        z = M(r) if M is not None else r  # z = M r
        p = z

        print(f"[Iter {iter_total}] res: {res:.2e}")

        for k in range(restart_iter):
            gamma = dot(r, z)                       # gamma = <r, z>
            q = Ax(p)                               # q = A p
            delta = dot(p, q)                       # delta = <q, p>
            if delta < 1e-15:
                print("Early termination: delta is too small.")
                break_flag = True
                break
            alpha = gamma / delta
            x = saxpy(alpha, p, x)                  # x = x + alpha * p
            r = saxpy(-alpha, q, r)                 # r = r - alpha * q
            res = dot(r, r)
            z = M(r) if M is not None else r        # z = M r
            gamma_prev = gamma
            gamma = dot(r, z)                       # Update gamma
            beta = gamma / gamma_prev
            p = saxpy(beta, p, z)                   # p = z + beta * p

            # if verbose:
            x_norm = math.sqrt(dot(x, x))
            print(f"[Iter {iter_total+1}] res: {res:.2e}, |x|: {x_norm:.2e}")
            # safe_interact(local=locals(), banner="Debugging CG")

            iter_total += 1
            if iter_total >= max_iter:
                break_flag = True
                break

        if break_flag:
            break

    r = saxpy(-1.0, Ax(x), b)         # r = b - A x
    res = dot(r, r)
    print(f"Final residual norm: {res:.2e}")

    return x


def cgls_damped(
    Ax,         # A @ x
    Atx,       # A.T @ y
    dot,            # dot(x, y)
    saxpy,          # ax + y
    b,              # right-hand side
    x0,             # initial guess
    damp=0.0,       # damping factor
    tol=1e-10,
    atol=0.0,
    max_iter=1000,
    restart_iter=5, # restart every `restart_iter` iterations
    callback=None,
    verbose=False
):
    x = x0
    iter_total = 0

    last_res = math.inf

    break_flag = False

    while iter_total < max_iter:
        print(f"Restarting CG iteration {iter_total + 1}...") if verbose else None

        r0 = saxpy(-1.0, Ax(x), b)         # r0 = b - A x0
        s0 = saxpy(-damp, x, Atx(r0))  # s0 = A^T r0 - λ^2 x0
        p0 = s0

        r = r0
        s = s0
        p = p0
        gamma = dot(s, s)  # Initial norm of s

        for k in range(restart_iter):
            q = Ax(p)                  # q = A p
            delta = dot(q, q) + dot(p, p, damp)  # delta = <q, q> + λ^2 * <p, p>
            if delta < 1e-20:
                print("Early termination: delta is too small.")
                break_flag = True
                break
            # print(f"delta: {delta:.4e}, gamma: {gamma:.4e}")
            alpha = gamma / delta
            x = saxpy(alpha, p, x)         # x = x + alpha * p
            r = saxpy(-alpha, q, r)        # r = r - alpha * q
            s = saxpy(-damp, x, Atx(r))  # s = A^T r - λ^2 x
            gamma_prev = gamma
            gamma = dot(s, s)  # Update norm of s
            beta = gamma / gamma_prev
            p = saxpy(beta, p, s)          # p = s + beta * p

            # if verbose:
            cur_r = saxpy(-1.0, Ax(x), b)         # r0 = b - A x0
            res = dot(cur_r, cur_r) + dot(x, x, damp)  # Compute residual norm
            print(f"[Iter {iter_total+1}] res: {res:.2e}")
            if res > last_res:
                print("Warning: Residual norm increased!")
                break_flag = True
                break

            last_res = res

            if gamma < max(tol * (gamma_prev ** 0.5), atol):
                if verbose:
                    print(f"Convergence achieved at iteration {iter_total+1}.")
                break_flag = True
                break

            iter_total += 1
            if iter_total >= max_iter:
                break_flag = True
                break

        if break_flag:
            break

    return x

