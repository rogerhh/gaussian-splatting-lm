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

def cgls_damped(
    matvec,         # A @ x
    matvec_T,       # A.T @ y
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
    damp_sq = damp * damp
    x = x0
    iter_total = 0

    while iter_total < max_iter:
        print(f"Restarting CG iteration {iter_total + 1}...") if verbose else None

        r0 = saxpy(-1.0, matvec(x), b)         # r0 = b - A x0
        s0 = saxpy(-damp_sq, x, matvec_T(r0))
        p0 = s0

        r = r0
        s = s0
        p = p0
        gamma = dot(s, s)  # Initial norm of s

        for k in range(restart_iter):
            iter_total += 1
            q = matvec(p)                  # q = A p
            delta = dot(q, q) + damp_sq * dot(p, p)  # delta = <q, q> + λ^2 * <p, p>
            if delta == 0:
                print("Early termination: delta is zero.")
                break
            # print(f"delta: {delta:.4e}, gamma: {gamma:.4e}")
            s = saxpy(-damp_sq, x, matvec_T(r))  # s = A^T r - λ^2 x
            alpha = gamma / delta
            x = saxpy(alpha, p, x)         # x = x + alpha * p
            r = saxpy(-alpha, q, r)        # r = r - alpha * q
            s = saxpy(-damp_sq, x, matvec_T(r))  # s = A^T r - λ^2 x
            gamma_prev = gamma
            gamma = dot(s, s)  # Update norm of s
            beta = gamma / gamma_prev
            p = saxpy(beta, p, s)          # p = s + beta * p

            if verbose:
                cur_r = saxpy(-1.0, matvec(x), b)         # r0 = b - A x0
                res = dot(cur_r, cur_r) + damp_sq * dot(x, x)  # Compute residual norm
                print(f"[Iter {iter_total+1}] res: {res:.2e}")

            if gamma < max(tol * (gamma_prev ** 0.5), atol):
                if verbose:
                    print(f"Convergence achieved at iteration {iter_total+1}.")
                break

    return x

