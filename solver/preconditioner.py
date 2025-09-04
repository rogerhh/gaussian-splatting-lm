import torch

class Preconditioner:
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, vector):
        # Placeholder for preconditioning logic
        return vector  # No actual preconditioning applied

class AdaHessianPreconditioner:
    def __init__(self, z_gen_func, beta1=0.9, beta2=0.999, eps=1e-8, hessian_power=1.0):
        self.iteration = 0
        self.z_gen_func = z_gen_func
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.D_sq = 0 

    def reset(self):
        self.iteration = 0
        self.D_sq = 0

    def update(self, Hz_func, max_iter):
        self.iteration += 1
        D_accum = 0
        for _ in range(max_iter):
            z = self.z_gen_func()
            Di = z * Hz_func(z)
            D_accum = Di * Di + D_accum
            del Di, z

        D_accum = D_accum / max_iter
        # D_accum.block_average_and_expand()
        self.D_sq = ((1 - self.beta2) / max_iter) * D_accum + self.beta2 * self.D_sq

    def __call__(self, v):
        D_corrected = self.D_sq / (1 - self.beta2 ** self.iteration)
        return v / (D_corrected.sqrt() + self.eps)
