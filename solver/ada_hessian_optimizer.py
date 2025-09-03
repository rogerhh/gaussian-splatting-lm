import torch

class AdaHessianOptimizer:
    def __init__(self, z_gen_func, beta1=0.9, beta2=0.999, eps=1e-8, hessian_power=1.0):
        self.z_gen_func = z_gen_func
        self.iteration = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = 0
        self.D_sq = 0 

    def reset(self):
        self.iteration = 0
        self.m = 0
        self.D_sq = 0

    def get_update_step(self, Hz_and_g_func, max_iter):
        self.iteration += 1

        g_accum = 0
        D_accum = 0
        for _ in range(max_iter):
            z = self.z_gen_func()
            Hz, g = Hz_and_g_func(z)
            g_accum = g + g_accum
            Di = z * Hz
            D_accum = Di * Di + D_accum
            del Hz, g, Di, z
            torch.cuda.empty_cache()

        g_accum = g_accum / max_iter
        D_accum = D_accum / max_iter

        D_accum.block_average_and_expand()

        self.D_sq = ((1 - self.beta2) / max_iter) * D_accum + self.beta2 * self.D_sq
        self.m = (1 - self.beta1) * g_accum + self.beta1 * self.m

        D_corrected = self.D_sq / (1 - self.beta2 ** self.iteration)
        m_corrected = self.m / (1 - self.beta1 ** self.iteration)

        step = m_corrected / (D_corrected.sqrt() + self.eps)

        return step
