import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from conjugate_gradient import cgls_damped


# Set dimensions
input_dim = 1000
output_dim = 2000
hidden_dim = 1500

# Create a fixed random nonlinear function f(x)
torch.manual_seed(0)
A1 = torch.randn(hidden_dim, input_dim) / 10
b1 = torch.randn(hidden_dim) / 10
A2 = torch.randn(output_dim, hidden_dim) / 10
b2 = torch.randn(output_dim) / 10

def nonlinear_function(x):
    h = F.relu(A1 @ x + b1)
    y = torch.tanh(A2 @ h + b2)
    y = torch.sqrt(torch.abs(y))  # Ensure non-negative output
    return y

# Generate a target output (pretend it comes from some ground truth x)
x_true = torch.randn(input_dim)
y_target = nonlinear_function(x_true)

# Optimization: try to recover x from y_target
# x_est = torch.randn(input_dim, requires_grad=True)
x_est = x_true.clone().detach() + 0.05 * torch.randn(input_dim)  # Start close to the true x

class CurState:
    def __init__(self, x):
        self.x = x.requires_grad_(True)
        self.y = nonlinear_function(self.x)

    def matvec(self, v):
        with fwAD.dual_level():
            x_dual = fwAD.make_dual(self.x, v)
            y_dual = nonlinear_function(x_dual)
            y_primal, y_tangent = fwAD.unpack_dual(y_dual)
            return y_tangent

    def matvec_T(self, v):
        assert self.y is not None
        self.x.grad = None
        self.y.backward(v, retain_graph=True)
        return self.x.grad

    def dot(self, v1, v2):
        return torch.dot(v1, v2)

    def saxpy(self, a, x, y):
        return a * x + y

    def update(self, x_est, s):
        # Update x_est with the solution s
        x_est += s

print("before cg")
b = y_target - nonlinear_function(x_est)
loss = b.norm()
print(f"Step -1: loss = {loss.item():.6f}")
import code; code.interact(local=locals())

for step in range(1000): 
    # In each iteration, solve argmin \|Js - b\|, then update x_est = x_est + s
    cur_state = CurState(x_est)

    with torch.no_grad():
        b = y_target - nonlinear_function(x_est)
        s0 = torch.zeros_like(x_est)
        s = cgls_damped(
            matvec=cur_state.matvec,
            matvec_T=cur_state.matvec_T,
            dot=cur_state.dot,
            saxpy=cur_state.saxpy,
            b=b,
            x0=s0,
            damp=1e-5,
            tol=1e-10,
            atol=0.0,
            max_iter=100,
            verbose=True,

        )
        cur_state.update(x_est, s)

        b = y_target - nonlinear_function(x_est)
        loss = b.norm()

        print(f"Step {step}: loss = {loss.item():.6f}")
        import code; code.interact(local=locals())



