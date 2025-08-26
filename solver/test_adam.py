import torch
import torch.nn.functional as F

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
    # x: [input_dim]
    h = F.relu(A1 @ x + b1)
    y = torch.tanh(A2 @ h + b2)
    y = torch.sqrt(torch.abs(y))  
    return y

# Generate a target output (pretend it comes from some ground truth x)
x_true = torch.randn(input_dim)
y_target = nonlinear_function(x_true)

# Optimization: try to recover x from y_target
x_est = x_true.clone().detach() + 0.05 * torch.randn(input_dim)  # Start close to the true x
x_est = x_est.requires_grad_(True)
optimizer = torch.optim.Adam([x_est], lr=1e-4)

for step in range(500):
    optimizer.zero_grad()
    y_pred = nonlinear_function(x_est)
    b = y_target - y_pred  # residual
    loss = b.norm()
    loss.backward()
    optimizer.step()

    print(f"Step {step}: loss = {loss.item():.6f}")

# Compare results
with torch.no_grad():
    b = y_target - nonlinear_function(x_est)
    loss = b.norm()
    print(f"\nRecovered x error norm: {loss:.6f}")

