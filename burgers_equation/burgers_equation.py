from typing_extensions import Self

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn

from progress.bar import Bar


class BurgersNN(nn.Module):
    def __init__(self, nu_init=0.1):
        super(BurgersNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Learnable viscosity parameter
        self.log_nu = nn.Parameter(torch.tensor([np.log(nu_init)]))
        
        # Store training history for visualization
        self.history = {
            'loss': [],
            'physics_loss': [],
            'data_loss': [],
            'nu': []
        }

    def forward(self, x, t):
        inp = torch.cat((x, t), dim=1)
        return self.net(inp)
    
    @property
    def nu(self):
        return torch.exp(self.log_nu) # Ensure positivity of viscosity
    
    def pde_residual(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        residual = u_t + u * u_x - self.nu * u_xx
        return residual
    
    def total_loss(self, data):
        u_pred_ic = self.forward(data['x_ic'], data['t_ic'])
        loss_ic = torch.mean((u_pred_ic - data['u_ic'])**2)

        u_pred_bc = self.forward(data['x_bc'], data['t_bc'])
        loss_bc = torch.mean((u_pred_bc - data['u_bc'])**2)

        res = self.pde_residual(data['x_col'], data['t_col'])
        loss_pde = torch.mean(res**2)

        u_pred_obs = self.forward(data['x_obs'], data['t_obs'])
        loss_obs = torch.mean((u_pred_obs - data['u_obs'])**2)

        # Don't log here — let the training loop handle it
        return {
            'total': loss_ic + loss_bc + loss_pde + loss_obs,
            'pde':   loss_pde,
            'data':  loss_ic + loss_bc + loss_obs
        }
    
    def train_model(self, data, num_epochs=10000, learning_rate=0.001):
        # Phase 1: Train with Adam optimizer
        optimizer_adam = torch.optim.Adam(self.parameters(), lr=learning_rate)
        with Bar('Training', max=num_epochs) as bar:
            for epoch in range(num_epochs):
                optimizer_adam.zero_grad()
                losses = self.total_loss(data)
                losses['total'].backward()
                optimizer_adam.step()

                if epoch % 10 == 0:
                    self.history['loss'].append(losses['total'].item())
                    self.history['physics_loss'].append(losses['pde'].item())
                    self.history['data_loss'].append(losses['data'].item())
                    self.history['nu'].append(self.nu.item())
                bar.next()
                
        # Phase 2: Fine-tune with LBFGS optimizer
        optimizer_lbfgs = torch.optim.LBFGS(self.parameters(), max_iter=50000, tolerance_grad=1e-9, line_search_fn='strong_wolfe')
        step_counter = [0]
        def closure():
            optimizer_lbfgs.zero_grad()
            losses = self.total_loss(data)
            losses['total'].backward()
            closure.last_losses = {k: v.item() for k, v in losses.items()}
            return losses['total']

        with Bar('Fine-tuning', max=500) as bar:
            for _ in range(500):
                optimizer_lbfgs.step(closure)
                step_counter[0] += 1
                if step_counter[0] % 10 == 0:
                    self.history['loss'].append(closure.last_losses['total'])
                    self.history['physics_loss'].append(closure.last_losses['pde'])
                    self.history['data_loss'].append(closure.last_losses['data'])
                    self.history['nu'].append(self.nu.item())
                bar.next()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# ==== 1. Data Generation ====
data = sp.io.loadmat('burgers_shock.mat')
# Extract data
t = data['t'].flatten()  # Time points
x = data['x'].flatten()  # Spatial points
usol = data['usol']  # Solution matrix (time x space)
# Initial condition (t=0)
x_ic = x
t_ic = np.zeros_like(x_ic)
u_ic = usol[:, 0]  # Initial velocity profile
# Boundary conditions (x=0 and x=1)
x_bc = np.concatenate([np.full_like(t, -1), np.full_like(t, 1)])
t_bc = np.concatenate([t, t])
u_bc = np.zeros(len(x_bc))  # u(0, t) = u(1, t) = 0
# Collocation points for physics loss
N_col = 10000
x_col = np.random.uniform(-1, 1, N_col)
t_col = np.random.uniform(0, 1, N_col)
# Inverse problem points
N_obs = 100
idx_x = np.random.choice(len(x), N_obs)
idx_t = np.random.choice(len(t), N_obs)
x_obs = x[idx_x]
t_obs = t[idx_t]
u_obs = usol[idx_x, idx_t] + 0.01 * np.random.randn(N_obs)  # Add noise
# Convert to PyTorch tensors
data_torch = {
    'x_ic': torch.tensor(x_ic, dtype=torch.float32).unsqueeze(1).to(device),
    't_ic': torch.tensor(t_ic, dtype=torch.float32).unsqueeze(1).to(device),
    'u_ic': torch.tensor(u_ic, dtype=torch.float32).unsqueeze(1).to(device),
    'x_bc': torch.tensor(x_bc, dtype=torch.float32).unsqueeze(1).to(device),
    't_bc': torch.tensor(t_bc, dtype=torch.float32).unsqueeze(1).to(device),
    'u_bc': torch.tensor(u_bc, dtype=torch.float32).unsqueeze(1).to(device),
    'x_col': torch.tensor(x_col, dtype=torch.float32).unsqueeze(1).to(device),
    't_col': torch.tensor(t_col, dtype=torch.float32).unsqueeze(1).to(device),
    'x_obs': torch.tensor(x_obs, dtype=torch.float32).unsqueeze(1).to(device),
    't_obs': torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1).to(device),
    'u_obs': torch.tensor(u_obs, dtype=torch.float32).unsqueeze(1).to(device)
}

# ==== 2. Model Training ====
model = BurgersNN(nu_init=0.1).to(device)
model.to(device)
model.train_model(data_torch, num_epochs=10000, learning_rate=0.001)

# ==== 3. Visualization ====
# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(model.history['loss'], color='red', label='Total Loss')
ax1.plot(model.history['physics_loss'], color='blue', label='Physics Loss')
ax1.plot(model.history['data_loss'], color='green', label='Data Loss')
ax1.legend()
ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid()
ax1.set_xlim(0, len(model.history['loss']))
ax2.plot(model.history['nu'], color='blue')
ax2.set_title('Learned Viscosity Nu')
ax2.set_xlabel('Epochs')
ax2.set_ylabel(r'$\nu$')
ax2.grid()
ax2.axhline(y=0.01/np.pi, color='black', linestyle='dashed', label=f'True $\\nu$ = {0.01/np.pi:.5f}')
ax2.legend()
ax2.set_xlim(0, len(model.history['nu']))
fig.savefig('training_history.png')
plt.tight_layout()


# Create an animation of the predicted and true solutions over time
XX, TT = np.meshgrid(x, t)
x_grid = torch.tensor(XX.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
t_grid = torch.tensor(TT.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    u_pred_grid = model(x_grid, t_grid).cpu().numpy().reshape(XX.shape)
    
fig, ax = plt.subplots()
ax.set_xlabel('x')
true_line = ax.plot(x, usol[:, 0], label='True Solution', color='black', linestyle='dashed')[0]
pred_line = ax.plot(x, u_pred_grid[0, :], label='Predicted Solution', color='red', linestyle='solid')[0]
ax.set_title('Burgers\' Equation Solution Over Time')
ax.legend()
ax.grid()
def update(frame):
    true_line.set_data(x, usol[:, frame])
    pred_line.set_data(x, u_pred_grid[frame, :])
    return true_line, pred_line
ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000/30)
ani.save('burgers_solution_animation.gif', writer='pillow', fps=30)
plt.show()