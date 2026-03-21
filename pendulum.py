from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn as nn
import torch
from progress.bar import Bar

class PendulumMLP(nn.Module):
    def __init__(self):
        super(PendulumMLP, self).__init__()
        self.input_layer = nn.Linear(1, 32)  # Input is time, output is angle
        self.hidden_layer_1 = nn.Linear(32, 32)
        self.hidden_layer_2 = nn.Linear(32, 32)
        self.hidden_layer_3 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)  # Output is the angle
        self.activation = nn.SiLU()  # Using SiLU activation function
    
    def forward(self, x):
        h1 = self.activation(self.input_layer(x))
        h2 = self.activation(self.hidden_layer_1(h1))
        h3 = self.activation(self.hidden_layer_2(h2))
        h4 = self.activation(self.hidden_layer_3(h3))
        output = self.output_layer(h4)
        return output
    
def train_model(model, t, true_angles, t_physics, t_eval, num_epochs=1000, learning_rate=0.001, lambda_physical = 0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loss_history = np.zeros(num_epochs)
    
    with Bar('Training', max=num_epochs) as bar:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Compute the MSE loss
            predicted_angles = model(t)
            mse_loss = loss_fn(predicted_angles, true_angles)
            
            # Compute the physical loss using automatic differentiation
            predicted_physics = model(t_physics)
            dtheta_dt = torch.autograd.grad(
                outputs=predicted_physics, 
                inputs=t_physics,
                grad_outputs=torch.ones_like(predicted_physics),
                create_graph=True,
                retain_graph=True
            )[0]
            domega_dt = torch.autograd.grad(
                outputs=dtheta_dt, 
                inputs=t_physics,
                grad_outputs=torch.ones_like(dtheta_dt),
                create_graph=True,
                retain_graph=True
            )[0]
            # Calculate the physical loss based on the pendulum equation
            physical_residual = domega_dt + (9.81 / 1.0) * torch.sin(predicted_physics)
            physical_loss = torch.mean(physical_residual ** 2)
            # Total loss is a combination of MSE loss and physical loss
            total_loss = mse_loss + lambda_physical * physical_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Print the loss every 1000 epochs
            # if epoch % 1000 == 0:
            #     print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Physical Loss: {physical_loss.item():.4f}")
            loss_history[epoch] = total_loss.item()
            
            # Eval and save the predicted angles for plotting every 200 epochs
            if epoch % 200 == 0:
                model.eval()
                with torch.no_grad():
                    predicted_angles_eval = model(t_eval).squeeze().numpy()
                    path = f"saves_without_physical_loss/pendulum_epoch_{epoch}.npy"
                    np.save(path, predicted_angles_eval)
                    # print(f"Saved predicted angles at epoch {epoch} to {path}")
            
            if t_physics.grad is not None:
                t_physics.grad.zero_()  # Clear the gradients for the next iteration
                
            bar.next()

    model.eval()
    with torch.no_grad():
        predicted_angles_eval = model(t_eval).squeeze().numpy()
        np.save(f"saves_without_physical_loss/pendulum_epoch_{num_epochs}.npy", predicted_angles_eval)
        # print("Saved final predicted angles to saves/pendulum_final.npy")

    return loss_history

    
if __name__ == "__main__":
    # Parameters for the pendulum
    g = 9.81  # Acceleration due to gravity
    L = 1.0   # Length of the pendulum
    theta0 = np.pi / 4  # Initial angle (45 degrees)
    omega0 = 0.0        # Initial angular velocity
    t_span = (0, 10)    # Time span for the simulation
    t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time points to evaluate
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define the pendulum dynamics
    def pendulum_dynamics(t, y):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = - (g / L) * np.sin(theta)
        return [dtheta_dt, domega_dt]
    
    # Solve the ODE to get the true angles over time
    sol = solve_ivp(pendulum_dynamics, t_span, [theta0, omega0], t_eval=t_eval)
    true_angles = sol.y[0]  # Extract the angles from the solution
    
    # Pick a subset of the data for training
    n_samples = 20
    indices = np.random.choice(len(t_eval), n_samples, replace=False)
    t_train = torch.tensor(t_eval[indices], dtype=torch.float32).unsqueeze(1)  # Used for the MSE loss, shape (n_samples, 1)
    true_angles_train = torch.tensor(true_angles[indices], dtype=torch.float32).unsqueeze(1)  # Used for the MSE loss, shape (n_samples, 1)
    t_physics = torch.linspace(t_span[0], t_span[1], 5000, dtype=torch.float32).unsqueeze(1).requires_grad_(True)  # Used for the physical loss, shape (1000, 1)
    t_eval_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(1)  # Used for evaluation, shape (10000, 1)
    
    # Add some noise to the training data to make it more realistic
    noise_std = 0.05
    true_angles_train += noise_std * torch.randn_like(true_angles_train)
    
    # Initialize the model and train it
    num_epochs = 6000
    model = PendulumMLP()
    loss_history = train_model(model, t_train, true_angles_train, t_physics, t_eval_tensor, num_epochs=num_epochs, learning_rate=0.001, lambda_physical=0)
    
    # Plot the loss history
    fig, ax = plt.subplots()
    ax.plot(loss_history, label='Total Loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(zorder=0)
    ax.set_xlim(0, len(loss_history))
    
    # Create an animation of the predicted angles over time
    fig2, ax2 = plt.subplots()
    ax2.set_xlim(t_span)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Pendulum Angle Over Time')
    line, = ax2.plot([], [], label='Predicted Angle', color='red')
    scat = ax2.scatter(t_train.numpy(), true_angles_train.numpy(), label='Training Data', color='black', zorder=5, marker='x')
    true_line, = ax2.plot(t_eval, true_angles, label='True Angle', color='black', linestyle='dashed', zorder=4, alpha=0.7)
    ax2.legend()
    ax2.grid(zorder=0)
    
    def update(frame):
        predicted_angles = np.load(f"saves_without_physical_loss/pendulum_epoch_{frame}.npy")
        line.set_data(t_eval, predicted_angles)
        return line,
    
    anim = FuncAnimation(fig2, update, frames=range(0, num_epochs + 1, 200), blit=True)
    anim.save('pendulum_animation_without_physical_loss.gif', writer='pillow', fps=5)
    plt.show()
    