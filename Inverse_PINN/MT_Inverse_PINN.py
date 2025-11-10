# ---------------------------------------------------------------------------------------------------------------
# Libraries and Configuration Setting
# ---------------------------------------------------------------------------------------------------------------
import os
os.environ["DDE_BACKEND"] = "pytorch"

import torch
import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepxde.icbc import PointSetBC, PeriodicBC

torch.cuda.empty_cache()

dde.config.set_random_seed(2025)
dde.config.set_default_float("float32")

# ---------------------------------------------------------------------------------------------------------------
# Defining Problem Parameters
# ---------------------------------------------------------------------------------------------------------------

# Spatial domain of the problem
Start_Length = 0.0
End_Length = 1.0

# Time domain of the problem
T_start = 0.0
T_end = 1.0

# Chemical driving force
DelF = 0.036
# Kinetic coefficient
L = 10
# Elastic constants
C11 = 264.24
C12 = 115.38
C44 = 153.86
# Energy density coefficients
a, b, c = 0.2, -12.6, 12.4  

# ---------------------------------------------------------------------------------------------------------------
# Scaling Functions
# ---------------------------------------------------------------------------------------------------------------

# Output scaling factors
u_scale = 1e-3 # Scaling factor for u1 and u2
eta_scale = 1e-1 # Scaling factor for eta1 and eta2
def transform_func(X, Y):
    # Extracting individual outputs
    u1_raw, u2_raw, eta1_raw, eta2_raw = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4]
    # Applying scaling
    u1_scaled = u1_raw * u_scale 
    u2_scaled = u2_raw * u_scale
    eta1_scaled = eta1_raw * eta_scale
    eta2_scaled = eta2_raw * eta_scale
    # Transformed outputs
    return torch.cat([u1_scaled, u2_scaled, eta1_scaled, eta2_scaled], dim=1)

# ---------------------------------------------------------------------------------------------------------------
# Defining the PDE Function
# ---------------------------------------------------------------------------------------------------------------
def pde(x, y):

    # Outputs of the neural network, u1, u2, eta1, eta2:  u1, and u2 are the displacement components, and eta1 and eta2 are the order parameters.
    u1, u2, eta1, eta2 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    u1_x = dde.grad.jacobian(y, x, i=0, j=0) # du1/dx
    u1_y = dde.grad.jacobian(y, x, i=0, j=1) # du1/dy

    u2_x = dde.grad.jacobian(y, x, i=1, j=0) # du2/dx
    u2_y = dde.grad.jacobian(y, x, i=1, j=1) # du2/dy

    eta1_xx = dde.grad.hessian(y, x, component=2, i=0, j=0) # d^2eta1/dx^2
    eta1_yy = dde.grad.hessian(y, x, component=2, i=1, j=1) # d^2eta1/dy^2
    
    eta2_xx = dde.grad.hessian(y, x, component=3, i=0, j=0) # d^2eta2/dx^2
    eta2_yy = dde.grad.hessian(y, x, component=3, i=1, j=1) # d^2eta2/dy^2

    eta1_t = dde.grad.jacobian(y, x, i=2, j=2) # deta1/dt
    eta2_t = dde.grad.jacobian(y, x, i=3, j=2) # deta2/dt

    # Stress components
    sigma11 = (7.443 * (eta2 ** 2 - eta1 ** 2) + C11 * u1_x + C12 * u2_y)
    sigma12 = (C44 * (u2_x + u1_y))
    sigma21 = sigma12
    sigma22 = (7.443 * (eta1 ** 2 - eta2 ** 2) + C12 * u1_x + C11 * u2_y)


    #  Stress derivatives and Mechanical equilibrium equations (Eq. 1) and (Eq. 2):
    eq1 = dde.grad.jacobian(sigma11, x, j=0) + dde.grad.jacobian(sigma12, x, j=1) # dsigma11/dx + dsigma12/dy
    eq2 = dde.grad.jacobian(sigma21, x, j=0) + dde.grad.jacobian(sigma22, x, j=1) # dsigma21/dx + dsigma22/dy

    # divergence of flux  =  -LB * (eta_xx + eta_yy)
    div_Gamma_eta1 = -(L*Beta_trainable) * (eta1_xx + eta1_yy)
    div_Gamma_eta2 = -(L*Beta_trainable) * (eta2_xx + eta2_yy)

    # Source terms for TDGL (Eq. 3) and (Eq. 4):
    f1 = L * ((1.4886 * eta1 * (-eta1 ** 2 + eta2 ** 2 + 10 * u1_x - 10 * u2_y)) - (DelF * (a * eta1 + b * eta1 ** 2 + c * eta1 * (eta1 ** 2 + eta2 ** 2))))
    f2 = -L * ((1.4886 * eta2 * (-eta1 ** 2 + eta2 ** 2 + 10 * u1_x - 10 * u2_y)) + (DelF * (a * eta2 + b * eta2 ** 2 + c * eta2 * (eta1 ** 2 + eta2 ** 2))))

    # TDGL equations (Eq. 3) and (Eq. 4):
    equ3 = eta1_t + div_Gamma_eta1 - f1
    equ4 = eta2_t + div_Gamma_eta2 - f2

    return [eq1, eq2, equ3, equ4]

# ---------------------------------------------------------------------------------------------------------------
# Defining Geometry and Time Domain
# ---------------------------------------------------------------------------------------------------------------

geom = dde.geometry.Rectangle(xmin=[Start_Length, Start_Length], xmax=[End_Length, End_Length]) # Spatial domain
timedomain = dde.geometry.TimeDomain(T_start, T_end) # Time domain
geomtime = dde.geometry.GeometryXTime(geom, timedomain) # Geometry and time domain

# ---------------------------------------------------------------------------------------------------------------
# Boundary Conditions
# --------------------------------------------------------------------------------------------------------------

# Boundary checking in x (x1) directions
def boundary_x(X, on_boundary):
    return on_boundary and (np.isclose(X[0], 0) or np.isclose(X[0], 1))

# Boundary checking in y (x2) directions
def boundary_y(X, on_boundary):
    return on_boundary and (np.isclose(X[1], 0) or np.isclose(X[1], 1))


# Periodic BCs in x (x1) direction
bc_eta1_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=2)  
bc_eta2_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=3) 

# Periodic BCs in y (x2) direction
bc_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=2)
bc_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=3)

# ---------------------------------------------------------------------------------------------------------------
# Initial Condition for the Inverse PINN
# ---------------------------------------------------------------------------------------------------------------

base_dir = "ic_file_path/"
ic_path = os.path.join(base_dir, "IC_inverse_PINN.txt")
ic_data = np.loadtxt(ic_path, delimiter=",")

xy_ic = ic_data[:, 0:2]
eta1_ic = ic_data[:, 2:3]
eta2_ic = ic_data[:, 3:4]
xyt_ic = np.hstack((xy_ic, np.zeros((xy_ic.shape[0], 1))))

ic_eta1 = PointSetBC(xyt_ic, eta1_ic, component=2)
ic_eta2 = PointSetBC(xyt_ic, eta2_ic, component=3)

# ---------------------------------------------------------------------------------------------------------------
# Observed data with different Grids to run
# ---------------------------------------------------------------------------------------------------------------

grid_names = ["grid_101x101", "grid_51x51", "grid_21x21", "grid_11x11"]

# Directory for saving models and plots
results_dir = base_dir
os.makedirs(results_dir, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------
# Loop over all grid files
# ---------------------------------------------------------------------------------------------------------------

for grid_name in grid_names:
    print(f"\n==================== Running inverse PINN for {grid_name} ====================")

    # ---------------------------------------------------------------------------------------------------------------
    # Observed data
    # ---------------------------------------------------------------------------------------------------------------
    obs_path = os.path.join(base_dir, f"{grid_name}.txt")
    observe_data = np.loadtxt(obs_path, delimiter=",")

    xy_observ = observe_data[:, 0:2]
    eta1_observ = observe_data[:, 2:3]
    eta2_observ = observe_data[:, 3:4]
    xyt_observ = np.hstack(
        (xy_observ, np.full((xy_observ.shape[0], 1), T_end))
    )

    observ_eta1 = PointSetBC(xyt_observ, eta1_observ, component=2)
    observ_eta2 = PointSetBC(xyt_observ, eta2_observ, component=3)

    # Anchor points (IC + observed)
    anchor_points = np.vstack((xyt_ic, xyt_observ))

    # ---------------------------------------------------------------------------------------------------------------
    # Network and data
   # ---------------------------------------------------------------------------------------------------------------
    layer_size = [3] + [128] * 4 + [4]
    activation = "tanh"
    initializer = "Glorot uniform"

    net = dde.nn.FNN(layer_size, activation, initializer)
    net.apply_output_transform(transform_func)

    # Data object for the PDE
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_eta1_x, bc_eta2_x, bc_eta1_y, bc_eta2_y,
         ic_eta1, ic_eta2, observ_eta1, observ_eta2],
        num_domain=20000,
        num_boundary=4000,
        train_distribution="Hammersley",
        anchors=anchor_points,
        num_test=50000,
    )

    model = dde.Model(data, net)

    # ---------------------------------------------------------------------------------------------------------------
    # Beta and callbacks
    # ---------------------------------------------------------------------------------------------------------------
    # New trainable Beta for this grid
    Beta_trainable = dde.Variable(1.0)

    # Resampler
    pde_resampler = dde.callbacks.PDEPointResampler(period=1000, pde_points=True, bc_points=True)

    # Beta tracker (one file per grid)
    beta_path = os.path.join(results_dir, f"{grid_name}_beta_vs_iterations.txt")
    Beta_tracker = dde.callbacks.VariableValue(
        Beta_trainable,
        period=100,
        filename=beta_path,
        precision=5,
    )

    # Loss weights
    l_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 1000]

    # ---------------------------------------------------------------------------------------------------------------
    # 4. Training phases with different learning rates
    # ---------------------------------------------------------------------------------------------------------------

    # Phase 1: lr = 1e-3
    model.compile("adam", lr=1e-3, loss="MSE", loss_weights=l_weights, external_trainable_variables=[Beta_trainable])
    losshistory, train_state = model.train(iterations=30000, display_every=1000, callbacks=[Beta_tracker, pde_resampler])

    # Phase 2: lr = 1e-4
    model.compile("adam", lr=1e-4, loss="MSE", loss_weights=l_weights, external_trainable_variables=[Beta_trainable])
    losshistory, train_state = model.train(iterations=10000, display_every=1000, callbacks=[Beta_tracker, pde_resampler])

    # Phase 3: lr = 1e-5
    model.compile("adam", lr=1e-5, loss="MSE", loss_weights=l_weights, external_trainable_variables=[Beta_trainable])
    losshistory, train_state = model.train(iterations=10000, display_every=1000, callbacks=[Beta_tracker, pde_resampler])

    # L-BFGS phase can be added here
    # dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=1e-12, gtol=1e-10, maxiter=50000, maxfun=None, maxls=50)
    # model.compile("L-BFGS", loss="MSE", loss_weights=initial_weights, external_trainable_variables=[Beta_trainable])
    # losshistory, train_state = model.train(display_every=100, callbacks=[Beta_tracker, pde_resampler])

    # ---------------------------------------------------------------------------------------------------------------
    # 5. Saving model
   # ---------------------------------------------------------------------------------------------------------------
    model_save_path = os.path.join(results_dir, f"{grid_name}_model")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
