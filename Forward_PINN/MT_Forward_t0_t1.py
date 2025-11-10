# ---------------------------------------------------------------------------------------------------------------
# Importing Libraries and Configuration Setting
# ---------------------------------------------------------------------------------------------------------------
import os
os.environ["DDE_BACKEND"] = "pytorch"
import torch
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.callbacks import Callback
from deepxde.icbc import PeriodicBC
from deepxde.icbc import PointSetBC

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

# Gradient energy coefficient
Beta = 1e-4
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
u_scale = 1e-5 # Scaling factor for u1 and u2
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
    div_Gamma_eta1 = -(L*Beta) * (eta1_xx + eta1_yy)
    div_Gamma_eta2 = -(L*Beta) * (eta2_xx + eta2_yy)

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
# Initial Conditions
# ---------------------------------------------------------------------------------------------------------------

# Initial condition function for eta1 for time step [0, T1]
def eta1_ic_func(X):
    x = X[:, 0:1] # x-coordinate (x1)
    y = X[:, 1:2] # y-coordinate (x2)
    return 0.1 + 0.4 * np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * 0.05**2))

# Initial condition function for eta2 for time step [0, T1]
def eta2_ic_func(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.5 - 0.4 * np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * 0.05**2))

# Initial condition applied at t = T_start
def initial_time(X, on_initial):
    return on_initial and np.isclose(X[2], 0.0)

# Initial conditions for eta1 and eta2
ic_eta1 = dde.IC(geomtime, eta1_ic_func, initial_time, component=2)
ic_eta2 = dde.IC(geomtime, eta2_ic_func, initial_time, component=3)


# ---------------------------------------------------------------------------------------------------------------
# Adaptive Refinement Strategy
# ---------------------------------------------------------------------------------------------------------------
def adaptive_sampling_eta(model, geomtime, pde, num_samples, neighborhood_size):

    # Generating a large set of random points
    x = geomtime.random_points(100000)
    # Computing the residuals at these points
    residuals = model.predict(x, operator=pde)

    # Residuals for each ouput
    u1_residual = np.abs(residuals[0])  # First residual is for u1
    u2_residual = np.abs(residuals[1])  # Second residual is for u2
    eta1_residual = np.abs(residuals[2])  # Third residual is for eta1
    eta2_residual = np.abs(residuals[3])  # Fourth residual is for eta2

    # Combining the residuals
    residual_combined = u1_residual+ u2_residual+ eta1_residual + eta2_residual

    # Sorting the indices of the points by the highest combined residuals
    sorted_indices = np.argsort(-residual_combined.flatten())[:num_samples]

    # Selecting the top points with the highest errors 
    high_error_points = x[sorted_indices]

    # Ensuring unique sampling if required
    if neighborhood_size > 0:
        new_points = []
        for point in high_error_points:
            # Generating points around the high-error points within a defined neighborhood
            for _ in range(3):  # Generating multiple new points for each high-error point
                perturbation = np.random.uniform(-neighborhood_size, neighborhood_size, size=point.shape)
                new_point = point + perturbation
                new_point[:2] = np.clip(new_point[:2], geomtime.geometry.bbox[0], geomtime.geometry.bbox[1])
                new_points.append(new_point)
        
        new_points = np.array(new_points)
        # Combining high-error points with neighborhood-generated points
        high_error_points = np.vstack([high_error_points, new_points])

    # Ensuring uniqueness of the points before returning
    high_error_points = np.unique(high_error_points, axis=0)

    print(f"Selected {len(high_error_points)} unique high-error points")

    # Returning the selected points
    return high_error_points


# ---------------------------------------------------------------------------------------------------------------
# Defining Neural Network and Model Creation
# ---------------------------------------------------------------------------------------------------------------

layer_size = [3] + [128] * 4 + [4]  # 3 inputs (x (x1), y (x2), t), 4 outputs (u1, u2, eta1, eta2)
activation = "tanh" # Activation function
initializer = "Glorot uniform" # Initializer for weights and biases
net = dde.nn.FNN(layer_size, activation, initializer) #  Neural network
# Applying the scaling transformation
net.apply_output_transform(transform_func)

# Data object for the PDE
data = dde.data.TimePDE(
            geomtime, # Spatial and time domain
            pde, # Partial differential equation
            [bc_eta1_x, bc_eta2_x, bc_eta1_y, bc_eta2_y, ic_eta1, ic_eta2], # Boundary and initial conditions
            num_domain=30000, # Number of domain points
            num_boundary=4000, # Number of boundary points
            num_initial=512, # Number of initial condition points
            train_distribution='Hammersley', # Distribution sequence for sampling points
            num_test=50000, # Number of testing points
        )

# Model building
model = dde.Model(data, net)


# ---------------------------------------------------------------------------------------------------------------
# Setting Up the Optimizer, Callbacks, and Loss Function, and model for training
# ---------------------------------------------------------------------------------------------------------------

# PDE resampler
pde_resampler = dde.callbacks.PDEPointResampler(period=1000, pde_points=True, bc_points=True)

# Weighting coefficients for the loss function
l_weight = [1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0] 

# Model restoration for transfer learning
# model.compile("L-BFGS", loss='MSE', loss_weights=l_weight)
# #model.restore("saved_model_path")

# # Adam optimizer training 
model.compile("adam", lr=1e-3, loss='MSE', loss_weights=l_weight)
losshistory, train_state = model.train(iterations=50000, display_every=1000, callbacks=[pde_resampler], model_save_path=f"./model_path")

# L-BFGS optimizer training
dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=1e-12, gtol=1e-10, maxiter=50000, maxfun=None, maxls=50)
model.compile("L-BFGS", loss='MSE', loss_weights = l_weight)
losshistory, train_state = model.train(callbacks=[pde_resampler], model_save_path=f"./model_path")


# ---------------------------------------------------------------------------------------------------------------
# Training with adaptive refinement IF needed 
# ---------------------------------------------------------------------------------------------------------------

# Adaptive refinement parameters
num_sampling_rounds = 0  # number of sampling rounds
num_samples = 5000  # number of points to be added during each sampling round
for sampling_round in range(num_sampling_rounds):
    print(f"Sampling round {sampling_round + 1}/{num_sampling_rounds}")

    # Generating new high-error points
    x_new = adaptive_sampling_eta(model, geomtime, pde, num_samples=num_samples, neighborhood_size=0.01)

    # Converting existing training data points to a set for efficient lookup
    existing_points_set = set(map(tuple, model.data.train_x))
    print(f"Number of existing points: {len(existing_points_set)}")

    # Filtering out points that already exist in the training data
    unique_new_points = np.array([point for point in x_new if tuple(point) not in existing_points_set])

    # If there are any unique points, evaluating their errors
    if len(unique_new_points) > 0:
        # Predicting the residuals or errors at these points
        residuals = model.predict(unique_new_points, operator=pde)
        residuals_combined = np.sum([np.abs(res) for res in residuals], axis=0)

        # Sorting points by error magnitude (descending order)
        sorted_indices = np.argsort(-residuals_combined.flatten())
        top_error_indices = sorted_indices[:num_samples]

        # Selecting the top error points
        unique_new_points = unique_new_points[top_error_indices]

    # Adding the unique high-error points to the training data if any
    if len(unique_new_points) > 0:
        print(f"Adding {len(unique_new_points)} new unique points based on high-error regions")
        model.data.add_anchors(unique_new_points)

        # ADAM optimizer training with refinement strategy
        model.compile("adam", lr=1e-3, loss='MSE', loss_weights = l_weight)
        losshistory, train_state = model.train(iterations=20000, display_every=1000, callbacks=[pde_resampler], model_save_path=f"./time_models/model_time_step_adam_best")

        # L-BFGS optimizer training with refinement strategy
        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=1e-12, gtol=1e-10, maxiter=20000, maxfun=None, maxls=50)
        model.compile("L-BFGS", loss='MSE', loss_weights = l_weight)
        losshistory, train_state = model.train(callbacks=[pde_resampler], model_save_path=f"./time_models/model_time_step_final_best")
    else:
        print("No new unique points to add in this sampling round.")



# ---------------------------------------------------------------------------------------------------------------
# Loading reference data for comparison with PINN predictions for eta1 and eta2 at t = T_start, and t = T_end
# ---------------------------------------------------------------------------------------------------------------

# Reference text file for the initial time (t = T_start)
ref_data_start_time = np.loadtxt(f"/home/asfandyarkhan/Desktop/Papers/MT_Paper/CMAME/PINN_Code/Forward/Final_Foward_GitHub/Ref_t0_.txt", delimiter=',')

# Reference text file for the end time (t = T_end)
ref_data_end_time =np.loadtxt(f"/home/asfandyarkhan/Desktop/Papers/MT_Paper/CMAME/PINN_Code/Forward/Final_Foward_GitHub/Ref_t1_.txt", delimiter=',')

# Unpacking the data for the initial time, Note: data has four columns: x, y, eta1, eta2
start_time_data = (ref_data_start_time[:, 0], ref_data_start_time[:, 1], ref_data_start_time[:, 2], ref_data_start_time[:, 3])

# Unpacking the data for the end time, Note: data has four columns: x, y, eta1, eta2
end_time_data = (ref_data_end_time[:, 0], ref_data_end_time[:, 1], ref_data_end_time[:, 2], ref_data_end_time[:, 3])


# ---------------------------------------------------------------------------------------------------------------
# Plotting the comparison between Reference and PINN results for eta1 and eta2
# ---------------------------------------------------------------------------------------------------------------
def plot_comparison(start_time_data, end_time_data, model):

    # Defining min and max values based on Ref data
    T_start_domain = T_start  # Time step from Reference data
    T_end_domain = T_end  # Next time step in Reference data

    # Defining the plotting parameters
    Font_size = 9
    Axis_size = 8
    levels_number = 200
    cmap_color = 'jet'

    # Ref data 
    x_ref_start, y_ref_start, ref_eta1_start, ref_eta2_start = start_time_data
    x_ref_end, y_ref_end, ref_eta1_end, ref_eta2_end = end_time_data

    # Preparing input data for PINN predictions
    t_norm_start = np.full_like(x_ref_start, T_start_domain)
    X_input_start = np.column_stack((x_ref_start, y_ref_start, t_norm_start))

    t_norm_end = np.full_like(x_ref_end, T_end_domain)
    X_input_end = np.column_stack((x_ref_end, y_ref_end, t_norm_end))

    # Model prediction for both time steps
    predictions_start = model.predict(X_input_start)
    predictions_end = model.predict(X_input_end)

    # Extracting eta1 and eta2 predictions
    predicted_eta1_start = predictions_start[:, 2]
    predicted_eta2_start = predictions_start[:, 3]
    predicted_eta1_end = predictions_end[:, 2]
    predicted_eta2_end = predictions_end[:, 3]

    # Grid for contour plots
    grid_x_start = np.unique(x_ref_start)
    grid_y_start = np.unique(y_ref_start)
    X_start, Y_start = np.meshgrid(grid_x_start, grid_y_start)

    grid_x_end = np.unique(x_ref_end)
    grid_y_end = np.unique(y_ref_end)
    X_end, Y_end = np.meshgrid(grid_x_end, grid_y_end)

    # Ref and predicted data to fit the grid
    Reshaped_ref_eta1_start = ref_eta1_start.reshape(len(grid_y_start), len(grid_x_start))
    Reshaped_ref_eta2_start = ref_eta2_start.reshape(len(grid_y_start), len(grid_x_start))
    Reshaped_predicted_eta1_start = predicted_eta1_start.reshape(len(grid_y_start), len(grid_x_start))
    Reshaped_predicted_eta2_start = predicted_eta2_start.reshape(len(grid_y_start), len(grid_x_start))

    Reshaped_ref_eta1_end = ref_eta1_end.reshape(len(grid_y_end), len(grid_x_end))
    Reshaped_ref_eta2_end = ref_eta2_end.reshape(len(grid_y_end), len(grid_x_end))
    Reshaped_predicted_eta1_end = predicted_eta1_end.reshape(len(grid_y_end), len(grid_x_end))
    Reshaped_predicted_eta2_end = predicted_eta2_end.reshape(len(grid_y_end), len(grid_x_end))

    # Absolute error
    error_eta1_start = np.abs(Reshaped_predicted_eta1_start - Reshaped_ref_eta1_start)
    error_eta2_start = np.abs(Reshaped_predicted_eta2_start - Reshaped_ref_eta2_start)
    error_eta1_end = np.abs(Reshaped_predicted_eta1_end - Reshaped_ref_eta1_end)
    error_eta2_end = np.abs(Reshaped_predicted_eta2_end - Reshaped_ref_eta2_end)

    # Creating plots
    plt.figure(figsize=(10, 14))

    def plot_subplot(index, X, Y, data, title):
        plt.subplot(4, 3, index)
        contour = plt.contourf(X, Y, data, levels=levels_number, cmap=cmap_color)
        cbar = plt.colorbar(contour)
        cbar.ax.tick_params(labelsize=Axis_size)
        plt.axis("off")

    # Ref results (eta1 and eta2)
    plot_subplot(1, X_start, Y_start, Reshaped_ref_eta1_start, f'Ref $\\eta_1$ at $t$={T_start_domain}')
    plot_subplot(4, X_end, Y_end, Reshaped_ref_eta1_end, f'Ref $\\eta_1$ at $t$={T_end_domain}')
    plot_subplot(7, X_start, Y_start, Reshaped_ref_eta2_start, f'Ref $\\eta_2$ at $t$={T_start_domain}')
    plot_subplot(10, X_end, Y_end, Reshaped_ref_eta2_end, f'Ref $\\eta_2$ at $t$={T_end_domain}')

    # PINN results (eta1 and eta2)
    plot_subplot(2, X_start, Y_start, Reshaped_predicted_eta1_start, f'PINN $\\eta_1$ at $t$={T_start_domain}')
    plot_subplot(5, X_end, Y_end, Reshaped_predicted_eta1_end, f'PINN $\\eta_1$ at $t$={T_end_domain}')
    plot_subplot(8, X_start, Y_start, Reshaped_predicted_eta2_start, f'PINN $\\eta_2$ at $t$={T_start_domain}')
    plot_subplot(11, X_end, Y_end, Reshaped_predicted_eta2_end, f'PINN $\\eta_2$ at $t$={T_end_domain}')

    # Absolute error (eta1 and eta2)
    plot_subplot(3, X_start, Y_start, error_eta1_start, f'Error $|\\eta_1|$ at $t$={T_start_domain}')
    plot_subplot(6, X_end, Y_end, error_eta1_end, f'Error $|\\eta_1|$ at $t$={T_end_domain}')
    plot_subplot(9, X_start, Y_start, error_eta2_start, f'Error $|\\eta_2|$ at $t$={T_start_domain}')
    plot_subplot(12, X_end, Y_end, error_eta2_end, f'Error $|\\eta_2|$ at $t$={T_end_domain}')

    # Final adjustments
    plt.tight_layout()
    plt.show()

# Calling function to compare Ref and PINN results
plot_comparison(start_time_data, end_time_data, model)