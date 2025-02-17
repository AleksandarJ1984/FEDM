"""
In this example, the glow discharge in argon at low pressure is modelled. The aim is to demonstrate
the use of the module functions for the case when several particle species and numerous processes are considered
in the model. Furthermore, the local mean energy approximation is used in this example. The transport and rate coefficients as well
as the reaction scheme required for generating gain, loss and power matrices, used to define the rates and source
term, are imported in the program from the external file. The transport and rate coefficients are stored in look-up
tables and interpolated for a given mean electron energy or reduced electric field in a function, when needed.
As in previous example, the coupled i.e., fully implicit approach, consisting in solving of the all equations at once is applied.
The semi-implicit treatment of the reaction rate and transport coefficients is used.
Again, the adaptive backwards differentiation formula of the order of two
with the PID controller for time-step size are used.
"""
from dolfin import *
import numpy as np
from timeit import default_timer as timer
import time
import os
import sys
from fedm.physical_constants import *
from fedm.file_io import *
from fedm.functions import *

# Optimization parameters.
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["std_out_all_processes"] = False
parameters['krylov_solver']['nonzero_initial_guess'] = True
parameters["form_compiler"]["quadrature_degree"] = 4


# Defining type of used solver and its parameters.
linear_solver = "mumps"  # Type of linear solvers: lu | mumps | gmres | bicgstab
maximum_iterations = 20 # Maximum number of nonlinear iterations
relative_tolerance = 1e-4 # Relative tolerance of nonlinear solver

# ============================================================================
# Definition of the simulation conditions, model, approximation, coordinates and path for input files
# ============================================================================
model = '4_particles'
coordinates = 'cylindrical'
semi_implicit = True
gas = 'Ar'
Tgas = 300.0  # [K]
p0 = 1.0   #[Torr]
N0 = p0*3.21877e22  #[m^-3]
U_w  = -250.0   #[V]
approximation = 'LMEA'  # Type of approximation used in the model
path = files.file_input / model  # Path where input files for desired model are stored
file_type = 'pvd'

# ============================================================================
# Reading species list and particle properties, obtaining number of species for which the problem is solved and creating
#  output files.
# ============================================================================
number_of_species, particle_species, particle_prop, particle_species_file_names = read_speclist(path)  # Reading species list to obtain number of species, their name and corresponding file names
M, sign = read_particle_properties(particle_prop, model)  # Reading particle properties from input files
charge = [i * elementary_charge for i in sign]
equation_type = ['reaction', 'diffusion-reaction', 'drift-diffusion-reaction', 'drift-diffusion-reaction']  # Defining the type of the equation (reaction | diffusion-reaction | drift-diffusion-reaction)
particle_type = ['Heavy', 'Heavy', 'Heavy', 'electrons']  # Defining particle type required for boundary condition: Heavy | electrons
particle_species_type = ['Neutral', 'Neutral', 'Ion', 'electrons']  # Defining particle type required for boundary condition: Neutral | Ion | electrons
n_ic = [N0, 1e12, 1e12, 1e12]  # Initial number density values

grad_diff = [pst == 'electrons' for pst in particle_species_type]

# ============================================================================
# Importing reaction matrices, reaction coefficients required for rates and transport coefficients
# ============================================================================
power_matrix, loss_matrix, gain_matrix = reaction_matrices(path, particle_species)  # Importing reaction matrices
k_file_names = rate_coefficient_file_names(path)  # Importing file names containing given rate coefficient
energy_loss = read_energy_loss(path)  # Reading energy loss for given reaction
number_of_reactions = len(k_file_names)  # Number of reactions

mu_x, mu_y, mobility_dependence = read_transport_coefficients(particle_species_file_names, 'mobility', model)  # Reading mobilities and dependence from files
D_x, D_y, Diffusion_dependence = read_transport_coefficients(particle_species_file_names, 'Diffusion', model)  # Reading diffusion coefficients and dependence from files
k_dependence = read_dependences(k_file_names)  # Reading dependence from files, required for rate coefficient interpolation
k_x, k_y = read_rate_coefficients(k_file_names, k_dependence)  # Reading rate coefficients

De_diff = np.gradient(D_y[number_of_species - 1], D_x[number_of_species - 1])/N0
mue_diff = np.gradient(mu_y[number_of_species - 1], mu_x[number_of_species - 1])/N0

k_diff = []

i = 0
while i < len(k_y):
    if k_dependence[i] == "Umean":
        k_diff.append(np.gradient(k_y[i], k_x[i]))
    else:
        k_diff.append(0.0)
    i += 1

# Setting up number of equations for given approximation
number_of_species, number_of_equations, particle_species, M, sign = modify_approximation_vars(approximation, number_of_species, particle_species, M, sign)

xdmf_file_u = output_files('xdmf', 'number density', particle_species_file_names)  # Creating list of output files
vtkfile_Phi = output_files('pvd', 'potential', ['Phi'])  # Creating list of output files
output_file_list = [vtkfile_Phi[0], xdmf_file_u[1], xdmf_file_u[2], xdmf_file_u[3]]  # Creating list of variables for output
file_type = ['pvd', 'xdmf', 'xdmf', 'xdmf'] # creates list of variables for output

# ============================================================================
# Definition of the time variables and relative error used for adaptive time stepping.
# ============================================================================
t_old = None  # Previous time step
t0 =  0.0  # Initial time step
t = t0  # Current time step
T_final = 5e-5  # Simulation time end [s]

dt_min = 1e-15  # Minimum time step [s]
dt_max = 1e-8  # Maximum time step [s]
dt_init = 1e-13  # Initial time step size [s]
dt_old_init = 1e30  # Initial time step size [s], extremely large value is used to initiate adaptive BDF2
dt = Expression("time_step", time_step = dt_init, degree = 0)  # Time step size [s]
dt_old = Expression("time_step", time_step = dt_old_init, degree = 0)  # Time step size expression [s], Initial value is set up to be large in order to reduce initial step of adaptive BDF formula to one.
dt_old1 = Expression("time_step", time_step = dt_old_init, degree = 0)  # Time step size expression [s], Initial value is set up to be large in order to reduce initial step of adaptive BDF formula to one.

ttol = 5e-4  # Tolerance for adaptive time stepping

### Setting-up output times and time steps. t_output_list and t_output_step_list need to have the equal length
t_output_list = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # List of time step intervals (consisting of two consecutive components) at which the results are printed to file
t_output_step_list = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-6]  # List of time step sizes for corresponding interval
t_output_step = t_output_list[0]  # Current time step at which the results are printed to file
t_output = t_output_step_list[0]  # Current output time length

number_of_iterations = None  # Nonlinear solver iteration number counter
convergence = False  # Nonlinear solver convergence values, True or False
error = [0.0]*(number_of_species+1)  # List of error values for particle species
max_error = [1]*3  # List of maximum error values in current, k-1 and k-2 time steps

# ============================================================================
# Defining the geometry of the problem and corresponding boundaries.
# ============================================================================
if coordinates == 'cylindrical':
    r = Expression('x[0]', degree = 1)
    z = Expression('x[1]', degree = 1)

gap_length = 0.01   # [m]
wall = 0.01  # [m]

boundaries = [['line', 0.0, 0.0, 0.0, wall],\
                        ['line', gap_length, gap_length, 0.0, wall],\
                        ['line', 0.0, gap_length, 0.0, 0.0],\
                        ['line', 0.0, gap_length, wall, wall]]  # Defining list of boundaries (line | circle) lying between z0 and z1 and r0 and r1
number_of_boundaries = len(boundaries)

ref_metallic = [0.3, 0.3, 5e-4, 0.3]  # Reflection coefficient for metalic surface for each particle
ref_zero = [1.0, 1.0, 1.0, 1.0]  # By seting reflection coefficient to one, zero flux boundary condition is applied in Boundary_flux() function
ref_coeff = [ref_metallic, ref_metallic, ref_zero, ref_zero]  # List of reflection coefficients for given boundary
gamma_metallic = 0.06  # Secondary electron emission coefficient for metalic surface
gamma = [gamma_metallic, gamma_metallic, 0, 0]  # List of secondary emission coefficients for the given boundaries
we_metalic = 5.0  # Mean energy of secondary emitted electrons [eV]

log('conditions', files.model_log, dt.time_step, U_w, p0, gap_length, N0, Tgas)  # Writting simulation conditions to log file
log('properties', files.model_log, gas, model, particle_species_file_names, M, charge)  # Writting particle properties into a log file

# ===========================================================================================================================
# Mesh setup and boundary measure redefinition. Structured mesh is generated using built-in mesh generator
# ===========================================================================================================================
mesh_plasma = RectangleMesh(Point(0, 0), Point(wall, gap_length), 100, 100, "crossed")  # Generating structured triangular mesh

mesh_statistics(mesh_plasma)  # Prints number of elements, minimum and maximum cell diameter

boundary_mesh_function = Marking_boundaries(mesh_plasma, boundaries)  # Marking boundaries required for boundary conditions
ds_plasma = Measure('ds', domain = mesh_plasma, subdomain_data = boundary_mesh_function)  # Boundary measure redefinition
normal_plasma = FacetNormal(mesh_plasma)  # Boundary normal

File(str(files.output_folder_path / 'mesh' / 'boundary_mesh_function.pvd')) << boundary_mesh_function  # Writting boundary mesh function to file
log('matrices', files.model_log, gain_matrix, loss_matrix, power_matrix)  # Writting the reaction matrices that determine rates and source term
log('initial time', files.model_log, t)  # Time logging

# ============================================================================
# Defining type of elements and function space, test functions, trial functions and functions for storing postprocessing variables.
# ============================================================================
P1 = FiniteElement("Lagrange", mesh_plasma.ufl_cell(), 1)  # Defining finite element and degree
elements_list = Mixed_element_list(number_of_equations, P1)  # Creating list of elements
Element = MixedElement(elements_list)  # Defining mixed element

ME = FunctionSpace(mesh_plasma, Element)  # Defining mixed function space
V = FunctionSpace(mesh_plasma, P1)  # Defining function space
W = VectorFunctionSpace(mesh_plasma, 'P', 1)  # Defining vector function space

function_space_list = Function_space_list(number_of_equations, V)  # Defining function space list

assigner = FunctionAssigner(function_space_list, ME)  # Assigning values of variables between ME and V function spaces FunctionAssigner(receiving_space, assigning_space)
rev_assigner = FunctionAssigner(ME, function_space_list)  # Assigning values of variables between V and ME function spaces FunctionAssigner(receiving_space, assigning_space)

temp_output_variable = Function(V)  # Temporary variable used for file output

# Defining variables for coupled approach
u = TrialFunction(ME)  # Trial functions for mixed formulation
v = TestFunctions(ME)  # Test functions for mixed formulation
u_new = Function(ME)  # Functions for mixed formulation for current time step
u_old = Function(ME)  # Functions for mixed formulation for k-1 time step
u_old1 = Function(ME)  # Functions for mixed formulation for k-2 time step
F = 0

# Defining variables for solving initial Poisson's equation
u_phi = TrialFunction(V)  # Potential trial function
v_phi = TestFunction(V)  # Potential test function
Phi = Function(V)  # Potential function in current time step
Phi_old = Function(V)  # Potential function in k-1 time step
Phi_old1 = Function(V)  # Potential function in k-2 time step
rho_poisson = 0  # Volume charge density
rho_poisson_C = 0  # Volume charge density
redE = Function(V)  # Reduced electric field
redE_old = Function(V)  # Reduced electric field in previous time step
E = -grad(u[number_of_equations - 1])  # Electric field
E_magnitude = sqrt(inner(E, E))  # Electric field magnitude

#Defining variables for postprocessing
u_oldV = Function_definition(V, 'Function', number_of_species)  # Number density in previous time step (Function return functions for desired number of particles)
u_old1V = Function_definition(V, 'Function', number_of_species)  # Number density in previous time step (Function return functions for desired number of particles)
u_newV = Function_definition(V, 'Function', number_of_species)  # Number density in current time step (Function return functions for desired number of particles)
mean_energy = Function(V)  # Mean electron energy
mean_energy_old = Function(V)  # Mean electron energy
mean_energy_old1 = Function(V)  # Mean electron energy
mean_energy_e = mean_energy_old + (exp(u[0]) - exp(u[number_of_species-1])*mean_energy_old)/exp(u_oldV[number_of_species-1])
Gamma = [0]  # First element is zero for LMEA, since it is replaced with Gamma _en for the electron energy flux

vth = [0]*number_of_species  # Creating list of thermal velocities
i = 1
while i < number_of_species - 1:
    vth[i] = np.sqrt(8.0*kB*Tgas/(pi*M[i]))  # Defining thermal velocities for particle species
    i += 1
vth[number_of_species - 1] = sqrt(16.0*elementary_charge*mean_energy/(3.0*pi*M[number_of_species - 1]))  # Electron thermal velocity
D = Function_definition(V, 'Function', number_of_species)  # Defining list of variables for diffusion coefficients
D_diff = Function_definition(V, 'Function', number_of_species)  # Defining list of variables for diffusion coefficients
mu = Function_definition(V, 'Function', number_of_species)  # Defining list of variables for mobilities
mu_diff = Function_definition(V, 'Function', number_of_species)  # Defining list of variables for diffusion coefficients
rate_coefficient = Function_definition(V, 'Function', number_of_reactions)  # Defining list of variables for rate coefficients
rate_coefficient_diff = Function_definition(V, 'Function', number_of_reactions)  # Defining list of variables for rate coefficients
Rate = Function_definition(V, 'Function', number_of_reactions)  # Defining list of variables for rates
epsilon = Constant(1.0)*epsilon_0

# ============================================================================
# Setting up initial conditions
# ============================================================================
n_init = [0]*number_of_species  # List of initial values for particle species
i = 0
while i < number_of_species:
    n_init[i] = Expression('std::log(ic)', ic = n_ic[i], degree = 1)  # Defining initial values
    i += 1

mean_energy_init = interpolate(Expression('3.0', degree = 1), V)  # Initial mean energy
mean_energy.assign(mean_energy_init)  # Setting Initial mean energy
mean_energy_old.assign(mean_energy_init)  # Setting Initial mean energy
mean_energy_old1.assign(Constant(0.0))  # Setting Initial mean energy

i = 0
while i < number_of_species:
    u_newV[i].assign(n_init[i])  # Setting up initial number density
    u_oldV[i].assign(n_init[i])  # Setting up initial number density
    u_old1V[i].assign(Constant(0.0))  # Setting  up initial number density
    rho_poisson += elementary_charge*sign[i]*exp(u_oldV[i])  # Setting up charge density for initial potential calculation
    rho_poisson_C += elementary_charge*sign[i]*exp(u[i])  # Setting up charge density for coupled equations. In this case, trial functions are used as varaibles
    i += 1
we_newV = interpolate(Expression('std::log(a) + b', a = mean_energy, b = u_oldV[number_of_species-1], degree = 1), V)  # Setting  up initial energy density
we_oldV = interpolate(Expression('std::log(a) + b', a = mean_energy, b = u_oldV[number_of_species-1], degree = 1), V)  # Setting  up initial energy density
we_old1V = interpolate(Constant(0.0), V)

# Writing initial values to file
i = 0
while i < number_of_species:
    temp_output_variable.assign(u_oldV[i])  # Setting value for output
    temp_output_variable.rename(particle_species_file_names[i], str(i))  # Renaming variable for output
    xdmf_file_u[i].write_checkpoint(temp_output_variable, particle_species_file_names[i], t*1e6, XDMFFile.Encoding.HDF5, False) # Writting initial particle number densities to file
    i += 1

# ===========================================================================================================================
# Solving initial Poisson equation to calculate the potential for the initial time step.
# The potential is required for interpolation of transport coefficients that depend on reduced electric field.
# ===========================================================================================================================
Phi_grounded = Constant(0.0)   # Potential at the grounded electrode in [V]
Phi_powered = Expression('U0*(1-exp(-t/1e-9))', U0 = U_w, t = t, pi = pi, degree = 0)  # Potential at the powered electrode in [V]

def Powered_electrode(x, on_boundary):
    return near(x[1], 0, DOLFIN_EPS) and on_boundary

def Grounded_electrode(x, on_boundary):
    return near(x[1], gap_length, DOLFIN_EPS) and on_boundary

# Defining Dirichlet boundary conditions
Powered_Electrode_bc = DirichletBC(V, Phi_powered, Powered_electrode)
Grounded_bc = DirichletBC(V, Phi_grounded, Grounded_electrode)
Voltage_bcs = [Powered_Electrode_bc, Grounded_bc]  # List of Dirichlet boundary conditions

f_potential = rho_poisson/epsilon  # Poisson equation source term

# Definition variational formulation of Poisson equation for the initial time step
F_potential = weak_form_Poisson_equation(dx, u_phi, v_phi, f_potential, r)
a_potential, L_potential = lhs(F_potential), rhs(F_potential)

# Creating lhs and rhs and setting boundary conditions
A_potential = None
A_potential = assemble(a_potential, tensor = A_potential)
[bc.apply(A_potential) for bc in Voltage_bcs]
b_potential = None
b_potential = assemble(L_potential, tensor = b_potential)
[bc.apply(b_potential) for bc in Voltage_bcs]

# Solving Poisson's equation for the initial time step
solve(A_potential, Phi.vector(), b_potential, 'mumps') # Solving initial Poisson equation using default solver

Phi_old1.assign(Phi_old)  # Updating potential values
Phi_old.assign(Phi)  # Updating potential values

temp_output_variable.assign(Phi)  # Setting value for output
temp_output_variable.rename('Phi', str(0))  # Renaming variable for output
vtkfile_Phi[0] << (temp_output_variable, t)  # Writting the potential in intial time step to the output file

redE.assign(project(1e21*sqrt(dot(-grad(Phi), -grad(Phi)))/N0, solver_type='mumps'))  # Calculating reduced electric field
redE_old.assign(redE)  # Updating reduced electric field in previous time step

# Calculating transport and rate coefficients using linear interpolation
Transport_coefficient_interpolation('initial', mobility_dependence, N0, Tgas, mu, mu_x, mu_y, mean_energy, redE, mu)  # Mobilities interpolation
Transport_coefficient_interpolation('initial', Diffusion_dependence, N0, Tgas, D, D_x, D_y, mean_energy, redE, mu)  # Diffusion coefficients interpolation
Rate_coefficient_interpolation('initial', k_dependence, rate_coefficient, k_x, k_y, mean_energy, redE, Te = 0, Tgas = 0)  # Rates coefficients interpolation

if semi_implicit == True:
    rate_coefficient_si = semi_implicit_coefficients(k_dependence, mean_energy_e, mean_energy_old, rate_coefficient, rate_coefficient_diff)
    mu_si = semi_implicit_coefficients(mobility_dependence, mean_energy_e, mean_energy_old, mu, mu_diff)
    D_si = semi_implicit_coefficients(Diffusion_dependence, mean_energy_e, mean_energy_old, D, D_diff)

    i = 0
    while i < len(k_y):
        if k_dependence[i] == "Umean":
            rate_coefficient_diff[i].vector()[:] = np.interp(mean_energy_old.vector()[:], k_x[i], k_diff[i])
        i += 1
else:
    rate_coefficient_si = rate_coefficient
    mu_si = mu
    D_si = D

mu_diff[number_of_species-1].vector()[:] = np.interp(mean_energy_old.vector()[:], mu_x[number_of_species-1], mue_diff)
D_diff[number_of_species-1].vector()[:] = np.interp(mean_energy_old.vector()[:], D_x[number_of_species-1], De_diff)

# ============================================================================
# Definition of variational formulation for coupled approach
# ============================================================================
Powered_Electrode_bc_C = DirichletBC(ME.sub(number_of_equations-1), Phi_powered, Powered_electrode)  # Setting Dirichlet boundary condition for Poisson equation in coupled approach at the powered electrode
Grounded_bc_C = DirichletBC(ME.sub(number_of_equations-1), Phi_grounded, Grounded_electrode)  # Setting Dirichlet boundary condition for Poisson equation in coupled approach at the grounded electrode
Voltage_bcs_C = [Powered_Electrode_bc_C, Grounded_bc_C]  # Creating the list of Dirichlet boundary conditions for Poisson equation in coupled approach

f_potential_C = rho_poisson_C/epsilon # Poisson equation source term

F_potential_C = weak_form_Poisson_equation(dx, u[number_of_equations-1], v[number_of_equations-1], f_potential_C, r)

Ion_flux = 0   # Sum of ion fluxes, required for secondary electron emission in boundary condition
i = 1
while i < number_of_species:
    Gamma.append(Flux(sign[i], u[i], D_si[i], mu_si[i], E, grad_diffusion = grad_diff[i], logarithm_representation = True))  # Setting up particle fluxes
    if particle_species_type[i] == 'Ion':
        Ion_flux += Max(dot(Gamma[i], normal_plasma), 0)  # Setting up ion fluxes for secondary electron emission in boundary condition
    i += 1
# Defining electron energy flux
Gamma_en = Flux(sign[number_of_species - 1], u[0], 5.0*D_si[number_of_species - 1]/3.0, 5.0*mu_si[number_of_species - 1]/3.0, E, grad_diffusion = grad_diff[number_of_species - 1], logarithm_representation = True)
u_see_met = Expression('u_p', u_p = we_metalic, degree = 1)  # Setting mean energy of secondary electrons

f = Source_term('coupled', approximation, power_matrix, loss_matrix, gain_matrix, rate_coefficient_si, N0, u)  # Particle source term definition
f_en = Energy_Source_term('coupled', power_matrix, loss_matrix, gain_matrix, rate_coefficient_si, energy_loss, u[0]/u[number_of_species-1], N0, u)  # Energy source term definition
f_en += -dot(Flux(sign[number_of_species-1], u[number_of_species-1], D_si[number_of_species-1], mu_si[number_of_species-1], E, grad_diffusion = grad_diff[number_of_species - 1], logarithm_representation = True), E) # Adding power input from the electric field

i = 1
while i < number_of_species:
    F += weak_form_balance_equation_log_representation(equation_type[i], dt, dt_old, dx, u[i], u_old[i], u_old1[i], v[i], f[i], Gamma[i], r, D_si[i])  # Definition of variational formulations of the balance equation for the particle species
    i += 1

# Adding boundary integral term in variational formulation for the particle species
i = 0
while i < number_of_boundaries:
    j = 1
    while j < number_of_species:
        Fb = Boundary_flux('flux source', equation_type[j], particle_type[j], sign[j], mu_si[j], E, normal_plasma, u[j], gamma[i], v[j], ds_plasma(i+1), r, vth[j], ref_coeff[i][j], Ion_flux)  # Setting up boundary conditions for particle balance equations
        F += Fb
        j += 1
    i += 1

# Definition of variational formulation of the electron energy balance equation
F_en = weak_form_balance_equation_log_representation(equation_type[number_of_species-1], dt, dt_old, dx, u[0], u_old[0], u_old1[0], v[0], f_en, Gamma_en, r)  # Setting up variational formulation of electron energy balance equations
# Adding boundary integral term in variational formulation of the electron energy balance equation
i = 0
while i < number_of_boundaries:
    F_en += Boundary_flux('flux source', equation_type[number_of_species-1], particle_type[number_of_species-1], sign[number_of_species-1], 5.0*mu_si[number_of_species-1]/3.0, E, normal_plasma, u[0], gamma[i]*u_see_met, v[0], ds_plasma(i+1), r, 1.3333*vth[number_of_species-1], ref_coeff[i][number_of_species-1], Ion_flux)
    i += 1

F += F_en  # Adding electron energy balance equation variational fromulation
F += F_potential_C  # Adding Poisson equation variational formulation

# ============================================================================
# Defining lists required for assigning values to functions defined on mixed function space and file output
# ============================================================================
variable_list_new = [we_newV, u_newV[1], u_newV[2], u_newV[3], Phi]
variable_list_old = [we_oldV, u_oldV[1], u_oldV[2], u_oldV[3], Phi_old]
variable_list_old1 = [we_old1V, u_old1V[1], u_old1V[2], u_old1V[3], Phi_old1]
output_old_variable_list = [Phi_old, u_oldV[1], u_oldV[2], u_oldV[3]]  # List of variables for file output
output_new_variable_list = [Phi, u_newV[1], u_newV[2], u_newV[3]]  # List of variables for file output
output_files_variable_names = ['Phi', particle_species_file_names[1], particle_species_file_names[2], particle_species_file_names[3]]

rev_assigner.assign(u_new, variable_list_new)  # Assigning values between function spaces in current time step
rev_assigner.assign(u_old, variable_list_old)  # Assigning values between function spaces in previous time step
rev_assigner.assign(u_old1, variable_list_old1)  # Assigning values between function spaces in k-2 time step

F = action(F, u_new)
J = derivative(F, u_new, u) # Defining Jacobian for nonlinear solver

# ============================================================================
# Defining nonlinear problem and setting-up nonlinear solver
# ============================================================================
problem = Problem(J, F, Voltage_bcs_C)

# Setting PETScSNESSolver and its parameters
nonlinear_solver = PETScSNESSolver()
nonlinear_solver.parameters['relative_tolerance'] = relative_tolerance
nonlinear_solver.parameters["linear_solver"] = linear_solver
# nonlinear_solver.parameters["preconditioner"] = "hypre_amg"  # setting the preconditioner, uncomment if iterative solver is used

# ============================================================================
# Time loop
# ============================================================================
no_iterations = [0]*3  # List of the number of newton iterations in current and previous time steps
convergence = 0  # Convergence
while t < T_final:
    t_old = t  # Updating old time step
    u_old1.assign(u_old)
    u_old.assign(u_new)
    assigner.assign(variable_list_old, u_old)
    redE_old.assign(redE)
    mean_energy_old1.assign(mean_energy_old)
    mean_energy_old.assign(mean_energy)

    # ============================================================================
    # Updating electric field and energy dependent rate and transport coefficients
    # ============================================================================
    redE.assign(project(1e21*sqrt(dot(-grad(Phi), -grad(Phi)))/N0, solver_type='mumps'))  # Updating reduced electric field
    Transport_coefficient_interpolation('update', mobility_dependence, N0, Tgas, mu, mu_x, mu_y, mean_energy_old, redE)  # Updating mobilities
    Transport_coefficient_interpolation('update', Diffusion_dependence, N0, Tgas, D, D_x, D_y, mean_energy_old, redE, mu)  # Updating diffusion coefficients
    Rate_coefficient_interpolation('update', k_dependence, rate_coefficient, k_x, k_y, mean_energy_old, redE, Te = 0, Tgas = 0)  # Updating rate coefficients
    i = 0
    while i < len(k_y):
        if k_dependence[i] == "Umean":
            rate_coefficient_diff[i].vector()[:] = np.interp(mean_energy_old.vector()[:], k_x[i], k_diff[i])
        i += 1

    mu_diff[number_of_species-1].vector()[:] = np.interp(mean_energy_old.vector()[:], mu_x[number_of_species-1], mue_diff)
    D_diff[number_of_species-1].vector()[:] = np.interp(mean_energy_old.vector()[:], D_x[number_of_species-1], De_diff)

    # ============================================================================
    # Solving the coupled equation ussing adaptive solver. The calculated values are assigned to the variables used for postprocessing.
    # ============================================================================
    t, _ = adaptive_solver(nonlinear_solver, problem, t, dt, dt_old, u_new, u_old, variable_list_new, variable_list_old, assigner, error, files.error_file, max_error, ttol, dt_min, time_dependent_arguments = [Phi_powered], approximation = approximation)

    log('time', files.model_log, t)  # Time logging

    mean_energy.vector()[:] = np.exp(we_newV.vector()[:] - u_newV[number_of_species-1].vector()[:])  # Mean energy calculations

     # ============================================================================
     # Writting results to the output files using file output. Linear interpolation of solutions is used for a desired output time step.
     # ============================================================================
    t_output, t_output_step = file_output(t, t_old, t_output, t_output_step, t_output_list, t_output_step_list, file_type, output_file_list, output_files_variable_names, output_new_variable_list, output_old_variable_list, unit = 'us')  # File output for desired list of variables. The values are calculated using linear interpolation at desired time steps

     # ============================================================================
     #Updating time steps using adaptive time step function.
     # ============================================================================
    dt_old1.time_step = dt_old.time_step
    dt_old.time_step = dt.time_step
    dt.time_step = adaptive_timestep(dt.time_step, max_error, ttol, dt_min, dt_max)  # Updating time-step size

    if(MPI.rank(MPI.comm_world)==0):
        print(str(dt_old.time_step) + '\t' + str(dt.time_step))

     # Updating maximum error in previous time steps
    max_error[2] = max_error[1]
    max_error[1] = max_error[0]
