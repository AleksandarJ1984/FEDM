"""
In this example the modeling of positive streamer in air, as described in Bagheri et al. Plasma Sources Sci. Technol. 27 (2018) 095002 is performed. The problem consists of
solving Poisson's and balance equations for electrons and ions. The variational formulation of the code is automatized using custom made functions. The coupled i.e.
fully implicit approach, consisting in solving all equations, at once is applied. In this case, rate and transport coefficients are given in a form of an expression, so
all variables are defined in a same time step. The code has an option to automatize definition of rate and transport coefficients by importing them from external files.
Moreover, the source term definition can be done by reading reaction scheme from file.
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

import sys

# my_mesh = Mesh()
# with XDMFFile("mesh.xdmf") as file:
#     file.read(my_mesh)
#plot(mesh)


##########

#Optimization parameters
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["std_out_all_processes"] = False
parameters['krylov_solver']['nonzero_initial_guess'] = True
#parameters["form_compiler"]["quadrature_degree"] = 2

# Defining tye of used solver and its parameters.
linear_solver = "mumps" # Type of linear solver lu | mumps | gmres | bicgstab
maximum_iterations = 2
relative_tolerance = 1e-4

# ============================================================================
# Definition of the simulation conditions, model, approximation, coordinates and path for input files
# ============================================================================
model = 'benchmark_model'
coordinates = 'cylindrical'
gas = 'Air'
Tgas = 300.0 #[K]
p0 = 760.0 #[Torr]
N0 = p0*3.21877e22 #[m^-3]
U_w  = 3000  #[V]
approximation = 'LFA' # Type of approximation used in the model
path = files.file_input / model # Path where input files for desired model are stored
files.output_folder_path = "./output_Galerkin" # Rename output dir from "output" to "output_Galerkin"

# ============================================================================
# Reading species list and particle properties, obtaining number of species for which the problem is solved and creating
#  output files.
# ============================================================================
number_of_species, particle_species, particle_prop, particle_species_file_names = read_speclist(path) # Reading species list to obtain number of species, their name and corresponding file names
M, sign = read_particle_properties(particle_prop, model)  # Reading particle properties from input files
equation_type = ['reaction', 'drift-diffusion-reaction'] # Defining the type of the equation (reaction | diffusion-reaction | drift-diffusion-reaction)
particle_species_type = ['Ions', 'electrons'] # Defining particle type required for boundary condition: Neutral | Ion | electrons

# Setting up number of equations for given approximation
number_of_species, number_of_equations, particle_species, M, sign = modify_approximation_vars(approximation, number_of_species, particle_species, M, sign)
charge = [i * elementary_charge for i in sign]

vtkfile_u = output_files('pvd', 'number density', particle_species_type) # Creates list of output files
vtkfile_Phi = output_files('pvd', 'potential', ['Phi']) # Creates list of output files
output_file_list = [vtkfile_Phi[0], vtkfile_u[0], vtkfile_u[1]] # Creates list of variables for output
file_type = ['pvd', 'pvd', 'pvd'] # creates list of variables for output

# ============================================================================
# Definition of the time variables and relative error used for adaptive time stepping.
# ============================================================================
t_old = None # Previous time step
t0 =  0.0 # Initial time step
t = t0 # Current time step
T_final = 6e-9 # Simulation time end [s]

dt_min = 1e-19 # Minimum time step [s]
dt_max = 5e-10 # Maximum time step [s]
dt_init = 5e-12 # Initial time step size [s]
dt_old_init = 1e30 # Initial time step size [s] setted up as extremely large value to initiate adaptive BDF2
dt = Expression("time_step", time_step = dt_init, degree = 0) # Time step size [s]
dt_old = Expression("time_step", time_step = dt_old_init, degree = 0) # Time step size expression [s], Initial value is set up to be large in order to reduce initial step of adaptive BDF formula to one.

ttol = 1e-3 # Tolerance for adaptive time stepping

### Setting-up output times and time steps. t_output_list and t_output_step_list need to have same length
t_output_list = [1e-12, 1e-11, 1e-10, 1e-9] # List of time step intervals (consisting of two consecutive components) at which the results are printed to file
t_output_step_list = [1e-12, 1e-11, 1e-10, 1e-9] # List of time step sizes for corresponding interval
t_output_step = t_output_list[0] # Current time step at which the results are printed to file
t_output = t_output_step_list[0] # Current output time length

number_of_iterations = None # Nonlinear solver iteration number counter
convergence = None # Nonlinear solver convergence values True or False
error = [0.0]*number_of_species # List of error values for particle species
max_error = [1]*3 # List of maximum error values in current, k-1 and k-2 time steps

# ============================================================================
# Defining the geometry of the problem and corresponding boundaries.
# ============================================================================
if coordinates == 'cylindrical':
    r = Expression('x[0]', degree = 1)
    z = Expression('x[1]', degree = 1)

box_width = 0.0011  #[m]
box_height = 0.002 #[m]
boundaries = [['line', 0.0, 0.0, 0.0, box_width],
              ['line', box_height, box_height, 0.0, box_width],
              ['line', 0.0, box_height, 0.0, 0.0],
              ['line', 0.0, box_height, box_width, box_width]]
number_of_boundaries = len(boundaries)
bc_type_grounded = ['zero flux', 'Neumann']
bc_type_powered =  ['zero flux', 'Neumann']
bc_type_axis = ['zero flux', 'zero flux']
bc_type_wall = ['zero flux', 'zero flux']
bc_type = [bc_type_grounded, bc_type_powered, bc_type_axis, bc_type_wall] # Boundary conditions for given particle
gamma = [0.0, 0.0] # Secondary electron emission coefficient

log('conditions', files.model_log, dt.time_step, U_w, p0, box_height, N0, Tgas) # Writting simulation conditions to log file
log('properties', files.model_log, gas, model, particle_species_file_names, M, charge) # Writting particle properties into a log file

# ===========================================================================================================================
# Mesh setup and boundary measure redefinition. Structured mesh is generated using built-in mesh generator
# ===========================================================================================================================
#mesh = RectangleMesh(Point(0, 0), Point(box_width, box_height), 500, 500, "crossed")
mesh = Mesh('mesh_compare.xml') # Importing mesh from xml file


#mesh_statistics(mesh) # Prints number of elements, minimum and maximal cell diameter
boundary_mesh_function = Marking_boundaries(mesh, boundaries) # Marking boundaries required for boundary conditions
normal = FacetNormal(mesh) # Boundary normal

File(str(files.output_folder_path / 'mesh' / 'boundary_mesh_function.pvd')) << boundary_mesh_function # Writting boundary mesh function to file

dx = Measure('dx', domain = mesh)
ds = Measure('ds', domain = mesh, subdomain_data = boundary_mesh_function)              #boundary measure redefinition


log('initial time', files.model_log, t) # Time logging

# ============================================================================
# Defining type of elements and function space, test functions, trial functions and functions for storing variables.
# ============================================================================
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # Defining finite element and degree
Element_list = Mixed_element_list(number_of_equations, P1) # Creating list of elements
ME = FunctionSpace(mesh, MixedElement(Element_list)) # Defining mixed function space
V = FunctionSpace(mesh, P1) # Function space for post-processing
W = VectorFunctionSpace(mesh, 'P', 1) # Defining vector function space

assigner = FunctionAssigner(Function_space_list(number_of_equations, V), ME) # Assigning values of variables between ME and V function spaces
rev_assigner = FunctionAssigner(ME, Function_space_list(number_of_equations, V)) # Assigning values of variables between V and ME function spaces

temp_output_variable = Function(V)  # Temporary variable used for file output

# Defining variables for coupled approach
u = TrialFunction(ME) # Trial functions for mixed formulation
v = TestFunctions(ME) # Test functions for mixed formulation
vsupg = TestFunction(V) # Test functions for stab
u_new = Function(ME) # Functions for mixed formulation for current time step
u_old  = Function(ME) # Functions for mixed formulation for k-1 time step
u_old1  = Function(ME) # Functions for mixed formulation for k-2 time step

# Defining variables for solving initial Poisson's equation
PhiV = TrialFunction(V) # Trial functions for potentialls

vp = TestFunction(V) # Test functions for potentialc
Phi = Function(V) # Potential  in current time step
Phi_old = Function(V) # Potential in k-1 time step

#Defining variables for postprocessing
u_newV = Function_definition(V, 'Function', number_of_equations) # Number density in previous time step (Function return functions for desired number of particles)
u_oldV = Function_definition(V, 'Function', number_of_equations) # Number density in previous time step (Function return functions for desired number of particles)
u_old1V = Function_definition(V, 'Function', number_of_equations) # Number density in current time step (Function return functions for desired number of particles)
mu = Function_definition(V, 'Function', number_of_equations) # Defining variable for mobility
D = Function_definition(V, 'Function', number_of_equations) # Defining variable for diffusion coefficient
Gamma = Function_definition(V, 'Function', number_of_equations) # Defining list of variables for particle flux
f = Function_definition(V, 'Function', number_of_equations) # Defining list of variables for source term

tauwgradv = Function_definition(V, 'Function', number_of_equations) # Defining variable for stabilization

# ============================================================================
# Setting up initial conditions
# ============================================================================
l = 1.0e-4
x1 = 1.6e-3
u0 = 1e13
u1 = 5.0e19

box_width = 0.0011  #[m]
box_height = 0.002 #[m]

u_oldV[0] = interpolate(Expression('u0 + u1 * exp(-(pow(x[0], 2)+pow(x[1]-x1, 2))/pow(l, 2))',u0=u0, u1=u1, x1=x1, l=l, degree = 1), V)
u_oldV[1] = interpolate(Expression('u0', u0=u0, degree = 1), V)

u_newV[0] = interpolate(Expression('u0 + u1 * exp(-(pow(x[0], 2)+pow(x[1]-x1, 2))/pow(l, 2))',u0=u0, u1=u1, x1=x1, l=l, degree = 1), V)
u_newV[1] = interpolate(Expression('u0', u0=u0, degree = 1), V)

# Writing initial values to file
i = 0
while i < number_of_species:
    temp_output_variable.assign(u_oldV[i]) # Setting value for output
    temp_output_variable.rename(particle_species_file_names[i+1], str(i+1))  # Renaming variable for output
    vtkfile_u[i] << (temp_output_variable, t) # Writting initial particle number densities to file
    i += 1

# ===========================================================================================================================
# Solving initial Poisson equation to calculate the potential for the initial time step.
# The potential is required for interpolation of transport coefficients that depend on reduced electric field.
# ===========================================================================================================================
Phi_cathode = Constant(0.0) # Potential at the grounded electrode in [V]
Phi_anode = Constant(U_w) # Potential at the powered electrode in [V]

def Cathode(x, on_boundary):
    if near(x[1], 0) and on_boundary:
        return True

def Anode(x, on_boundary):
    if near(x[1], box_height) and on_boundary:
        return True

# Defining Dirichlet boundary conditions
potential_Cathode_bc = DirichletBC(V, Phi_cathode, Cathode)
potential_Anode_bc = DirichletBC(V, Phi_anode, Anode)
bcs_potential = [potential_Cathode_bc, potential_Anode_bc] # List of Dirichlet boundary conditions

potential_f = ((u_oldV[0]) - (u_oldV[1])) * elementary_charge/epsilon_0 # Poisson equation source term

# Definition variational formulation of Poisson equation for the initial time step
Fp = weak_form_Poisson_equation(dx, PhiV, vp, potential_f, r)
a, L = lhs(Fp), rhs(Fp)

# Creating lhs and rhs and setting boundary conditions
potential_A = assemble(a)
[bc.apply(potential_A) for bc in bcs_potential]
potential_b = assemble(L)
[bc.apply(potential_b) for bc in bcs_potential]

# Solving Poisson's equation for the initial time step
solve(potential_A, Phi.vector(), potential_b)

temp_output_variable.assign(Phi) # Setting value for output
vtkfile_Phi[0] << (temp_output_variable, t) # Writting the potential in intial time step to the output file

E = -grad(u[2]) # Setting up electric field
E_m = sqrt(inner(-grad(u[2]), -grad(u[2]))) # Setting up electric field magnitude

# Updating initail values
u_oldV[2].assign(Phi) # Updating initial value of potential
u_newV[2].assign(Phi) # Updating potential value

D_x, D_y, Diffusion_dependence = read_transport_coefficients(particle_species, 'Diffusion', model) # Reading diffusion coefficients from input files
mu_x, mu_y, mu_dependence = read_transport_coefficients(particle_species, 'mobility', model) # Reading mobilities from input files

# ============================================================================
# Definition of variational formulation for coupled approach
# ============================================================================
bc = [DirichletBC(ME.sub(2), Phi_cathode, Cathode), DirichletBC(ME.sub(2), Phi_anode, Anode)]  # Creating of the list of Dirichlet boundary conditions for Poisson equation in coupled approach

mu[0] = mu_y[0] # Setting up mobility for ions
D[0] = D_y[0] # Setting up diffusion coefficient for ions
mu[1] = eval(mu_y[1])  # Setting up mobility for electrons
D[1] = eval(D_y[1]) # Setting up diffusion coefficient for electrons
alpha = (1.1944e6 + 4.3666e26 * E_m**(-3))*exp(-2.73e7/E_m)-340.75 # Setting up ionization coefficient

# h_mesh = 1e-6 # mesh_metric(mesh)
# tauwgradv[0] = 0.0*v[0]
# tauwgradv[1] = sign[1]*(E_m**(-1)*(h_mesh/2)*dot(E, grad(v[1])))

Gamma[0] = 0.0 # Setting up ion flux
Gamma[1] = Flux(sign[1], u[1], D[1], mu[1], E, logarithm_representation=False) # Setting up electron flux

f[0] = alpha*mu[1]*E_m*u[1] # Ion source term definition
f[1] = alpha*mu[1]*E_m*u[1] # Electron source term definition
i = 0
while i < number_of_species:
    f[2] += sign[i] * (u[i])* elementary_charge/epsilon_0 # Poisson's equation source term definition
    i += 1


F = 0.0
# ============================================================================
# Defining variational formulation of the Poisson equation for the inital step
# ============================================================================
i = 0
#theta = 1.0
while i < number_of_species:

    F += weak_form_balance_equation(
        equation_type=equation_type[i], 
        dt=dt, 
        dt_old=dt_old, 
        dx=dx, 
        u=u[i], 
        u_old=u_old[i], 
        u_old1=u_old1[i], 
        v=v[i], 
        f=f[i], 
        Gamma=Gamma[i], 
        r=r, 
        D=D[i], 
        log_representation=False, 
        ) # Definition of variational formulation of the balance equation for the electrons

    # F += weak_form_supg_balance_equation(
    #     equation_type=equation_type[i], 
    #     dt=dt, 
    #     dt_old=dt_old, 
    #     dx=dx, 
    #     u=u[i], 
    #     u_old=u_old[i], 
    #     u_old1=u_old1[i], 
    #     tauwgradv=tauwgradv[i], 
    #     f=f[i], 
    #     Gamma=Gamma[i], 
    #     r=r, 
    #     D=D[i],
    #     ) # Setting up variational formulation of electron energy balance equations
    i += 1

F += weak_form_Poisson_equation(dx, u[number_of_equations - 1], v[number_of_equations - 1], f[number_of_equations - 1], r) # Adding Poisson equation variational formulation
# ===========================================================================================================================
# Setting Neumann boundary condition on all boundaries. For sepperate boundaries use ds(i), where i is boundary index.
# The index for each boundary can be seen by plotting boundary mesh function in paraview
# ===========================================================================================================================
i = 0
while i < number_of_boundaries:
    j = 0
    while j < number_of_species:
        F += Boundary_flux(bc_type[i][j], equation_type[j], particle_species_type[j], sign[j], mu[j], E, normal, u[j], gamma[j], v[j], ds(i+1), r, log_representation=False)
        j += 1
    i += 1

# ============================================================================
# Defining list required for value assigning between function spaces and file output
# ============================================================================
variable_list_new = [u_newV[0], u_newV[1], u_newV[2]] #List of variables for easier assigning between function spaces
variable_list_old = [u_oldV[0], u_oldV[1], u_oldV[2]] # List of variables for easier assigning between function spaces
output_old_variable_list = [u_oldV[2], u_oldV[0], u_oldV[1]] # List of variables for file output
output_new_variable_list = [u_newV[2], u_newV[0], u_newV[1]] # List of variables for file output
output_files_variabe_names = ['Phi', particle_species_type[0], particle_species_type[1]]

rev_assigner.assign(u_old, variable_list_old) # Assigning values between function spaces in previous time step
rev_assigner.assign(u_new, variable_list_new) # Assigning values between function spaces in current time step

# ============================================================================
# Defining nonlinear problem and setting up solver
# ============================================================================
F = action(F, u_new)
J = derivative(F, u_new, u)

problem = Problem(J, F, bc)

## Setting PETScSNESSolver and its parameters
nonlinear_solver = PETScSNESSolver()
nonlinear_solver.parameters['relative_tolerance'] = relative_tolerance
nonlinear_solver.parameters["linear_solver"]= linear_solver
nonlinear_solver.parameters['maximum_iterations'] = maximum_iterations
if linear_solver == 'gmrs':
    nonlinear_solver.parameters["preconditioner"] = "hypre_amg" # setting the preconditioner, uncomment if iterative solver is used

ite_tot = 0
output_ite_file = str(files.output_folder_path)
output_ite_file += "/iteration.csv"
with open(output_ite_file,"w") as f:
    f.write("iteration, dt\n")


# ============================================================================
# Time loop
# ============================================================================
while abs(t-T_final)/T_final > 1e-6:
    t_old = t # Updating old time step
    u_old1.assign(u_old)
    u_old.assign(u_new)
    assigner.assign(variable_list_old, u_old)

    ## Solving problem with adaptive time step
    t, ite = adaptive_solver(nonlinear_solver, problem, t, dt, dt_old, u_new, u_old, variable_list_new, variable_list_old, assigner, error, files.error_file, max_error, ttol, dt_min, time_dependent_arguments = [], approximation = approximation)

    ite_tot += ite

    ite_tot = 0

    with open(output_ite_file,"a") as f:
        f.write("%s, %s\n" % (ite_tot, dt.time_step))

    ## For the constant time step, comment previous and  uncomment following code block
    # t += dt.time_step
    # try_except = True
    #
    # assigner.assign(var_list_new, u_new)
    # with open(files.error_file, "a") as f_err:
    #     i = 0
    #     while i < len(var_list_new) - 1:
    #         # temp1 = project(exp(var_list_new[i]), solver_type = 'gmres')
    #         # temp0 = project(exp(var_list_old[i]), solver_type = 'gmres')
    #         # error[i] = l2_norm(t, dt.time_step, temp1, temp0)
    #         error[i] = l2_norm(t, dt.time_step, var_list_new[i], var_list_old[i])
    #         f_err.write("{:<23}".format(str(error[i])) + '  ')
    #         i += 1
    #     f_err.write("{:<23}".format(str(dt_old.time_step)) + '  ' + "{:<23}".format(str(dt.time_step)) + '\n')
    #     f_err.flush()
    # max_error[0] = max(error)

    log('time', files.model_log, t) # Time logging

# ============================================================================
# Time step refinement
# ============================================================================
    dt_old.time_step = dt.time_step # Constant time step
    dt.time_step = adaptive_timestep(dt.time_step, max_error, ttol, dt_min, dt_max) # Adaptive time step

    # Updating maximum error in previous time steps
    max_error[2] = max_error[1]
    max_error[1] = max_error[0]

    # ============================================================================
    # Writting results to the files using file output. Linear interpolation of solutions is used for a desired output time step.
    # ============================================================================
    t_output, t_output_step = file_output(t, t_old, t_output, t_output_step, t_output_list, t_output_step_list, file_type, output_file_list, output_files_variabe_names, output_new_variable_list, output_old_variable_list) # File output for desired list of variables. The values are calculated using linear interpolation at desired time steps
