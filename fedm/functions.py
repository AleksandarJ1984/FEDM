# Fluid modelling functions module

import warnings
from typing import List, Tuple, Any, Optional, Union
from pathlib import Path
from textwrap import dedent

import dolfin as df
import numpy as np

from .physical_constants import elementary_charge, kB, kB_eV
from .utils import print_rank_0, comma_separated


def modify_approximation_vars(
    approximation_type: str,
    number_of_species: int,
    particle_species: List[str],
    masses: List[float],
    charges: List[float],
) -> Tuple[int, int, List[str], List[float], List[float]]:
    """
    Depending on approximation used, the number of equations, charge and mass variables
    are modified. Returns number species, number of equations, particle species,
    masses, and charges. particle_species, masses, and charges may be modified.
    """
    approximation_types = ["LFA", "LMEA"]
    if approximation_type not in approximation_types:
        err_msg = dedent(
            f"""\
            fedm.modify_approximation_vars: The approximation type {approximation_type}
            is not recognised. Must be one of {comma_separated(approximation_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    # IF LFA, remove the first species from each list
    if approximation_type == "LFA":
        number_of_species -= 1
        particle_species.pop(0)
        masses.pop(0)
        charges.pop(0)
    # Number of equations should be 1 more than the number of species in each case
    number_of_eq = number_of_species + 1
    return number_of_species, number_of_eq, particle_species, masses, charges


class CircleSubDomain(df.SubDomain):
    def __init__(
        self,
        center_z: float,
        center_r: float,
        radius: float,
        gap_length: float,
        submesh: bool = False,
        tol: float = 1e-8,
    ):
        super().__init__()
        self._center_z = float(center_z)
        self._center_r = float(center_r)
        self._radius = float(radius)
        self._submesh = bool(submesh)
        self._tol = float(tol)

    def inside(self, x: List[float], on_boundary: bool) -> bool:
        r, z = x[0], x[1]
        dist_from_center_squared = (r - self._center_r) ** 2 + (z - self._center_z) ** 2
        within_tol = abs(dist_from_center_squared - self._radius**2) <= self._tol
        z_correct = z <= 0 if self._center_z <= 0 else z >= self._gap_length
        return within_tol and z_correct and (on_boundary or self._submesh)


class LineSubDomain(df.SubDomain):
    def __init__(self, r_range: Tuple[float, float], z_range: Tuple[float, float]):
        super().__init__()
        self._r_range = r_range
        self._z_range = z_range

    def inside(self, x: List[float], on_boundary: bool) -> bool:
        r, z = x[0], x[1]
        in_r_range = df.between(r, self._r_range)
        in_z_range = df.between(z, self._z_range)
        return in_r_range and in_z_range and on_boundary
    
class PointSubDomain(df.SubDomain):
    def __init__(self, z_range: Tuple[float, float]):
        super().__init__()
        self._z_range = z_range

    def inside(self, x: List[float], on_boundary: bool) -> bool:
        z = x[0]
        in_z_range = df.between(z, self._z_range)
        return in_z_range and on_boundary


def Marking_boundaries(
    mesh: df.Mesh,
    boundaries: List[List[Any]],
    submesh: bool = False,
    gap_length: float = 0.01,
) -> df.MeshFunction:
    """
    Marking boundaries of a provided mesh. Currently, straight-line and circular
    boundaries are supported. First argument is the mesh, the second argument is a list
    of boundary properties (boundary type and coordinates).
    """

    boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    for idx, boundary in enumerate(boundaries):
        boundary_type = boundary[0]

        print_rank_0(boundary_type)

        if boundary_type == "circle":
            center_z, center_r, radius = boundary[1:4]
            bmark = CircleSubDomain(center_z, center_r, radius, gap_length, submesh)
        elif boundary_type == "line":
            eps = df.DOLFIN_EPS
            z1, z2 = boundary[1] - eps, boundary[2] + eps
            r1, r2 = boundary[3] - eps, boundary[4] + eps
            bmark = LineSubDomain((r1, r2), (z1, z2))
        elif boundary_type == "point":
            eps = df.DOLFIN_EPS
            z1, z2 = boundary[1] - eps, boundary[2] + eps
            bmark = PointSubDomain((z1, z2))
        else:
            err_msg = dedent(
                f"""\
                fedm.Marking_boundaries: Invalid boundary_type '{boundary_type}'.
                Possible values are 'circle', 'line'.
                """
            )
            raise ValueError(err_msg.rstrip().replace("\n", " "))

        bmark.mark(boundary_markers, idx + 1)

    return boundary_markers


def Mixed_element_list(
    number_of_equations: int, element: df.FiniteElement
) -> List[df.FiniteElement]:
    """
    Defines list of mixed elements. Input arguments are number of equations and element
    type.
    """
    # WARNING: All elements of the list refer to the same object!
    return [element] * number_of_equations


def Function_space_list(
    number_of_equations: int, function_space: df.FunctionSpace
) -> List[df.FunctionSpace]:
    """
    Defines list of function spaces. Input arguments are number of equations and
    function space.
    """
    # WARNING: All elements of the list refer to the same object!
    return [function_space] * number_of_equations


def Function_definition(
    function_space: df.FunctionSpace, function_type: str, eq_number: int = 1
) -> List[Any]:
    """
    Defines list of desired function type (TrialFunction, TestFunction or ordinary
    Function). Input arguments are function space, type of desired function and number
    of equations, where the default value is one.
    """
    functions = {
        "TrialFunction": df.TrialFunction,
        "TestFunction": df.TestFunction,
        "Function": df.Function,
    }
    if function_type not in functions:
        err_msg = dedent(
            f"""\
            fedm.Function_definition: Invalid function_type '{function_type}'.
            Possible values are {comma_separated(functions)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))
    function = functions[function_type]
    return [function(function_space) for _ in range(eq_number)]


class Problem(df.NonlinearProblem):
    """
    Nonlinear problem definition. Input parameters are F - weak form of the equation
    J - Jacobian and bcs -Dirichlet boundary conditions. The part of code is provided
    by user Nate Sime from FEniCS forum:
    https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/3
    """

    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        df.NonlinearProblem.__init__(self)

    def F(self, b, x):
        """
        Linear form assembly
        """
        df.assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        """
        Bilinear form assembly
        """
        df.assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


def Max(a, b):
    """
    Returns maximum value of a and b.
    """
    return (a + b + abs(a - b)) / df.Constant(2.0)


def Min(a, b):
    """
    Returns minimum value of a and b.
    """
    return (a + b - abs(a - b)) / df.Constant(2.0)
    

def Flux(sign, u, D, mu, E, grad_diffusion = True, logarithm_representation = True):
    """
    Defines particle flux using drift-diffusion approximation.
    Input arguments are particle charge, number density,
    diffusion coefficient, mobility and electric field. The additional
    arguments are used to specify if the gradient of the diffusion 
    should be considered and whether logarithmic representation 
    is used.
    """
    if logarithm_representation == True:
        u_e = df.exp(u)
    else:
        u_e = u
    Drift_component = sign * mu * E * u_e
    if grad_diffusion == True:
        Diffusion_component = - df.grad(D * u_e)
    else:
        Diffusion_component = - D * df.grad(u_e)
    return Diffusion_component + Drift_component
    

def weak_form_balance_equation(
    equation_type: str,
    dt: df.Expression,
    dt_old: df.Expression,
    dx: df.Measure,  # Can use built-in dolfin.dx here
    u: Any,  # obtain by indexing result of df.TrialFunction
    u_old: Any,  # obtain by indexing result of df.Function
    u_old1: Any,  # obtain by indexing result of df.Function
    v: Any,  # obtain by indexing result of df.TestFunctions
    f: df.Function,  # obtain by indexing result of df.Function_definition
    Gamma: df.Function,  # obtain by indexing result of df.Function_definition
    r: float = 0.5 / df.pi,
    D: Optional[df.Function] = None,  # get by indexing result of df.Function_definition
    log_representation: bool = False,
    Na: Optional[float] = 0.,  
    psi: Optional[float] = 0.01,  
) -> df.Form:
    """
    Returns the weak form of the particle balance equations.

    If log_representation is True, solves:

    2.0 * pi * exp(u) * (
        (
            (1.0 + 2.0 * dt / dt_old) / (1.0 + dt / dt_old)
        )*(
            u - u_old * pow(1.0 + dt / dt_old, 2.0) / (1.0 + 2.0 * dt / dt_old)
              + u_old1 * pow(dt / dt_old, 2.0) / (1.0 + 2.0 * dt / dt_old)
        )
    ) * (v/dt) * r * dx

    Otherwise, solves:

    2.0 * pi * (
        (
            (1.0 + 2.0 * dt / dt_old) / (1.0 + dt / dt_old)
        )*(
            u - u_old * pow(1.0 + dt / dt_old, 2.0) / (1.0 + 2.0 * dt / dt_old)
              + u_old1 * pow(dt / dt_old, 2.0) / (1.0 + 2.0 * dt / dt_old)
        )
    ) * (v/dt) * r * dx

    If solving diffusion-reaction equation, also includes the term:

    -2.0 * pi * dot(-grad(D * exp(u)), grad(v)) * r * dx     (log representation)
    -2.0 * pi * dot(-grad(D * u), grad(v)) * r * dx          (standard)

    If solving drift-diffusion-reaction, instead includes the term:

    -2.0 * pi * dot(Gamma, grad(v)) * r * dx

    In all cases, also includes the source term:

    - 2.0 * pi * f * v * r * dx

    Parameters
    ----------
    equation_type : str
        Type of equation to solve. Options are 'reaction', 'diffusion-reaction', or
        'drift-diffusion-reaction'.
    dt : df.Expression
        Current time-step size
    dt_old : df.Expression
        Previous time-step size
    dx : df.Measure
        dV, used to build integrals. Recommended to use dolfin.dx.
    u
        Trial function
    u_old
        Value of variable in current time step
    u_old1
        Value of variable in previous time step
    v
        Test function
    f : df.Function
        Source term
    Gamma : df.Function
        particle flux
    r : float, default 0.5 / pi
        r coordinate
    D : df.Function, default None
        diffusion coefficient, only required for the diffusion equation.
    log_representation : bool, default False
        Use logarithmic representation.
    Na: float, default 0.
        First log stabilisation source term (useful when f = 0)
    psi: float, default 0.01
        Second log stabilisation source term (useful when f = 0) 

    Returns
    -------
    df.Form

    Raises
    ------
    ValueError
        If equation_type is not recognised, or if D is not supplied when solving the
        diffusion-reaction equation.
    """
    equation_types = ["reaction", "diffusion-reaction", "drift-diffusion-reaction"]
    if equation_type not in equation_types:
        err_msg = dedent(
            f"""\
            fedm.weak_form_balance_equation_log_representation: The equation type
            {equation_type}' is not recognised. Must be one of
            {comma_separated(equation_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    if equation_type == "diffusion-reaction" and D is None:
        raise ValueError(
            "fedm.weak_form_balance_equation_log_representation: When 'equation_type' "
            "is diffusion-reaction, must also supply the diffusion coefficient 'D'."
        )
    
    tr = dt / dt_old
    trp1 = 1.0 + tr
    tr2p1 = 1.0 + 2.0 * tr
    # If logarithmic, we include a factor of exp(u) in the integral
    expu_or_1 = df.exp(u) if log_representation else 1.0
    # Standard part
    u_part = (u * tr2p1 - trp1**2.0 * u_old + tr**2.0 * u_old1) / trp1
    time_derivative = 2.0 * df.pi * (expu_or_1 * u_part * v / dt) * r * dx
    # Source terms
    source = 2.0 * df.pi * v * f * r * dx
    # Diffusion terms
    diffusion = 0.0
    if equation_type == "diffusion-reaction":
        expu_or_u = df.exp(u) if log_representation else u
        diffusion = 2.0 * df.pi * df.dot(-df.grad(D * expu_or_u), df.grad(v)) * r * dx
    if equation_type == "drift-diffusion-reaction":
        diffusion = 2.0 * df.pi * df.dot(Gamma, df.grad(v)) * r * dx

    # Source terms
    source = 2.0 * df.pi * v * f * r * dx

    # Logarithmic optional stabilisation term (prevent u -> 0 )
    if log_representation & (Na > 0.):
        source += 2.0 * df.pi * Na * df.exp(-psi *  u) * v * r * dx

    # Return with integral bits
    return time_derivative - diffusion - source

    
def weak_form_supg_balance_equation(
    equation_type: str,
    dt: df.Expression,
    dt_old: df.Expression,
    dx: df.Measure,  # Can use built-in dolfin.dx here
    u: Any,  # obtain by indexing result of df.TrialFunction
    u_old: Any,  # obtain by indexing result of df.Function
    u_old1: Any,  # obtain by indexing result of df.Function
    tauwgradv: Any,  # obtain by indexing result of df.TestFunctions
    f: df.Function,  # obtain by indexing result of df.Function_definition
    Gamma: df.Function,  # obtain by indexing result of df.Function_definition
    r: float = 0.5 / df.pi,
    D: Optional[df.Function] = None,  # get by indexing result of df.Function_definition
) -> df.Form:
    """
    Returns the SUPG stabilisation term form of the particle balance equations.

    2.0 * pi * (
        (
            (1.0 + 2.0 * dt / dt_old) / (1.0 + dt / dt_old)
        )*(
            u - u_old * pow(1.0 + dt / dt_old, 2.0) / (1.0 + 2.0 * dt / dt_old)
              + u_old1 * pow(dt / dt_old, 2.0) / (1.0 + 2.0 * dt / dt_old)
        )
    ) * (tauwgradv/dt) * r * dx

    If solving diffusion-reaction equation, also includes the term:

    -2.0 * pi * dot(-grad(D * u), grad(tauwgradv)) * r * dx          

    If solving drift-diffusion-reaction, instead includes the term:

    -2.0 * pi * dot(Gamma, grad(tauwgradv)) * r * dx

    In all cases, also includes the source term:

    - 2.0 * pi * f * tauwgradv * r * dx

    Parameters
    ----------
    equation_type : str
        Type of equation to solve. Options are 'reaction', 'diffusion-reaction', or
        'drift-diffusion-reaction'.
    dt : df.Expression
        Current time-step size
    dt_old : df.Expression
        Previous time-step size
    dx : df.Measure
        dV, used to build integrals. Recommended to use dolfin.dx.
    u
        Trial function
    u_old
        Value of variable in current time step
    u_old1
        Value of variable in previous time step
    tauwgradv
        SUPG test function
    f : df.Function
        Source term
    Gamma : df.Function
        particle flux
    r : float, default 0.5 / pi
        r coordinate
    D : df.Function, default None
        diffusion coefficient, only required for the diffusion equation.
    log_representation : bool, default False
        Use logarithmic representation.

    Returns
    -------
    df.Form

    Raises
    ------
    ValueError
        If equation_type is not recognised, or if D is not supplied when solving the
        diffusion-reaction equation.
    """
    equation_types = ["reaction", "diffusion-reaction", "drift-diffusion-reaction"]
    if equation_type not in equation_types:
        err_msg = dedent(
            f"""\
            fedm.weak_form_balance_equation_log_representation: The equation type
            {equation_type}' is not recognised. Must be one of
            {comma_separated(equation_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    if equation_type == "diffusion-reaction" and D is None:
        raise ValueError(
            "fedm.weak_form_balance_equation_log_representation: When 'equation_type' "
            "is diffusion-reaction, must also supply the diffusion coefficient 'D'."
        )

    # tr = timestep_ratio
    tr = dt / dt_old
    trp1 = 1.0 + tr
    tr2p1 = 1.0 + 2.0 * tr
    # Standard part
    u_part = (u * tr2p1 - trp1**2.0 * u_old + tr**2.0 * u_old1) / trp1

    # Diffusion terms
    diffusion = 0.0

    if equation_type == "diffusion-reaction":
        diffusion = 2.0 * df.pi * df.div(df.grad(D * u)) * tauwgradv * r * dx
    if equation_type == "drift-diffusion-reaction":
        diffusion = - 2.0 * df.pi * df.div(Gamma) * tauwgradv * r * dx
    
    time_derivative = 2.0 * df.pi * (u_part * tauwgradv / dt) * r * dx

    # Source terms
    source = 2.0 * df.pi * tauwgradv * f * r * dx

    # Return with integral bits
    return time_derivative - diffusion - source


def weak_form_balance_equation_log_representation(*args, **kwargs) -> df.Form:
    """
    Convenience function, calls weak_form_balance_equation with log_representation set
    to True.
    """
    return weak_form_balance_equation(*args, **kwargs, log_representation=True)


def weak_form_Poisson_equation(dx, u, v, f, r=0.5 / df.pi) -> df.Form:
    """
    Returns a weak form of Poisson equation. Input arguments are dV, trial function,
    test function, source term and r coordinate.

    Parameters
    ----------
    dx : df.Measure
        dV, used to build integrals. Recommended to use dolfin.dx.
    u
        Trial function
    v
        Test function
    f
        Source term
    r
        r coordinate

    Returns
    -------
    df.Form
    """
    return 2.0 * df.pi * r * (df.inner(df.grad(u), df.grad(v)) - f * v) * dx


def Boundary_flux(
    bc_type: str,
    equation_type: str,
    particle_type: str,
    sign: float,
    mu: Any,
    E: Any,
    normal: Any,
    u: Any,
    gamma: Any,
    v: Any,
    ds_temp: df.Measure,
    r: float = 0.5 / df.pi,
    vth: float = 0.0,
    ref: float = 1.0,
    Ion_flux: float = 0.0,
    log_representation: bool = True,
):
    """
    Function defines boundary conditions for different equations.

    Parameters
    ----------
        Type of boundary condition. Options are 'zero flux', 'flux source' or 'Neumann'
    equation_type : str
        Choices are 'reaction', 'diffusion-reaction' or 'drift-diffusion-reaction'
    particle_type : str
        Choices are 'Heavy' or 'electrons'.
    sign: float
        Particle charge sign
    mu: df.Function
        Mobility
    E
        Electric field, Dolfin ComponentTensor
    normal
        Dolfin FacetNormal
    u
        Trial function
    gamma
        Secondary electron emission coefficient
    v
        Test function
    ds_temp : df.Measure
        ds, surface element used to build integrals (??)
    r: float, default 0.5/pi
        r coordinate
    vth: float, default 0.0
        Thermal velocity
    ref: float, default 1.0
        Reflection coefficient for specified particles species and boundary.
    Ion_flux: float, default 0.0
        Flux of ions
    log_representation : bool, default False
        Use logarithmic representation.

    Returns
    -------
    df.Form
        If bc_type is 'flux source' and there is a diffusion term in equation_type.
        Also if the user combines Neumann boundaries with a drift-diffusion-reaction
        equation.
    float
        Otherwise.

    Raises
    ------
    ValueError
        If bc_type is not recognised. If bc_type is not 'zero flux', also raised if
        equation_type is not recognised. Furthermore, if equation_type is
        'drift-diffusion-reaction', raises if particle_type is not recognised.
    """
    bc_types = ["zero flux", "flux source", "Neumann"]
    equation_types = ["reaction", "diffusion-reaction", "drift-diffusion-reaction"]
    particle_types = ["Heavy", "electrons"]

    expu_or_u = df.exp(u) if log_representation else u

    # If provided bc_type has an underscore instead of a space, correct it
    if "_" in bc_type:
        warnings.warn("fedm.BoundaryFlux: bc_type should have spaces, not underscores")
        bc_type = bc_type.replace("_", " ")

    if bc_type not in bc_types:
        err_msg = dedent(
            f"""\
            fedm.Boundary_flux: boundary condition type '{bc_type}' not recognised.
            Must be one of {comma_separated(bc_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    # Only raise error on bad 'equation_type' if the value is needed
    if bc_type != "zero flux" and equation_type not in equation_types:
        err_msg = dedent(
            f"""\
            fedm.Boundary_flux: equation type '{equation_type}' not recognised.
            Must be one of {comma_separated(equation_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    # Only raise error on bad 'particle_type' if the value is needed
    if (
        bc_type == "flux source"
        and equation_type == "diffusion-reaction"
        and particle_type not in particle_types
    ):
        err_msg = dedent(
            f"""\
            fedm.Boundary_flux: particle type '{particle_type}' not recognised.
            Must be one of {comma_separated(particle_types)}.
            """
        )
        raise ValueError(err_msg.rstrip())

    if bc_type == "flux source" and equation_type != "reaction":
        result = (1.0 - ref) / (1.0 + ref)
        if equation_type == "diffusion-reaction":
            result *= 0.5 * vth * expu_or_u
        if equation_type == "drift-diffusion-reaction":
            result *= (0.5 * vth + abs(sign * mu * df.dot(E, normal))) * expu_or_u
            if particle_type == "electrons":
                result -= 2.0 * gamma * Ion_flux / (1.0 + ref)
        result = 2.0 * df.pi * result * v * r * ds_temp
    elif bc_type == "Neumann" and equation_type == "drift-diffusion-reaction":
        result = 2.0 * df.pi * df.dot(sign * mu * E, normal) * expu_or_u * v * r * ds_temp
    else:
        # default, if bc_type is 'zero_flux', or if no other conditions are met.
        result = 0.0
    return result


def Boundary_flux_1d(
    bc_type: str,
    equation_type: str,
    particle_type: str,
    sign: float,
    mu: Any,
    E: Any,
    normal: Any,
    u: Any,
    gamma: Any,
    v: Any,
    vth: float = 0.0,
    ref: float = 1.0,
    Ion_flux: float = 0.0,
    log_representation: bool = True,
):
    """
    Function defines boundary conditions for different equations.

    Parameters
    ----------
        Type of boundary condition. Options are 'zero flux', 'flux source' or 'Neumann'
    equation_type : str
        Choices are 'reaction', 'diffusion-reaction' or 'drift-diffusion-reaction'
    particle_type : str
        Choices are 'Heavy' or 'electrons'.
    sign: float
        Particle charge sign
    mu: df.Function
        Mobility
    E
        Electric field, Dolfin ComponentTensor
    normal
        Dolfin FacetNormal
    u
        Trial function
    gamma
        Secondary electron emission coefficient
    v
        Test function
    vth: float, default 0.0
        Thermal velocity
    ref: float, default 1.0
        Reflection coefficient for specified particles species and boundary.
    Ion_flux: float, default 0.0
        Flux of ions
    log_representation : bool, default False
        Use logarithmic representation.

    Returns
    -------
    df.Form
        If bc_type is 'flux source' and there is a diffusion term in equation_type.
        Also if the user combines Neumann boundaries with a drift-diffusion-reaction
        equation.
    float
        Otherwise.

    Raises
    ------
    ValueError
        If bc_type is not recognised. If bc_type is not 'zero flux', also raised if
        equation_type is not recognised. Furthermore, if equation_type is
        'drift-diffusion-reaction', raises if particle_type is not recognised.
    """
    bc_types = ["zero flux", "flux source", "Neumann"]
    equation_types = ["reaction", "diffusion-reaction", "drift-diffusion-reaction"]
    particle_types = ["Heavy", "electrons"]

    expu_or_u = df.exp(u) if log_representation else u

    # If provided bc_type has an underscore instead of a space, correct it
    if "_" in bc_type:
        warnings.warn("fedm.BoundaryFlux: bc_type should have spaces, not underscores")
        bc_type = bc_type.replace("_", " ")

    if bc_type not in bc_types:
        err_msg = dedent(
            f"""\
            fedm.Boundary_flux: boundary condition type '{bc_type}' not recognised.
            Must be one of {comma_separated(bc_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    # Only raise error on bad 'equation_type' if the value is needed
    if bc_type != "zero flux" and equation_type not in equation_types:
        err_msg = dedent(
            f"""\
            fedm.Boundary_flux: equation type '{equation_type}' not recognised.
            Must be one of {comma_separated(equation_types)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    # Only raise error on bad 'particle_type' if the value is needed
    if (
        bc_type == "flux source"
        and equation_type == "diffusion-reaction"
        and particle_type not in particle_types
    ):
        err_msg = dedent(
            f"""\
            fedm.Boundary_flux: particle type '{particle_type}' not recognised.
            Must be one of {comma_separated(particle_types)}.
            """
        )
        raise ValueError(err_msg.rstrip())

    if bc_type == "flux source" and equation_type != "reaction":
        result = (1.0 - ref) / (1.0 + ref)
        if equation_type == "diffusion-reaction":
            result *= 0.5 * vth * expu_or_u
        if equation_type == "drift-diffusion-reaction":
            result *= (0.5 * vth + abs(sign * mu * df.dot(E, normal))) * expu_or_u
            if particle_type == "electrons":
                result -= 2.0 * gamma * Ion_flux / (1.0 + ref)
        result = result * v
    elif bc_type == "Neumann" and equation_type == "drift-diffusion-reaction":
        result = df.dot(sign * mu * E, normal) * expu_or_u * v
    else:
        # default, if bc_type is 'zero_flux', or if no other conditions are met.
        result = 0.0
    return result


def Transport_coefficient_interpolation(
    status: str,
    dependences: List[str],
    N0: float,
    Tgas: float,
    k_coeffs: List[df.Function],
    kxs: List[Any],
    kys: List[Any],
    energy: df.Function,
    redfield: df.Function,
    mus: Optional[List[df.Function]] = None,
) -> None:
    """
    Function for linear interpolation of transport coefficients. Modifies k_coeffs.

    Parameters
    ----------
    status: str
        Possible values are 'initial' or 'update'. The former should be used when
        determining the initial definition.
    dependences: List[str]
        Possible values are 'const', 'Umean', 'E/N', 'ESR', or 'Tgas'. Can also be
        set to 0 to skip this k_coeff.
    N0: float
        Gas number density
    Tgas: float
        Gas temperature
    k_coeff: List[df.Function]
        List of coefficient variables
    kx: List[Any]
        Look-up table data
    ky: List[Any]
        Look-up table data
    energy: df.Function
        Energy function
    redfield: df.Function
        Reduced electric field.
    mus: List[df.Function], default None
        Mobilities

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If given incorrect status or dependences, or if using ESR dependence without
        providing mus, or if the provided lists are not all of the same length.
    """

    possible_statuses = ["initial", "update"]
    possible_dependences = [0, "const", "Umean", "E/N", "ESR", "Tgas"]

    if status not in possible_statuses:
        err_msg = dedent(
            f"""\
            fedm.Transport_coefficient_interpolation: status '{status}' not recognised.
            Must be one of {comma_separated(possible_statuses)}.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    for dependence in dependences:
        if dependence not in possible_dependences:
            err_msg = dedent(
                f"""\
                fedm.Transport_coefficient_interpolation: dependence '{dependence}' not
                recognised. Must be one of {comma_separated(possible_dependences)}.
                """
            )
            raise ValueError(err_msg.rstrip().replace("\n", " "))

    # Handle case when mu's are not provided
    if mus is None:
        if "ESR" in dependences:
            raise ValueError(
                "fedm.Transport_coefficient_interpolation: Must provide mus "
                "(mobilities) when using ESR dependence."
            )
        # Expand mus to size of k_coeffs to ensure it can be zipped properly
        mus = [None] * len(k_coeffs)

    # Ensure all args have the correct lengths
    if not all([len(x) == len(k_coeffs) for x in [dependences, kxs, kys, mus]]):
        raise ValueError(
            "fedm.Transport_coefficient_interpolation: The lists 'dependences', 'kxs', "
            "'kys', 'k_coeffs', and (optionally) 'mus' must be the same length."
        )

    for k_coeff, dependence, kx, ky, mu in zip(k_coeffs, dependences, kxs, kys, mus):
        # For 'const', only do something if status is 'initial'
        if dependence == "const" and status == "initial":
            k_coeff.vector()[:] = ky / N0
            k_coeff.vector().update_ghost_values()
        elif dependence == "Umean":
            k_coeff.vector()[:] = np.interp(energy.vector()[:], kx, ky) / N0
            k_coeff.vector().update_ghost_values()
        elif dependence == "E/N":
            k_coeff.vector()[:] = np.interp(redfield.vector()[:], kx, ky) / N0
            k_coeff.vector().update_ghost_values()
        elif dependence == "ESR":
            k_coeff.vector()[:] = kB * Tgas * mu.vector()[:] / elementary_charge
            k_coeff.vector().update_ghost_values()
        elif dependence == "Tgas":
            k_coeff.vector()[:] = np.interp(Tgas, kx, ky) / N0
            k_coeff.vector().update_ghost_values()
        else:
            pass  # If no conditions are met, do nothing


def Rate_coefficient_interpolation(
    status: str,
    dependences: List[str],
    k_coeffs: List[df.Function],
    kxs: List[Any],
    kys: List[Any],
    energy: df.Function,
    redfield: df.Function,
    Te: float = 300.0,
    Tgas: float = 300.0,
) -> None:
    """
    Function for linear interpolation of rate coefficients.

    WARNING: When using the dependences 'fun:Te,Tgas' and 'fun:Tgas', this function
    will evaluate the corresponding contents of 'ky' using 'eval'. It is the user's
    responsibility to ensure no malicious code is injected here.

    Parameters
    ----------
    status: str
        Possible values are 'initial' or 'update'. The former should be used when
        determining the initial definition.
    dependences: List[str]
        Possible values are 'const', 'Umean', 'E/N', 'ESR', or 'Tgas'. Can also be set
        to zero to skip this k_coeff.
    k_coeff: List[df.Function]
        List of coefficient variables
    kx: List[Any]
        Look-up table data
    ky: List[Any]
        Look-up table data
    energy: df.Function
        Energy function
    redfield: df.Function
        Reduced electric field.
    Tgas: float, default 300.0
        Gas temperature. Not used directly, but may be used by user-supplied function.
    Te: float, default 0.0
        Electron temperature. Not used directly, but may be used by user-supplied
        function.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If given incorrect status or dependences, or if the provided lists are not all
        of the same length.
    Exception
        The user may supply their own functions using the 'fun:Te,Tgas' or 'fun:Tgas'
        dependences. These are evaluated using 'eval', meaning anything could happen.
    """
    possible_statuses = ["initial", "update"]
    possible_dependences = [0, "const", "Umean", "E/N", "Te", "fun:Te,Tgas", "fun:Tgas"]

    # Avoid linter warnings about unused variables...
    # Tgas = float(Tgas)
    # Te = float(Te)

    if status not in possible_statuses:
        raise ValueError(
            f"fedm.Rate_coefficient_interpolation: status '{status}' not recognised. "
            f"Must be one of {comma_separated(possible_statuses)}."
        )

    for dependence in dependences:
        if dependence not in possible_dependences:
            raise ValueError(
                f"fedm.Rate_coefficient_interpolation: dependence '{dependence}' not "
                f"recognised. Must be one of {comma_separated(possible_dependences)}."
            )

    # Ensure all args have the correct lengths
    if not all([len(x) == len(k_coeffs) for x in [dependences, kxs, kys]]):
        raise ValueError(
            "fedm.Rate_coefficient_interpolation: The lists 'dependences', 'kxs', "
            "'kys', and 'k_coeffs' must be the same length."
        )

    for k_coeff, dependence, kx, ky in zip(k_coeffs, dependences, kxs, kys):
        # For 'const' and 'fun:...' only do something if status is 'initial'
        if dependence == "const" and status == "initial":
            k_coeff.vector()[:] = ky
            k_coeff.vector().update_ghost_values()
        # Catch both 'fun:Te,Tgas' and 'fun:Tgas'
        elif dependence == "fun" and status == "initial":
            try:
                k_coeff = eval(ky)
                k_coeff.vector().update_ghost_values()
            except Exception as exc:
                raise RuntimeError(
                    "fedm.Rate_coefficient_interpolation: ky eval failed"
                ) from exc
        elif dependence == "Te":
            k_coeff.vector()[:] = np.interp(
                2 * energy.vector()[:] / (3 * kB_eV), kx, ky
            )
            k_coeff.vector().update_ghost_values()
        elif dependence == "Umean":
            k_coeff.vector()[:] = np.interp(energy.vector()[:], kx, ky)
            k_coeff.vector().update_ghost_values()
        elif dependence == "E/N":
            k_coeff.vector()[:] = np.interp(redfield.vector()[:], kx, ky)
            k_coeff.vector().update_ghost_values()
        else:
            pass  # If no conditions are met, do nothing


def semi_implicit_coefficients(
    dependences: List[str],
    mean_energy_new: Any,  # Generated via an expression on mean_energy_old
    mean_energy_old: df.Function,
    coefficients: List[df.Function],
    coefficient_diffs: List[df.Function],
) -> List[df.Function]:
    # TODO needs docstring

    if not all([len(x) == len(dependences) for x in [coefficients, coefficient_diffs]]):
        raise ValueError(
            "fedm.semi_implicit_coefficients: The lists 'dependences', 'coefficients', "
            "and 'coefficient_diffs' must be the same length."
        )

    si_coefficients = []
    for coeff, diff, dep in zip(coefficients, coefficient_diffs, dependences):
        if dep == "Umean":
            si_coefficients.append(coeff + diff * (mean_energy_new - mean_energy_old))
        else:
            si_coefficients.append(coeff)
    return si_coefficients


def Source_term(
    coupling: str,
    approx: str,
    p_matrix: np.ndarray,
    l_matrix: np.ndarray,
    g_matrix: np.ndarray,
    k_coeffs: List[df.Function],
    N0: float,
    u: Any,
) -> List[Any]:
    """
    Defines source term for coupled or uncoupled approach, with
    LFA (counting particles from 0) or LMEA approximation
    (counting particle from 1 and the zeroth equation is energy).

    Parameters
    ----------
    coupling: str
        Either 'coupled' or 'uncoupled'
    approx: str
        Either 'LFA' or 'LMEA'
    p_matrix: np.ndarray
        Power matrix
    l_matrix: np.ndarray
        Loss matrix
    g_matrix: np.ndarray
        Gain matrix
    k_coeffs: List[df.Function]
        Rate coefficient, energy loss and mean energy
    N0: float
        Gas number density
    u
        Trial function

    Returns
    -------
    List[Any]
        Source terms

    Raises
    ------
    ValueError
        If coupling or approx are invalid. Could also be raised if the inputs have the
        wrong shapes.
    """
    couplings = ["coupled", "uncoupled"]
    approximations = ["LFA", "LMEA"]

    if coupling not in couplings:
        raise ValueError("fedm.Source_term: coupling must be 'coupled' or 'uncoupled'.")

    if approx not in approximations:
        raise ValueError("fedm.Source_term: approx must be 'LFA' or 'LMEA'.")

    # In all cases, the zeroth element is the gas number density
    # If coupled and LMEA, we discard the last u, i.e. [N0, *exp(u[1:-1])]
    # If coupled and LFA, we discard the first u, i.e. [N0, *exp(u[0:-1])]
    # If uncoupled, we discard the first u, i.e. [N0, *exp(u[1:])]
    start = 0 if coupling == "coupled" and approx == "LFA" else 1
    end = len(u) - 1 if coupling == "coupled" else len(u)
    exp_u = [N0] + [df.exp(u[i]) for i in range(start, end)]

    temp = np.power(exp_u, p_matrix).prod(axis=-1)
    rate = temp * k_coeffs
    f_temp = (rate[:, np.newaxis] * (g_matrix - l_matrix)).sum(axis=0)

    return list(f_temp)

def Energy_Source_term(
    coupling: str,
    p_matrix: np.ndarray,
    l_matrix: np.ndarray,
    g_matrix: np.ndarray,
    k_coeffs: List[df.Function],
    u_loss: List[float],
    mean_energy: Any,  # dolfin expression
    N0: float,
    n: Any,  # Trial function
    Ei=0,
) -> List[Any]:
    """
    Defines energy source term for LMEA approximation. Function arguments
    are power, loss and gain matrices, rate coeffiients, energy losses for specific
    process, mean electron energy, gas number density and particle number density
    variable.
    """
    """
    Defines energy source term for LMEA approximation.

    Parameters
    ----------
    coupling: str
        Either 'coupled' or 'uncoupled'
    p_matrix: np.ndarray
        Power matrix
    l_matrix: np.ndarray
        Loss matrix
    g_matrix: np.ndarray
        Gain matrix
    k_coeffs: List[df.Function]
        Rate coefficients
    u_loss: List[float]
        Energy losses for specific process
    mean_energy
        Mean electron energy.
    N0: float
        Gas number density
    n:
        Trial function (???)
    Ei:
        particle number density

    Returns
    -------
    List[Any]
        Source terms

    Raises
    ------
    ValueError
        If coupling is invalid. Could also be raised if the inputs have the wrong
        shapes.
    """

    neq = len(n) - 1 if coupling == "coupled" else len(n)
    exp_u = [N0] + [df.exp(n[i]) for i in range(1, neq)]
    temp = np.power(exp_u, p_matrix).prod(axis=-1)
    rate = -temp * k_coeffs
    for idx, loss in enumerate(u_loss):
        if loss > 7e77 and loss < 8e77:
            rate[idx] *= Ei - mean_energy
        elif loss > 9e99 and loss < 1e100:
            rate[idx] *= mean_energy
        else:
            rate[idx] *= loss
    return rate.sum()


def adaptive_timestep(dt, error, tol=1e-4, dt_min=1e-13, dt_max=1e-9):
    """
    Function calculates new time step based on a PID controller M.  Moeller, Time
    stepping  methods,  ATHENS  course: Introductioninto Finite Elements, Delft
    Institute of Applied Mathematics, TU Delft(2015). Input arguments are time step,
    error, tolerance for time step controller, minimal and maximal time step.
    """
    dt *= (
        (error[1] / error[0]) ** 0.075
        * (tol / error[0]) ** 0.175
        * (error[1] ** 2 / (error[0] * error[2])) ** 0.01
    )
    return max(min(dt, dt_max), dt_min)


def adaptive_timestep_PI34(dt, error, tol=1e-4, dt_min=1e-13, dt_max=1e-9):
    """
    Function calculates new time step size using PI.3.4 controller (G. Soederlind
    Numerical Algorithms 31: 281-310, 2002). Input arguments are time-step size, error,
    tolerance for time step controller, minimal and maximal time-step size.
    """
    dt *= (0.8 * tol / error[0]) ** (0.3 / 3) * (0.8 * error[1] / error[0]) ** (0.4 / 3)
    return max(min(dt, dt_max), dt_min)


def adaptive_timestep_H211b(dt, dt_old, error, tol=1e-4, dt_min=1e-13, dt_max=1e-9):
    """
    Function calculates new time step size using H211b controller (G. Soederlind, Acm.
    T. Math. Software 29: 1-26, 2003). Input arguments are time step size, previous time
    step size, error, tolerance for time step controller, minimal and maximal time step.
    """
    dt *= (
        (0.8 * tol / error[0]) ** (1 / 12)
        * (0.8 * tol / error[1]) ** (1 / 12)
        * (dt / dt_old) ** (-1 / 4)
    )
    return max(min(dt, dt_max), dt_min)


class ErrorGreaterThanTTOL(Exception):
    pass


def adaptive_solver(
    nonlinear_solver: Union[df.PETScSNESSolver, df.NonlinearVariationalSolver],
    problem: Problem,
    t: float,
    dt: df.Expression,
    dt_old: df.Expression,
    u_new: df.Function,
    u_old: df.Function,
    var_list_new: List[Any],  # List from Function_definition
    var_list_old: List[Any],  # List from Function definition
    assigner: df.FunctionAssigner,
    error: List[float],
    error_file: Path,
    max_error: List[float],
    ttol: float,
    dt_min: float,
    ite: int = 0,
    time_dependent_arguments: Optional[List[Any]] = None,
    approximation: str = "LMEA",
) -> float:
    """
    This function is used for solving the problem when adaptive time stepping is used.

    Parameters
    ----------
    nonlinear_solver: Union[df.PETScSNESSolver, df.NonlinearVariationalSolver]
        The Dolfin solver to use.
    problem: Problem
        Nonlinear problem definition
    t: float
        Time
    dt: df.Expression
        Time step size
    dt_old: df.Expression
        Previous time step size
    u_new: df.Function
        New variables defined on mixed function space
    u_old: df.Function
        Old variables defined on mixed function space
    var_list_new: List[Any]
        List of new variables for postprocessing
    var_list_old: List[Any]
        List of old variables for postprocessing
    assigner: df.FunctionAssigner
        Assigns values between variables
    error: List[float]
        Record of errors
    error_file: Path
        Error file name
    max_error: List[float]
        Maximum errors
    ttol: float
        Timestepping tolerance
    dt_min: float
        Minimum timestep allowed
    time_dependent_arguments: List[Any], default None
        List of functions that need to be updated with time
    approximation: str, default 'LMEA'
        Type of approximation to use, options are 'LMEA' or 'LFA'.

    Returns
    -------
    float
        The new time 't'

    Raises
    ------
    SystemExit
        If the time step size is reduced below dt_min
    """

    print_rank_0(
        f"Attempting to solve the equation for t = {t} with dt = {dt.time_step}",
        flush=True,
    )

    # Try to advance a time step
    # If an exception is raised, or the error is too large, reset and try again with
    # a smaller time step.
    # Force exit the program if dt < dt_min
    try:
        # Updating time step
        t += dt.time_step

        # Updating time dependent expressions, if there are any.
        if time_dependent_arguments is not None:
            for arg in time_dependent_arguments:
                arg.t = t

        # solving the equation
        ite0, _  = nonlinear_solver.solve(problem, u_new.vector())
        ite += ite0

        # assigning newly calculated values to post-processing variablables
        assigner.assign(var_list_new, u_new)

        # Error estimation.
        # Depending on the used approximation, it is determined from electron
        # energy density, electron number density or, if nothing is specified as
        # an argument, from all the variables solved for.
        if approximation == "LMEA" or approximation == "LFA":
            idx = 0 if approximation == "LMEA" else -2
            var_new, var_old = var_list_new[idx], var_list_old[idx]
        else:
            var_new, var_old = u_new, u_old
        # l2_norm(t, dt.time_step, we_newV, we_oldV)
        error[0] = df.norm(
            var_new.vector() - var_old.vector() + df.DOLFIN_EPS
        ) / df.norm(var_old.vector() + df.DOLFIN_EPS)

        # Writing relative error to file
        with open(error_file, "a") as f_err:
            f_err.write(f"{error[0]:<23}  {dt_old.time_step:<23}  {dt.time_step:<23}\n")
            f_err.flush()

        # Update maximum error
        max_error[0] = max(error)

        # If maximum error is greater than the time stepping tolerance, the
        # variables are reset to previous time step and calculations are repeated
        # with the reduced time step size.
        if error[0] >= ttol:
            raise ErrorGreaterThanTTOL

    except Exception as exc:
        # Reseting time step to the previous time step
        t -= dt.time_step

        # Reducing time-step size and print error msg to screen
        if isinstance(exc, ErrorGreaterThanTTOL):
            dt.time_step *= 0.5 * ttol / max_error[0]
            print_rank_0(
                "Residual is greater than the prescribed tolerance. Reducing "
                "time-step size and repeating calculation."
            )
        else:
            dt.time_step *= 0.5
            print_rank_0(
                "An exception was raised while solving. Reducing time-step size "
                "and repeating calculation."
            )

        # If it's too small, force close program
        if dt.time_step < dt_min:
            raise SystemExit("Minimum time-step size reached, program is terminating.")

        # reseting variables to the previous time step
        u_new.assign(u_old)

        # assigning reset values to post-processing variablables
        assigner.assign(var_list_new, u_new)

        # Call self with args reset and new time step
        t, ite = adaptive_solver(
            nonlinear_solver,
            problem,
            t,
            dt,
            dt_old,
            u_new,
            u_old,
            var_list_new,
            var_list_old,
            assigner,
            error,
            error_file,
            max_error,
            ttol,
            dt_min,
            ite,
            time_dependent_arguments,
            approximation,
        )

    # Return the new time step
    return t, ite


def Normal_vector(mesh: df.Mesh):
    # TODO Write docstring
    W = df.VectorFunctionSpace(mesh, "CG", 1)

    # Projection of the normal vector on P1 space
    u = df.TrialFunction(W)
    v = df.TestFunction(W)
    n = df.FacetNormal(mesh)
    a = df.inner(u, v) * df.ds
    L = df.inner(n, v) * df.ds

    # Solve system
    A = df.assemble(a, keep_diagonal=True)
    b = df.assemble(L)
    A.ident_zeros()
    n = df.Function(W)
    df.solve(A, n.vector(), b, "mumps")

    return n


def Poisson_solver(A, L, b, bcs, u, solver_type="mumps", preconditioner="hypre_amg"):
    # TODO Write docstring
    b = df.assemble(L, tensor=b)
    [bc.apply(b) for bc in bcs]
    if solver_type == "mumps":
        df.solve(A, u.vector(), b, solver_type)
    else:
        df.solve(A, u.vector(), b, solver_type, preconditioner)


def BoundaryGradient(var, zeroDomain, source_term, ds_extract, epsilon=8.854187817e-12):
    """
    The function is an adaptation of the code snippet by D. Kamensky from
    https://fenicsproject.discourse.group/t/compute-gradient-of-scalar-field-on-boundarymesh/1172/2.
    It is used for the accurate calculation of the flux (in this case the electric
    field) across the specific boundary. Input parameters are variables whose gradient
    needs to be determined, marker of the whole domain except the boundary on which
    flux is calculated, the source term of the equation, list of ds of the specific
    boundaries (irrelevant boundaries should be marked as zero).
    """
    V = var.ufl_function_space()
    antiBCs = [df.DirichletBC(V, df.Constant(0.0), zeroDomain())]
    n = df.FacetNormal(V.mesh())

    E = df.TrialFunction(V)  # Electric field trial function
    v_phi = df.TestFunction(V)  # Potential test function
    En = df.Function(V)

    # Poisson equation residual
    res = (
        epsilon * df.inner(df.grad(var), df.grad(v_phi)) * df.dx
        - source_term * v_phi * df.dx
    )

    consistencyTerm = epsilon * df.inner(-df.grad(var), n) * v_phi * ds_extract[0]
    F = 0
    for i, ds in enumerate(ds_extract):
        if i == 0:
            F += -consistencyTerm
        else:
            F += epsilon * E * v_phi * df.ds

    F += res

    a, L = df.lhs(F), df.rhs(F)

    A = df.assemble(a, keep_diagonal=True)
    [bc.apply(A) for bc in antiBCs]

    b = df.assemble(L)
    [bc.apply(b) for bc in antiBCs]

    df.solve(A, En.vector(), b, "gmres", "hypre_amg")

    return En

