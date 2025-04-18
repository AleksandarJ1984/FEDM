# Input and output module

import re
import itertools
from pathlib import Path
from textwrap import dedent
from typing import List, Any

from .utils import comma_separated, print_rank_0, mesh_info

import numpy as np
import pandas as pd
import dolfin as df

# TODO This could be set via a 'verbose' argument to a solver.
level = 40  # False | 10 | 13 | 16 | 20 | 30| 40 | 50
df.set_log_level(level)

# Functions for managing input/output files


def truncate_file(path: Path) -> None:
    """
    Deletes contents of file, leaving it as an empty file.
    If the file does not exist, creates the empty file and any directories up to it.
    """
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    with open(path, "w") as _:
        pass


class Files:
    """
    The Files class provides access to the files and directories used by FEDM. It should
    normally be accessed via the global instance `fedm.files`.

    It has the following attributes:
    - file_input: Directory containing input files, default './file_input'. Can be
      assigned to a new path. Throws an error if the new path doesn't exist.
    - output_folder_path: Directory which will contain output files, default './output'.
      Can be assigned to a new path. This will create a new directory if it doesn't
      already exist.
    - error_file: File storing relative errors. Cannot be assigned, and will always
      return 'output_folder_path/relative error.log'. The first time the user accesses
      it per run, the file will be created/truncated.
    - model_log: File storing logs describing the run. Cannot be assigned, and will
      always return 'output_folder_path/model.log'. The first time the user accesses
      it per run, the file will be created/truncated.

    Examples:

    -- Assign a new value to file_input (folder must exist)
    files.file_input = "my_file_input"
    -- Get the current value of output_folder_path
    output = files.output_folder_path
    -- Get error file (first access will truncate the file)
    error_file = files.error_file
    """

    def __init__(self):
        self._input_dir = Path.cwd() / "file_input"
        self._output_dir = Path.cwd() / "output"
        self._error_file_accessed = False
        self._model_log_accessed = False

    @property
    def file_input(self) -> Path:
        return self._input_dir

    @file_input.setter
    def file_input(self, value: Path) -> None:
        value = Path(value)
        if not value.is_dir():
            raise RuntimeError(f"fedm.files.file_input: '{value}' is not a directory")
        self._input_dir = value

    @property
    def output_folder_path(self) -> Path:
        return self._output_dir

    @output_folder_path.setter
    def output_folder_path(self, value: Path) -> None:
        value = Path(value)
        # If setting to a new output dir, ensure error/log files will be truncated
        # on next access
        if value.resolve() != self._output_dir.resolve():
            self._error_file_accessed = False
            self._model_log_accessed = False
        # If the directory doesn't exist, create it
        if not value.is_dir():
            value.mkdir()
        # Finally, set internal output dir
        self._output_dir = value

    @property
    def error_file(self) -> Path:
        result = self.output_folder_path / "relative error.log"
        # If this is the first time we've accessed this error file, truncate
        if not self._error_file_accessed:
            truncate_file(result)
            self._error_file_accessed = True
        return result

    @property
    def model_log(self) -> Path:
        result = self.output_folder_path / "model.log"
        # If this is the first time we've accessed this error file, truncate
        if not self._model_log_accessed:
            truncate_file(result)
            self._model_log_accessed = True
        return result


# global Files instance
files = Files()


# Utilities for use in this module


def no_convert(x: Any) -> Any:
    """
    This utility function is used in functions that may optionally try to convert
    inputs to a given type. 'no_convert' may be used in place of 'float', 'str', etc
    to prevent the function from converting types.
    """
    return x


def decomment(lines: List[str]) -> str:
    """
    Removes comment at end of each line, denoted with '#'. If the line is blank, or the
    line starts with a '#', returns an empty string. Works as a generator function.
    Code snippet addapted from:
    https://stackoverflow.com/questions/14158868/python-skip-comment-lines-marked-with-in-csv-dictreader
    """
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            yield line


# file_io functions


def output_files(
    file_type: str, type_of_output: str, output_file_names: List[str]
) -> List[Any]:
    """
    Creates desired number of output files.

    Parameters
    ----------
    file_type: str
        'pvd' or 'xdmf'
    type_of_output: str
        Name of folder where files should be saved
    output_file_names
        List of files to create
    """
    FileTypes = {
        "pvd": df.File,
        "xdmf": df.XDMFFile,
    }

    try:
        FileType = FileTypes[file_type]
    except KeyError as exc:
        err_msg = dedent(
            f"""\
            fedm.output_files: file type '{file_type}' is not valid. Options are
            'pvd' or 'xdmf'.
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " ")) from exc

    output_dir = files.output_folder_path / type_of_output

    file_paths = []
    for file_name in output_file_names:
        file_path = FileType(str(output_dir / file_name / f"{file_name}.{file_type}"))
        if file_type == "xdmf":
            file_path.parameters["flush_output"] = True
        file_paths.append(file_path)

    return file_paths


def read_single_value(file_name, convert=no_convert):
    """
    Reads file containing only constant.
    Input parameter is file name.
    """
    with open(file_name, "r", encoding="utf8") as f_input:
        # Finds first non-comment and non-blank line, passes through convert function
        return convert(next(decomment(f_input)))
    raise RuntimeError(f"fedm.read_single_value: No value found in file '{file_name}'")


def read_single_float(file_name, convert=no_convert):
    """
    Reads file containing only constant.
    Input parameter is file name.
    """
    return read_single_value(file_name, convert=float)


def read_single_string(file_name):
    """
    Reads file containing one column.
    Input parameter is file name.
    """
    return read_single_value(file_name, convert=str)


def read_and_decomment(file_name: str) -> List[str]:
    """
    Reads file, returns list of strings. Comment lines and blank lines are removed.
    """
    with open(file_name, "r") as f_input:
        return [line for line in decomment(f_input)]


def read_two_columns(file_name):
    """
    Reads two column files.
    Input parameter is file name.
    """
    data = pd.read_csv(file_name, comment="#", header=None, sep=r"\s+", dtype=float)
    # TODO test if we can avoid list conversions
    return list(data[0]), list(data[1])


def flatten(input_list: List[List[Any]]) -> List[Any]:
    """
    Reduces 2D list to 1D list.
    """
    return list(itertools.chain.from_iterable(input_list))


def flatten_float(input_list: List[List[Any]]) -> List[float]:
    """
    Reduces 2D list to 1D list and converts elements to float.
    """
    return [float(x) for x in flatten(input_list)]


def read_speclist(file_path):
    """
    Function reads list of species from 'speclist.cfg' file
    and extracts species names and species properties filename.
    """

    file_name = Path(file_path) / "speclist.cfg"

    # Get all lines from the file which have 'file:' in them
    lines = [line for line in read_and_decomment(file_name) if "file:" in line]

    # remove "file:" from each line and split by whitespace
    lines = [line.replace("file:", "").split() for line in lines]

    # Get data to return
    species_names = [line[0] for line in lines]
    species_properties_file_names = [line[1] for line in lines]
    species_name_tc = [line[1].split(".")[0] for line in lines]
    p_num = len(species_names)

    return p_num, species_names, species_properties_file_names, species_name_tc


def reaction_matrices(path: str, species: List[str]):
    """
    Reads the reaction scheme from "reacscheme.cfg" file and creates power, loss and
    gain matrices.
    """

    file_name = Path(path) / "reacscheme.cfg"

    reactions = [line.partition(" Type:")[0] for line in read_and_decomment(file_name)]
    loss = [reaction.partition(" -> ")[0].rstrip() for reaction in reactions]
    gain = [reaction.partition(" -> ")[2].rstrip() for reaction in reactions]

    l_matrix = np.empty([len(reactions), len(species)], dtype=int)
    g_matrix = np.empty([len(reactions), len(species)], dtype=int)
    for i, j in itertools.product(range(len(reactions)), range(len(species))):
        l_matrix[i, j] = loss[i].count(species[j])
        g_matrix[i, j] = gain[i].count(species[j])

    power_matrix = l_matrix
    lg_matrix = l_matrix - g_matrix
    loss_matrix = np.where(lg_matrix > 0, lg_matrix, 0)
    gain_matrix = np.where(lg_matrix < 0, -lg_matrix, 0)

    return power_matrix, loss_matrix, gain_matrix


def rate_coefficient_file_names(path):
    """
    Reads names of reaction coefficient files from "reacscheme.cfg" file.
    Input parameter is the path to the folder.
    """

    reaction_scheme_file = Path(path) / "reacscheme.cfg"
    rate_coefficient_dir_path = Path(path) / "rate_coefficients"

    lines = read_and_decomment(reaction_scheme_file)
    regex = re.compile(r"kfile: ([A-Za-z0-9_]+.[A-Za-z0-9_]+)")
    file_names = flatten([regex.findall(line) for line in lines])
    return [rate_coefficient_dir_path / file_name for file_name in file_names]


def read_energy_loss(path):
    """
    Reads energy loss values from "reacscheme.cfg" file.
    Input argument is the path to the folder.
    """

    reaction_scheme_file = Path(path) / "reacscheme.cfg"
    lines = read_and_decomment(reaction_scheme_file)

    regex = re.compile(r"Uin:\s?([+-]?\d+.\d+[eE]?[-+]?\d+|0|1.0)")
    energy_loss_value = flatten_float([regex.findall(line) for line in lines])

    print_rank_0(energy_loss_value)
    return energy_loss_value


def read_dependence(file_name: str):
    """
    Reads dependence of rate coefficients from the corresponding file.
    """
    file_name = Path(file_name)
    if not file_name.is_file():
        raise FileNotFoundError(f"fedm.read_dependence: file '{file_name}' not found")
    with open(file_name, "r", encoding="utf8") as f_input:
        for line in f_input:
            if "Dependence:" in line:
                return line.split()[2]
    raise RuntimeError(
        f"fedm.read_dependence: Did not find dependence in file '{file_name}'"
    )


def read_dependences(file_names: List[str], zero_if_file_missing=False):
    """
    Reads dependence of rate coefficients from a list of corresponding files.
    """
    dependences = []
    for file_name in file_names:
        try:
            dependences.append(read_dependence(file_name))
        except FileNotFoundError as exc:
            if zero_if_file_missing:
                dependences.append(0)
            else:
                raise exc
    return dependences


def read_rate_coefficients(rc_file_names: List[str], k_dependences: List[str]):
    """
    Reading rate coefficients from files. Input are file names and dependences.
    """
    if len(rc_file_names) != len(k_dependences):
        raise ValueError(
            "fedm.read_rate_coefficients: rc_file_names and k_dependences should be "
            "the same length."
        )

    float_dependences = ["const"]
    str_dependences = ["fun:Te,Tgas", "fun:Tgas"]
    two_col_dependences = ["Umean", "E/N", "ElecDist"]
    all_dependences = float_dependences + str_dependences + two_col_dependences
    for dependence in k_dependences:
        if dependence not in all_dependences:
            raise ValueError(
                f"fedm.read_rate_coefficients: The dependence '{dependence}' is not "
                f"recognised. Options are {comma_separated(all_dependences)}."
            )

    kxs, kys = [], []
    for dependence, rc_file_name in zip(k_dependences, rc_file_names):
        print_rank_0(rc_file_name)
        if dependence in two_col_dependences:
            kx, ky = read_two_columns(rc_file_name)
        elif dependence in float_dependences:
            kx, ky = 0.0, read_single_float(rc_file_name)
        else:  # dependence in string_dependences
            kx, ky = 0.0, read_single_string(rc_file_name)
        kxs.append(kx)
        kys.append(ky)

    return kxs, kys


def read_transport_coefficients(
    particle_names: List[str], transport_type: str, model: str
):
    """
    Reading transport coefricients from files. Input are
    particle names, type of transport coefficient (diffusion or mobility)
    and model.
    """
    path = files.file_input / model / "transport_coefficients"
    if not path.is_dir():
        raise FileNotFoundError(
            f"fedm.read_transport_coefficients: Transport coeff dir '{path}' not found."
        )

    float_dependences = ["const", "const."]
    str_dependences = ["fun:Te,Tgas", "fun:E"]
    two_col_dependences = ["Umean", "E/N", "Tgas", "Te"]
    all_dependences = float_dependences + str_dependences + two_col_dependences
    if transport_type == "Diffusion":
        all_dependences.append("ESR")
    if transport_type == "mobility":
        all_dependences.append(0)

    # Get dependences
    file_suffix = "_ND.dat" if transport_type == "Diffusion" else "_Nb.dat"
    file_names = [path / (particle + file_suffix) for particle in particle_names]
    k_dependences = read_dependences(
        file_names,
        zero_if_file_missing=(transport_type == "mobility"),
    )

    # Throw error if any dependences aren't recognised
    for dependence in k_dependences:
        if dependence not in all_dependences:
            err_msg = dedent(
                f"""\
                fedm.read_transport_coefficients: Dependence '{dependence}' not
                recognised. For the transport type '{transport_type}', the possible
                options are {comma_separated(all_dependences)}.
                """
            )
            raise ValueError(err_msg.rstrip().replace("\n", " "))

    # Get kx and ky from each file
    kxs, kys = [], []
    for file_name, dependence in zip(file_names, k_dependences):

        # If transport is 'mobility' and dependence is 0, the file was missing.
        # Set to zeros and skip.
        if transport_type == "mobility" and dependence == 0:
            kxs.append(0)
            kys.append(0)
            continue

        print_rank_0(file_name)

        if dependence in two_col_dependences:
            kx, ky = read_two_columns(file_name)
        elif dependence == "ESR":
            kx, ky = 0.0, 0.0
        elif dependence in float_dependences:
            kx, ky = 0.0, read_single_float(file_name)
        else:  # dependence in str_dependences
            kx, ky = 0.0, read_single_string(file_name)

        if dependence == "fun:Te,Tgas":
            try:
                ky_str = ky  # save reference in case we need it in an error message
                ky = eval(ky)
            except Exception as exc:
                raise RuntimeError(
                    f"fedm.read_transport_coefficients: ky eval failed, '{ky_str}'"
                ) from exc

        kxs.append(kx)
        kys.append(ky)

    return kxs, kys, k_dependences


def read_particle_properties(file_names: List[str], model: str):
    """
    Reads particle properties from input file.
    Input are file names and model.
    """
    path = files.file_input / model / "species"

    masses, charges = [], []
    regex_mass = re.compile(r"Mass\s?=\s?([+-]?\d+.\d+[eE]?[-+]?\d+|0|1.0)")
    regex_charge = re.compile(r"Z\s+?=\s+?([+-]?\d+)")

    for file_name in file_names:
        # Get full file name, test if file exists
        file_name = path / file_name
        if not file_name.is_file():
            raise RuntimeError(
                f"fedm.read_particle_properties: File '{file_name}' not found."
            )
        print_rank_0(file_name)

        # Read file
        lines = read_and_decomment(file_name)

        # Get mass and charge from file
        mass_found, charge_found = False, False
        for line in lines:
            print_rank_0(line)
            mass, charge = regex_mass.findall(line), regex_charge.findall(line)
            if mass:
                mass_found = True
                masses.append(float(mass[0]))
            if charge:
                charge_found = True
                charges.append(float(charge[0]))
        if not mass_found:
            raise RuntimeError(
                f"fedm.read_particle_properties: No mass found in file '{file_name}'."
            )
        if not charge_found:
            raise RuntimeError(
                f"fedm.read_particle_properties: No charge found in file '{file_name}'."
            )

    return masses, charges


def print_time_step(dt):
    """
    Prints time step.
    """
    print_rank_0("Time step is dt =", dt)


def print_time(t):
    """
    Prints time.
    """
    print_rank_0("t =", t)


def file_output(
    t,
    t_old,
    t_out,
    step,
    t_out_list,
    step_list,
    file_type,
    output_file_list,
    particle_name,
    u_old,
    u_old1,
    unit="s",
):
    """
    Writing value of desired variable to file. Value is calculated by linear
    interpolation.  Input arguments are current time step, previous time step, current
    time step length, previous time step length, list of output times, list of steps
    (which needs have same size), list of output files, list of variable values in
    current and previous time steps, and (optional) units of time.
    """
    units = {
        "ns": 1e9,
        "us": 1e6,
        "ms": 1e3,
        "s": 1.0,
    }

    try:
        scale = units[unit]
    except KeyError:
        err_msg = dedent(
            f"""\
            fedm.file_output: unit '{unit}' not valid.
            Options are {comma_separated(list(units))}.'
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    if t > max(t_out_list):
        index = len(t_out_list) - 1
    else:
        index = next(x for x, val in enumerate(t_out_list) if val > t)

    while t_out <= t:
        for i in range(len(output_file_list)):
            temp = df.Function(u_old1[i].function_space())
            temp.vector()[:] = u_old1[i].vector()[:] + (t_out - t_old) * (
                u_old[i].vector()[:] - u_old1[i].vector()[:]
            ) / (t - t_old)
            if file_type[i] == "pvd":
                temp.rename(particle_name[i], str(i))
                # TODO find functional version of this, avoid C++ operator overloads
                output_file_list[i] << (temp, t_out * scale)
            elif file_type[i] == "xdmf":
                # appending to file
                output_file_list[i].write_checkpoint(
                    temp,
                    particle_name[i],
                    t_out * scale,
                    df.XDMFFile.Encoding.HDF5,
                    True,
                )
            else:
                err_msg = dedent(
                    f"""\
                    fedm.file_output: file type '{file_type}' not recognised.
                    Options are 'pvd' and 'xdmf'.
                    """
                )
                raise ValueError(err_msg.rstrip().replace("\n", " "))

        if t_out >= 0.999 * t_out_list[index - 1] and t_out < 0.999 * t_out_list[index]:
            step = step_list[index - 1]
        elif t_out >= 0.999 * t_out_list[index]:
            step = step_list[index]
        # FIXME undefined if t_out < 0.999 * t_out_list[index - 1], need else statement
        t_out += step
    return t_out, step


def mesh_statistics(mesh: df.Mesh) -> None:
    """
    Returns mesh size and, maximum and minimum element size.
    Input is mesh.
    """
    mesh_dir = files.output_folder_path / "mesh"
    vtkfile_mesh = df.File(str(mesh_dir / "mesh.pvd"))
    vtkfile_mesh.write(mesh)
    info_str = mesh_info(mesh)
    if df.MPI.rank(df.MPI.comm_world) == 0:
        print(info_str.rstrip())
        with open(mesh_dir / "mesh info.txt", "w") as mesh_information:
            mesh_information.write(info_str)


def numpy_2d_array_to_str(x: np.ndarray) -> str:
    # Remove '[' and  ']' from str representation
    no_brackets = str(np.asarray(x)).replace("[", "").replace("]", "")
    # Remove extra whitespace on each line, return
    return "\n".join([y.strip() for y in no_brackets.split("\n")])


def log(log_type, log_file_name, *args):
    """
    The function is used to log model data and its progress.
    The input arguments vary depending on the type of logging.
    For type = properties, input are gas, model, particle species
    names, particle mass, and charge.
    For type = conditions, input arguments are time step length,
    working voltage, pressure, gap length, gas number density,
    and gas temperature.
    For type = matrices, input arguments are gain, loss and power
    matrices.
    For type = initial_time, input argument is time.
    For type = time, input argument is time.
    For type = mesh, input argument is mesh.
    """

    if df.MPI.rank(df.MPI.comm_world) != 0:
        return

    if log_type == "properties":
        gas, model, particle_species_file_names, M, charge = args
        log_str = dedent(
            f"""\
            Gas:\t{gas}

            model:\t{model}

            Particle names:
            {particle_species_file_names}

            Mass:
            {M}

            Charge:
            {charge}
            """
        )
    elif log_type == "conditions":
        dt_var, U_w, p0, gap_length, N0, Tgas = args
        log_str = dedent(
            f"""\
            dt = {dt_var} s,
            U_w = {U_w} V,
            p_0 = {p0} Torr,
            d = {gap_length} m,
            N_0 = {N0} m^-3,
            T_gas = {Tgas} K
            """
        )
        log_str = log_str.rstrip().replace("\n", "\t ")
        log_str = f"Simulation conditions:\n{log_str}\n"
    elif log_type == "matrices":
        gain, loss, power = args
        log_str = dedent(
            f"""\
            Gain matrix:
            {numpy_2d_array_to_str(gain)}

            Loss matrix:
            {numpy_2d_array_to_str(loss)}

            Power matrix:
            {numpy_2d_array_to_str(power)}
            """
        )
    elif log_type == "initial time":
        log_str = f"Time:\n{args[0]}"
    elif log_type == "time":
        log_str = str(args[0])
    elif log_type == "mesh":
        log_str = mesh_info(args[0])
    else:
        err_msg = dedent(
            f"""\
            fedm.log: log_type '{log_type}' not recognised. Options are 'properties',
            'conditions', 'matrices', 'initial time', 'time', or 'mesh'
            """
        )
        raise ValueError(err_msg.rstrip().replace("\n", " "))

    with open(log_file_name, "a") as log_file:
        log_file.write(log_str)
        log_file.write("\n")
        log_file.flush()
