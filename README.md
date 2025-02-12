# FEDM

[![logo](docs/build/html/_images/logo.svg)](https://doi.org/10.1088/1361-6595/acc54b)

## Description

Finite Element Discharge Modelling (FEDM) code utilises the FEniCS (https://fenicsproject.org) computing platform for fluid modelling of the electrical discharges operating under various conditions. The code extends FEniCS with features that allow the automated implementation and numerical solution of fully-coupled fluid-Poisson models including an arbitrary number of particle balance equations. The fluid-Poisson models comprise the system of balance equations for particle species, coupled with Poisson's equation and, depending on the used approximation, the electron energy balance equation. In practice, it is often required to take into account a large number of particle species and processes, so the manual implementation of the balance equations and the source terms becomes a time-consuming, tedious and error-prone process. This package automates the procedure by providing a set of functions that allows an easy definition of the problem. In addition, the package offers a method for the time discretisation of the time-dependent equations using a variable time-step backward differentiation formula with appropriate time-step size control. This time-discretisation method allows one to tackle the problem of stiff equations that commonly occur in plasma modelling due to the very different time scales of the various reaction processes.

## Build status

Initial build - version 1.0

## Features

- automated implementation of the variational forms of the Poisson's equation and an arbitrary number of balance equations for particle species based on a predefined species list
- automated source term generation based on reading the reaction scheme from the input file
- adaptive time step refinement using variable time-step Backward differentiation formula (BDF) of the second-order


## Installation

FEniCS version 2019.1.0. is required to run the FEDM code. The easiest way to obtain the desired version of FEniCS is using a Docker image. Alternatively, FEniCS can be installed via Anaconda or from binary files as described [here](https://fenicsproject.org/download/archive/). More information on how to install the Docker and run FEDM can be found in documentation.  The brief description, assuming that FEniCS is already installed, is given below.

The FEDM code directory has the following structure:

```
FEDM
├── examples
│   ├── glow_discharge
│   ├── streamer_discharge
│   ├── time_of_flight
│   └── time_of_flight_1D
├── fedm
│   ├── file_io.py
│   ├── functions.py
│   ├── __init__.py
│   ├── physical_constants.py
│   ├── __pycache__
│   └── utils.py
├── LICENSE
├── pyproject.toml
├── README.md
├── run_tests.sh
├── setup.cfg
├── setup.py
└── tests
    ├── integrated_tests
    └── unit_tests
```

FEDM can be installed within the Docker container using:

```bash
python3 -m pip install --user .
```

The directory `examples` contains the code for the three case studies described in following [article](https://doi.org/10.1088/1361-6595/acc54b). One can execute each example by running the following command in the corresponding directory (you may need to use `sudo`):

```bash
python3 fedm-name_of_example.py
```

or in parallel using MPI:

```bash
mpirun –np 8 python3 fedm-name_of_example.py
```

Note that the new experimental version FEniCSx 0.7 has been recently published, while the FEniCS 2019.1.0 is considered to be deprecated. Currently, there are no plans to update FEDM to be compatible with the newest version (at least until the stable FEniCSx version is published).

## Testing

Testing must be performed within the Docker container. The testing dependencies should
be installed using:

```bash
python3 -m pip install --user .[tests]
```

The Python `vtk` library is needed to read output files, which in turn requires OpenGL,
so you may need to install this:

```bash
sudo apt update
sudo apt install libgl1
```

The tests are then run using the `run_tests.sh` script (which may need `sudo`):

```bash
./run_tests.sh
```

The tests will run each of the integrated tests and store data in the container's `/tmp`
directory. This is why the container must be run with `-v /tmp`.

Note that calling `pytest tests/` will likely lead to false negatives. FEniCS must reset
between each run, so each integrated test after the first one is likely to fail.

## How to use?

FEDM consists of the three modules

- `functions.py` storing the functions used for problem definition,

- `file_io.py` for input and output of the data and

- `physical_constants.py` for storing the physical constants.

The main script is used to set up and solve the problem using the functions stored in these modules.

## Illustration
Streamer development in dielectric barrier discharge modelled with FEDM

![Streamer development in dielectric barrier discharge modelled with FEDM](docs/build/html/_images/animation.gif)


## Note from the author
Initally, the FEDM code was developed while the author was learning the Python language. Most of the functions were not written in a Pythonic way. Thanks to Dr. Peter Hill and Dr. Liam Pattinson of the PlasmaFAIR project, the health check of the code has been performed and the code has been significantly improved.

## Contributing
Any kind of contribution that would improve the project, whether new feature suggestions, bug reports or providing your own code suggestions, is welcome and encouraged.  Please, try to follow the following guidelines when contributing to the project.

**Bug reports**

Before creating bug reports, please check if somebody already reported the same bug, use a title that clearly describes the issue, provide a detailed description of the bug (including how you use FEDM, on what machine and operating system) and the minimum working example that illustrates the bug.

**Feature suggestion**

If you have a new feature suggestion, please check if it has already been suggested and then clearly describe the idea behind the new feature you would like to see in the FEDM.

**Code suggestion**

When contributing new code or modifying the existing, the authors would appreciate it if you would first discuss the changes you wish to make by creating a new issue or sending an e-mail to the authors.

In any case, communicating with the authors will make it easier to incorporate your changes and make the experience smoother for everyone. The authors look forward to your input.

## License

In agreement with the FEniCS licensing, FEDM is open source code developed under LGPLv3 (GNU Lesser General Public License version 3).

## Acknowledgment

The development of the FEDM is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)—project number 407462159. The authors wish to thank the users of [FEniCS forum](https://fenicsproject.discourse.group) for useful information and discussion. Finally, the authors are grateful to Dr. Peter Hill and Dr. Liam Pattinson of the PlasmaFAIR project for carrying out the health check, and proposing and implementing significant improvements to the code. This support of PlasmaFAIR, funded by EPSRC (grant no. EP/V051822/1), is gratefully acknowledged.

## Citation

Please cite the article Aleksandar P. Jovanović, Detlef Loffhagen, Markus M. Becker 2023 Introduction and verification of FEDM, an open-source FEniCS-based discharge modelling code *Plasma Sources Sci. Technol.* **32** 044003 https://doi.org/10.1088/1361-6595/acc54b if you use the FEDM code for your research.

```
@article{Jovanovic-2023,
author    = {Jovanović, Aleksandar P and Loffhagen, Detlef and Becker, Markus M},
title     = {Introduction and verification of {FEDM}, an open-source {FE}ni{CS}-based
            discharge modelling code},
journal   = {Plasma Sources Science and Technology},
year      = {2023},
volume    = {32},
pages     = {044003},
number    = {4},
doi       = {10.1088/1361-6595/acc54b},
publisher = {IOP Publishing}
}
```

## Contact

[aleksandar.jovanovic@inp-greifswald.de](mailto:aleksandar.jovanovic@inp-greifswald.de)

[markus.becker@inp-greifswald.de](mailto:markus.becker@inp-greifswald.de)
