import yaml
from mpi4py import MPI
from MMC.error_exit import error_exit

class Settings:
    def __init__(
            self,
            system,
            PyLAMMPS,
            MetropolisMC,
            visual,
    ):
        self.system = system
        self.PyLAMMPS = PyLAMMPS
        self.MetropolisMC = MetropolisMC
        self.visual = visual

    def __str__(self):
        keys = ["system", "PyLAMMPS", "MetropolisMC", "visual"]
        s = ""
        for key in keys:
            s += key + ":\n"
            if key == "system":
                s += str(self.system) + "\n"
            elif key == "PyLAMMPS":
                s += str(self.PyLAMMPS) + "\n"
            elif key == "MetropolisMC":
                s += str(self.MetropolisMC) + "\n"
            elif key == "visual":
                s += str(self.visual) + "\n"

        return s

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            parameters = yaml.safe_load(f)

        thissystem = {"significant_figures": 6, "float_precision": 3, "VerySmallNumber": 1.0e-20,
                       "Tolerance": 0.1}

        if "system" in parameters:
            tsystem = parameters["system"]
            for key in tsystem:
                thissystem[key] = tsystem[key]

        thisPyLAMMPS = {"Screen": False, "Log": False,
                        "FileName": "lmp.data", "atom_style": "atomic", "Input4LAMMPS": "in.lmp"}

        PyLAMMPS = parameters["PyLAMMPS"]
        if "FileName" not in PyLAMMPS:
            errormsg = "There must be a FileName for data!"
            error_exit(errormsg)
        if "Input4LAMMPS" not in PyLAMMPS:
            errormsg = "There must be an atom_style for data!"
            error_exit(errormsg)

        for key in PyLAMMPS:
            thisPyLAMMPS[key] = PyLAMMPS[key]

        thisMMC = {"ntypes": 1, "EREFs": None, "ff_elements": None,
                   "ratio_hot": 0.1, "ratio_hot2": 0.3, "norm": "auto", "min_norm": 0.02,
                   "ratio_shift": 0.1,
                   "Exclude_types": None, "Enforce_types": None,
                   "Inteval4Enforce": 1,
                   "Exclude_mid": False,
                   "LoopMax": 1000000,
                   "Nsteps4Relax": 100,
                   "Nsteps4Checkpoint": 100,
                   "Nsteps4UpdateEREFs": 1000,
                   "Temperature": 300.0,
                   "Tolerance": 0.001,
        }

        MMC = parameters['MetropolisMC']
        if "ntypes" not in MMC:
            errormsg = "There must be a ntypes for # of types in data file!"
            error_exit(errormsg)
        for key in MMC:
            thisMMC[key] = MMC[key]

        thisvisual = {"SummaryFile": "MMC_Summary.csv", "LogFile": "MMC.log",
                      "Screen": True, "Log": True, "DataOut_Path": "DataOut",
                      "Nsteps4WriteData": 100, "Nsteps4Visual": 10, "Nsteps4Summary": 10,
                      }

        visual = parameters['visual']
        for key in visual:
            thisvisual[key] = thisvisual[key]

        thissett = cls(thissystem, thisPyLAMMPS, thisMMC, thisvisual)
        return thissett