import os
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
                        "FileName": "lmp.data", "atom_style": "atomic",
                        "Input4LAMMPS": "in.lmp", "Input4Relax": None}

        PyLAMMPS = parameters["PyLAMMPS"]
        if "FileName" not in PyLAMMPS:
            errormsg = "There must be a FileName for data!"
            error_exit(errormsg)
        if "Input4LAMMPS" not in PyLAMMPS:
            errormsg = "There must be an atom_style for data!"
            error_exit(errormsg)

        for key in PyLAMMPS:
            thisPyLAMMPS[key] = PyLAMMPS[key]
        if isinstance(thisPyLAMMPS["Input4Relax"], str) and os.path.isfile(thisPyLAMMPS["Input4Relax"]):
            with open(thisPyLAMMPS["Input4Relax"], "r") as f:
                lines = f.readlines()
            thisPyLAMMPS["relax_lines"] = lines
        else:
            thisPyLAMMPS["relax_lines"] = []

        thisMMC = {"ntypes": 1, "EREFs": None, "ff_elements": None,
                   "select_style": 0,
                   "ratio_hot": 0.1, "ratio2": 1.0,
                   "ratio2_style": 0,
                   "norm": "none", "min_norm": 0.02,
                   "ratio_shift": 0.1,
                   "Exclude_types": None, "Enforce_type": None,
                   "Inteval4Enforce": 1,
                   "Exclude_mid": False,
                   "LoopMax": 1000000,
                   "Nsteps4Relax": 100,
                   "Nsteps4Checkpoint": 100,
                   "Nsteps4UpdateEREFs": 1000,
                   "Temperature": 300.0,
                   "Tolerance": 0.001,
        }
        '''
        norm: "auto", "none" or float or list of floats
               auto: 1.0 / max(abs(EREF), min_norm)
               none: 1.0
        ratio2_style: 0, 1, 2
               0: hot and hot
               1: hot and cold
               2: all atoms for 2nd selection
        '''
        MMC = parameters['MetropolisMC']
        if "ntypes" not in MMC:
            errormsg = "There must be a ntypes for # of types in data file!"
            error_exit(errormsg)
        for key in MMC:
            thisMMC[key] = MMC[key]

        if isinstance(thisMMC["Exclude_types"], int):
            pass
        elif isinstance(thisMMC["Exclude_types"], list):
            isvalid = True
            for i in range(len(thisMMC["Exclude_types"])):
                try:
                    thisMMC["Exclude_types"][i] = int(thisMMC["Exclude_types"][i])
                except:
                    isvalid = False
            if not isvalid:
                thisMMC["Exclude_types"] = None
        else:
            thisMMC["Exclude_types"] = None

        if not isinstance(thisMMC["Enforce_type"], int):
            thisMMC["Enforce_type"] = None

        if not isinstance(thisMMC["select_style"], int):
            thisMMC["select_style"] = 0

        if not isinstance(thisMMC["ratio2_style"], int):
            thisMMC["ratio2_style"] = 0

        if not isinstance(thisMMC["ratio2"], float):
            thisMMC["ratio2"] = 1.0

        if thisMMC["ratio2_style"] >= 2:
            thisMMC["ratio2"] = 1.0

        thisvisual = {"SummaryFile": "MMC_Summary.csv", "LogFile": "MMC.log",
                      "Screen": True, "Log": True, "DataOut_Path": "DataOut",
                      "Nsteps4WriteData": 100, "Nsteps4Visual": 10, "Nsteps4Summary": 10,
                      }

        visual = parameters['visual']
        for key in visual:
            thisvisual[key] = thisvisual[key]

        thissett = cls(thissystem, thisPyLAMMPS, thisMMC, thisvisual)
        return thissett