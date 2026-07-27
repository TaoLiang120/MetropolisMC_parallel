import os
from logging import warning

import numpy as np
import pandas as pd
import copy
import ctypes
from mpi4py import MPI

from lammps import lammps

from pymatgen.io.lammps.data import ATOMS_HEADERS
from pymatgen.io.lammps.data import LammpsBox, LammpsData, lattice_2_lmpbox
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from MMC.error_exit import error_exit

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

class DFWriter:
    def __init__(self, fname):
        self.fname = fname
        self.cols = ["iloop", "iaccept", "ireject", "total_energy"]
        self.df = pd.DataFrame(columns=self.cols)
        self.to_file()

    def append_to_file(self, iloop, iaccept, ireject, total_energy):
        thisdict = {"iloop": iloop, "iaccept": iaccept, "ireject": ireject, "total_energy": total_energy}
        #self.df.loc[len(self.df)] = thisdict
        thisstr = str(iloop)+","+str(iaccept)+","+str(ireject)+","+str(total_energy)
        with open(self.fname, "a") as f:
            f.write(thisstr + "\n")

    def to_file(self):
        self.df.to_csv(self.fname, index=False)
    
class LOGWriter:
    def __init__(self, fname, screen=True, log=True):
        self.fname = fname
        self.screen = screen
        self.log = log

    def write_to_file(self, thisstring, open_style="a"):
        if self.screen:
            print(thisstring)
        if self.log:
            with open(self.fname, open_style) as fh:
                fh.write(thisstring + "\n")

class PyLMP4MMC:
    def __init__(self, Screen=False, Log=False, comm=None):
        args = []
        if not Screen: args = ["-screen", "none"]
        if isinstance(Log, str):
            if Log.upper() == "NONE":
                Log = "none"
            else:
                pass
        else:
            Log = "none"
        args += ["-log", Log]
        if comm is None:
            comm = comm_world
        self.bin = lammps(cmdargs=args, comm=comm)

    def excute_lines(self, lines):
        for line in lines:
            self.bin.command(line)

    def excute_file(self, fname):
        with open(fname, "r") as f:
            lines = f.readlines()
        self.excute_lines(lines)

    def get_coordinates(self):
        coords = self.gather_atoms("x", 1, 3)
        coords = np.ctypeslib.as_array(coords)
        return coords

    def get_total_energy_types(self, iloop, nsteps4relax=1000):
        if iloop % nsteps4relax == 0:
            #pass
            self.bin.command("minimize     0.0 1.0e-6 10000   100000")
        self.bin.command("run 0")
        etotal = self.bin.get_thermo("etotal")
        types = self.bin.gather_atoms("type", 0, 1)
        types = np.ctypeslib.as_array(types)
        try:
            molids = self.bin.gather_atoms("molecule", 0, 1)
            molids = np.ctypeslib.as_array(molids)
        except:
            molids = None
        return etotal, types, molids

    @staticmethod
    def parse_dumps(fname, natoms):
        isValid = True
        while isValid:
            with open(fname, "r") as fh:
                lines = fh.readlines()
            if len(lines) >= natoms + 8 + 1:
                isValid = False
        lines[8] = lines[8].replace("ITEM: ATOMS ", "")
        with open("tmp.csv", "w") as fh:
            fh.writelines(lines[8:natoms + 8 + 1])
        df = pd.read_csv("tmp.csv", sep=" ")
        return df
             
    def get_eatoms(self, iloop, natoms):
        self.bin.command("dump mydump all custom 1 thisdump id c_eatoms")
        self.bin.command("dump_modify mydump sort id")
        self.bin.command("run 0")
        eatoms = None
        if rank_world == 0:
            while os.path.isfile("thisdump"):
                df = PyLMP4MMC.parse_dumps("thisdump", natoms)
                eatoms = df["c_eatoms"].to_numpy()
                os.remove("thisdump")
            if eatoms is None:
                raise ValueError(f"Cannot extract energy per atoms at {iloop} loop!")
        else:
            eatoms = None
        comm_world.Barrier()
        eatoms = comm_world.bcast(eatoms, root=0)
        self.bin.command("undump mydump")
        return eatoms

    def scatter_this_types(self, this_types):
        c_int_p = ctypes.POINTER(ctypes.c_uint)
        x_p = this_types.ctypes.data_as(c_int_p)
        self.bin.scatter_atoms("type", 0, 1, x_p)
   
    def write_data(self, iloop, DataOut="DataOut"):
        self.bin.command("run 0")
        self.bin.command("write_data  " +  DataOut+ "/MMC" + str(iloop) + ".dat")

    def close(self):
        self.bin.close()

class MMC:
    def __init__(self, ntypes, EREFs=None, ff_elements=None, ratio_hot1=0.1, ratio_hot2=0.2,
                 norm="auto", min_norm=0.02):
        self.kB = 8.617333262145e-5
        self._natoms = 1
        self.ntypes = ntypes
        if EREFs is None:
            self.EREFs = np.zeros(ntypes, dtype=float)
        if ff_elements is None:
            ff_elements = np.arange(self.ntypes, dtype=int) + 1
        self.ff_elements = np.array(ff_elements)
        self.this_types = np.ones(self._natoms, dtype=int)
        self.last_types = np.ones(self._natoms, dtype=int)
        self.this_TE = 0.0
        self.last_TE = 0.0
        self.ratio_hot1 = ratio_hot1
        self.ratio_hot2 = ratio_hot2
        if isinstance(self.ratio_hot1, float):
            self.ratio_hot1 = [self.ratio_hot1] * ntypes
        if isinstance(self.ratio_hot2, float):
            self.ratio_hot2 = [self.ratio_hot2] * ntypes
        self.norm = norm

        if isinstance(self.norm, float):
            self.norm = [self.norm] * ntypes
        elif self.norm is None:
            self.norm = ["none"] * ntypes
        elif self.norm.lower() == "none":
            self.norm = ["none"] * ntypes
        else:
            self.norm = ["auto"] * ntypes
        self.min_norm = min_norm


    @property
    def natoms(self):
        return self._natoms

    @natoms.setter
    def natoms(self, value):
        if value >= 1:
            self._natoms = int(value)

    def update_EREFs(self, types, eatoms, molids=None, Exclude_mid=False):
        apptypes = types.astype(int) - 1
        eatoms_mols = copy.deepcopy(eatoms)
        inds = np.arange(len(eatoms_mols)).astype(int)
        if Exclude_mid and molids is not None:
            inds = np.compress(molids >= 0, inds)
            eatoms_mols = eatoms_mols[inds]
            apptypes = apptypes[inds]
            inds = np.arange(len(eatoms_mols)).astype(int)

        EREFs = np.zeros(self.ntypes, dtype=float)
        for i in range(self.ntypes):
            thisinds = np.compress(apptypes == i, inds)
            if len(thisinds) > 0:
                thiseatoms = eatoms_mols[thisinds]
                EREFs[i] = np.sum(thiseatoms)/len(thiseatoms)
        self.EREFs = EREFs

    def write_shifted_data(self, types, eatoms, ratio_shift, filein, atom_style="atomic", DataOut_Path="DataOut"):
        lmpdata = LammpsData.from_file(os.path.join(DataOut_Path, filein), atom_style, sort_id=True)
        ff_elements = np.append(self.ff_elements, self.ff_elements)
        force_field = {}
        for i in range(len(ff_elements)):
            force_field[str(i + 1)] = ff_elements[i]
        lmpdata.masses = pd.concat([lmpdata.masses, lmpdata.masses])
        lmpdata.masses = lmpdata.masses.reset_index().set_index(np.arange(len(lmpdata.masses), dtype=int) + 1)
        if "index" in lmpdata.masses.columns:  lmpdata.masses.drop(columns=["index"])
        apptypes = types - 1
        earefs = self.EREFs[apptypes]
        eadiff = eatoms - earefs
        natoms4mmc = len(eadiff)
        inds = np.argsort(-eadiff)
        iratio = int(natoms4mmc*ratio_shift)
        sorted_types = types[inds]
        hot_types = sorted_types[0:iratio] + self.ntypes
        sorted_types = np.append(hot_types, sorted_types[iratio:natoms4mmc])
        sinds = np.argsort(inds)
        shifted_types = sorted_types[sinds]
        lmpdata.atoms['type'] = shifted_types
        lmpdata.write_file(os.path.join(DataOut_Path, filein + "_shifted.dat"))

    def get_select_ids(self, iloop, types, eatoms, Exclude_types=None,
                       Enforce_types=None, Inteval4Enforce=1,
                       molids=None, Exclude_mid=False):
        if isinstance(Enforce_types, int):
            Enforce_types = [Enforce_types]
        elif isinstance(Enforce_types, str):
            Enforce_types = None
        if isinstance(Exclude_types, int):
            Exclude_types = [Exclude_types]
        elif isinstance(Exclude_types, str):
            Exclude_types = None

        apptypes = types - 1
        earefs = self.EREFs[apptypes]
        eadiff = eatoms - earefs

        inds = np.arange(len(eadiff)).astype(int)
        inds_mols = np.arange(len(eadiff)).astype(int)
        if Exclude_mid and molids is not None:
            inds_mols = np.compress(molids >= 0, inds)
        eadiff_mols = eadiff[inds_mols]
        apptypes = apptypes[inds_mols]
        inds = np.arange(len(eadiff_mols)).astype(int)

        inds_type = []
        local_inds_type = []
        maxdiff_type = []
        natype = []
        for i in range(self.ntypes):
            thisnorm = self.norm[i]
            thisref = self.EREFs[i]
            if isinstance(thisnorm, str) and thisnorm.lower() == "none":
                thisnorm = 1.0
            elif isinstance(thisnorm, str) and thisnorm.lower() == "auto":
                thisnorm = max(abs(thisref), self.min_norm)
            if thisnorm is None:
                thisnorm = 1.0
            else:
                continue

            thisinds = np.compress(apptypes == i, inds)
            if len(thisinds) > 0:
                thiseadiff = eadiff_mols[thisinds]
                local_inds = np.argsort(thiseadiff)[::-1]
                thismax = thiseadiff[local_inds[0]] / thisnorm
            else:
                thisinds = np.array([]).astype(int)
                local_inds = np.array([])
                thismax = 0.0

            inds_type.append(thisinds)
            local_inds_type.append(local_inds)
            maxdiff_type.append(thismax)
            natype.append(len(thisinds))

        maxdiff_type = np.array(maxdiff_type)
        natype = np.array(natype).astype(int)
        id_type = np.arange(self.ntypes).astype(int)

        thisExclude_types = []
        for i in range(self.ntypes):
            if natype[i] == 0:
                thisExclude_types.append(i+1)
                print(f"WARNING: No atoms in type {i+1}!")
        if Exclude_types is None:
            pass
        else:
            thisExclude_types += Exclude_types
        if len(thisExclude_types) > 0:
            id_type = np.delete(id_type, np.array(thisExclude_types).astype(int) - 1)
            maxdiff_type = maxdiff_type[id_type]
            natype = natype[id_type]

        if len(id_type) < 2:
            raise ValueError("System is has less than two types for MMMC.")

        if Enforce_types is not None:
            if iloop % Inteval4Enforce != 0:
                id1_type = np.array(Enforce_types).astype(int) - 1
                local_typeid1 = np.random.randint(len(id1_type), size=1)[0]
                typeid1 = id_type[local_typeid1]
                if natype[typeid1] < 2:
                    local_typeid1 = np.argmax(np.array(maxdiff_type))
                    typeid1 = id_type[local_typeid1]
            else:
                local_typeid1 = np.argmax(np.array(maxdiff_type))
                typeid1 = id_type[local_typeid1]
        else:
            local_typeid1 = np.argmax(np.array(maxdiff_type))
            typeid1 = id_type[local_typeid1]

        sym1 = self.ff_elements[typeid1]
        thisratio1 = self.ratio_hot1[typeid1]
        natoms1 = len(local_inds_type[typeid1])
        iratio1 = int(natoms1*thisratio1)
        if iratio1 < 2:
            raise ValueError("System is too small for iratio first pick. Set iratio_hot1 to 1.0 and rerun it.")
        local_sid1 = np.random.randint(iratio1, size=1)[0]
        global_sid1 = local_inds_type[typeid1][local_sid1]
        sid1 = inds_type[typeid1][global_sid1]
        sid1 = inds_mols[sid1]


        id_type = np.delete(id_type, [typeid1])
        local_typeid2 = np.random.randint(len(id_type), size=1)[0]
        typeid2 = id_type[local_typeid2]
        sym2 = self.ff_elements[typeid2]
        thisratio2 = self.ratio_hot2[typeid2]
        natoms2 = len(local_inds_type[typeid2])
        iratio2 = int(natoms2*thisratio2)
        if iratio2 < 2:
            raise ValueError("System is too small for iratio second pick. Set iratio_hot2 to 1.0 and rerun it.")
        local_sid2 = np.random.randint(iratio2, size=1)[0]
        global_sid2 = local_inds_type[typeid2][local_sid2]
        sid2 = inds_type[typeid2][global_sid2]
        sid2 = inds_mols[sid2]

        #print(f"sym1:{sym1} typeid1:{typeid1} sid1:{sid1} e_1:{eatoms[sid1]}")
        #print(f"sym2:{sym2} typeid2:{typeid2} sid2:{sid2} e_2:{eatoms[sid2]}")
        #print("----")
        return sid1, sid2

    def get_this_types(self, id_hot, id_cold):
        self.this_types = copy.deepcopy(self.last_types)
        itmp = self.this_types[id_hot]
        self.this_types[id_hot] = self.this_types[id_cold]
        self.this_types[id_cold] = itmp
        return self.this_types

    def MMC(self, iaccept, ireject, Temp=800):
        ediff = self.this_TE - self.last_TE
        if ediff <= 0:
            isAccept = True
        else:
            p = np.exp(-ediff/self.kB/Temp)
            r = np.random.rand(1)
            if r[0] < p:
                isAccept = True
            else:
                isAccept = False

        if isAccept:
            self.last_TE = self.this_TE
            self.last_types = copy.deepcopy(self.this_types)
            #self.last_eatoms = copy.deepcopy(self.this_eatoms)
            iaccept += 1
        else:
            ireject += 1
        return isAccept, iaccept, ireject

    def checkpoint(self, iloop, isValid, energy_checkpoint,
                   nsteps4checkpoint=100, tol=0.01,
                   loopmax=2000000):
        if iloop % nsteps4checkpoint == 0:
            ediff = energy_checkpoint - self.this_TE
            if abs(ediff) < tol:
                isValid = False
            energy_checkpoint = self.this_TE

        if iloop >= loopmax:
            isValid = False
        return isValid, energy_checkpoint

