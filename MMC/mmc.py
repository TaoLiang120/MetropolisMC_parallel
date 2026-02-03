import os
import numpy as np
import pandas as pd
import copy
import ctypes
from mpi4py import MPI

from lammps import lammps

from pymatgen.io.lammps.data import ATOMS_HEADERS
from pymatgen.io.lammps.data import LammpsBox, LammpsData, lattice_2_lmpbox
from pymatgen.io.lammps.outputs import parse_lammps_dumps

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
    def __init__(self, fname):
        self.fname = fname

    def write_to_file(self, thisstring, open_style="a"):
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
        return etotal, types

    @staticmethod
    def parse_dumps(fname, natoms):
        isValid = True
        while isValid:
            with open(fname, "r") as fh:
                lines = fh.readlines()
            if len(lines) >= natoms + 8 + 1:
                isValid = False
        lines[8] = lines[8].replace("ITEM: ATOMS ", "")
        with open("../test/tmp.csv", "w") as fh:
            fh.writelines(lines[8:natoms + 8 + 1])
        df = pd.read_csv("../test/tmp.csv", sep=" ")
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
    def __init__(self, ntypes, EREFs=None, ff_elements=None, ratio_hot=0.1, ratio_cold=0.5, reverse_cold=True):
        self.kB = 8.617333262145e-5
        self._natoms = 1
        self.ntypes = ntypes
        if EREFs is None:
            self.EREFs = np.zeros(ntypes, dtype=float)
        if ff_elements is None:
            ff_elements = np.arange(self.ntypes, dtype=int) + 1
        self.ff_elements = ff_elements
        self.this_types = np.ones(self._natoms, dtype=int)
        self.last_types = np.ones(self._natoms, dtype=int)
        self.this_TE = 0.0
        self.last_TE = 0.0
        self.ratio_hot = ratio_hot
        self.ratio_cold = ratio_cold
        self.reverse_cold = reverse_cold

    @property
    def natoms(self):
        return self._natoms

    @natoms.setter
    def natoms(self, value):
        if value >= 1:
            self._natoms = int(value)

    def update_EREFs(self, types, eatoms):
        apptypes = types.astype(int) - 1
        inds = np.arange(len(eatoms))
        EREFs = np.zeros(self.ntypes, dtype=float)
        for i in range(self.ntypes):
            thisinds = np.compress(apptypes == i, inds)
            if len(thisinds) > 0:
                thiseatoms = eatoms[thisinds]
                EREFs[i] = np.sum(thiseatoms)/len(thiseatoms)
        self.EREFs = EREFs     

    def write_shifted_data(self, types, eatoms, ratio_shift, filein, atom_style="atomic"):
        lmpdata = LammpsData.from_file(os.path.join("DataOut", filein), atom_style, sort_id=True)
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
        lmpdata.write_file(os.path.join("DataOut", filein + "_shifted.dat"))

    def get_select_ids(self, types, eatoms, Exclude_types=None, Enforce_types=None, maxloop=100):
        if isinstance(Enforce_types, int):
            Enforce_types = [Enforce_types]      
        apptypes = types - 1
        earefs = self.EREFs[apptypes]
        eadiff = eatoms - earefs
        if isinstance(Exclude_types, int):
            Exclude_types = [Exclude_types]
        if isinstance(Exclude_types, list) or isinstance(Exclude_types, np.ndarray):
            Exclude_types = np.array(Exclude_types).astype(int)
            Exclude_types -= 1
            local_apptypes = types - 1
            goodinds = np.arange(len(eadiff), dtype=int)
            for i in range(len(Exclude_types)):
                thistype = Exclude_types[i]
                local_goods = np.arange(len(eadiff), dtype=int)
                local_goods = np.compress(local_apptypes != thistype, local_goods)
                goodinds = goodinds[local_goods]
                local_apptypes = local_apptypes[local_goods]
                eadiff = eadiff[local_goods]
        else:
            goodinds = np.arange(len(eadiff), dtype=int)
        natoms4mmc = len(eadiff)
        inds = np.argsort(eadiff)
        iratio_hot = int(natoms4mmc*self.ratio_hot)
        if iratio_hot < 2:
            raise ValueError("System is too small for iratio_hot. Set iratio_hot to 1.0 and rerun it.")

        isValid = False
        iloop = 0
        while not isValid:
            sid_hot = np.random.randint(iratio_hot, size=1)
            sid_hot = natoms4mmc - sid_hot[0] - 1
            sid_hot = inds[sid_hot]
            sid_hot = goodinds[sid_hot]
            type_hot = apptypes[sid_hot]
            if Enforce_types is None:
                isValid = True
            else:
                if type_hot + 1 in Enforce_types:
                    isValid = True
            if iloop > maxloop:
                isValid = True
            iloop += 1
        sym_hot = self.ff_elements[type_hot]
 
        syms_cold = self.ff_elements[apptypes[goodinds]]
        syms_cold = syms_cold[inds]
        inds_cold = np.compress(syms_cold != sym_hot, inds)
        iratio_cold = int(len(inds_cold)*self.ratio_cold)
        if iratio_cold < 2:
            raise ValueError("System is too small for iratio_cold. Set iratio_cold to 1.0 and rerun it.") 
        sid_cold_local = np.random.randint(iratio_cold, size=1)
        if self.reverse_cold:
            sid_cold_local = len(inds_cold) - sid_cold_local[0] - 1
        else:
            sid_cold_local = sid_cold_local[0]
        sid_cold_in_inds = inds_cold[sid_cold_local]
        sid_cold = goodinds[sid_cold_in_inds]
        type_cold = apptypes[sid_cold]
        sym_cold = self.ff_elements[type_cold]
        #print(f"sym_hot:{sym_hot}   type_hot:{type_hot}   e_hot: {eatoms[sid_hot]}")
        #print(f"sym_cold:{sym_cold} type_cold:{type_cold} e_cold:{eatoms[sid_cold]}")
        #print("----")
        return sid_hot, sid_cold

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

