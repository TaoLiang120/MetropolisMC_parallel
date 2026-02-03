import os
import numpy as np
import shutil
from mpi4py import MPI
from MMC.mmc import DFWriter, LOGWriter, PyLMP4MMC, MMC

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()
 
fsummary = "mySummary.csv"
flogfile = "myMMC.log"
SummaryDF = DFWriter(fsummary)
Logfile = LOGWriter(flogfile)

DataOut_Path = "DataOut"
if rank_world == 0:
    if not os.path.isdir(DataOut_Path):
        os.makedirs(DataOut_Path)

EREFs = np.array([-4.81, -3.8, -4.81, -3.8])  ##must be consistent with # of types
ff_elements = np.array(["Fe", "Cr", "Fe", "Cr"])
EREFs = None #np.array([-4.81, -3.8])  ##must be consistent with # of types
ff_elements = np.array(["Fe", "Cr"])
Exclude_types = None #[3, 4]
Enforce_types = None #[1]
ntypes = 2

tol = 0.01
loopmax = 10000
nsteps4relax = 100
nsteps4writedata = 100
nsteps4checkpoint = 100
nsteps4visual = 10
nsteps4summary = 10
nsteps4updateEREFs = 1000
Temperature = 300.0

Screen = False #True
Log = False #"log.lammps"
lmp = PyLMP4MMC(Screen=Screen, Log=Log, comm=comm_world)

infile = "in.MMC"
fdata = "SIL010_r7.5.dat"
ratio_hot = 0.03
ratio_cold = 1.0
reverse_cold = True
mydata = MMC(ntypes, EREFs=EREFs, ff_elements=ff_elements,
             ratio_hot=ratio_hot, ratio_cold=ratio_cold, reverse_cold=reverse_cold)
if rank_world == 0:
    shutil.copy(fdata, os.path.join(DataOut_Path, "MMC0.dat"))
comm_world.Barrier()

iloop = 0
iaccept = 0
ireject = 0
lmp.excute_file(infile)
mydata.last_TE, mydata.last_types = lmp.get_total_energy_types(iloop)
mydata.natoms = len(mydata.last_types)
eatoms = lmp.get_eatoms(iloop, mydata.natoms)
mydata.update_EREFs(mydata.last_types, eatoms)
init_energy = mydata.last_TE
energy_checkpoint = init_energy

if rank_world == 0:
    ratio_shift = 0.01
    atom_style = "atomic"
    filein = MMC + str(iloop) + ".dat"
    mydata.write_shifted_data(mydata.last_types, eatoms, ratio_shift, filein, atom_style=atom_style)

    logstr = f"loopmax:{loopmax} Temp:{Temperature} convergenc: energy < {tol} in {nsteps4checkpoint} steps"
    print(logstr)
    Logfile.write_to_file(logstr, open_style="w")

    logstr = f"Reference energies for {nstypes} types at {iloop} step are :{mydata.EREFs}"
    print(logstr)
    Logfile.write_to_file(logstr, open_style="a")

    logstr = f"start MMC for fname:{fdata} natoms:{mydata.natoms} ratio_hot:{ratio_hot} ratio_cold: {ratio_cold}"
    print(logstr)
    Logfile.write_to_file(logstr, open_style="a")

    logstr = f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
    SummaryDF.append_to_file(iloop, iaccept, ireject, init_energy)
    print(logstr)
    Logfile.write_to_file(logstr, open_style="a")

isValid = True
while isValid:
    if rank_world == 0:
        id_hot, id_cold = mydata.get_select_ids(mydata.last_types, eatoms, Exclude_types=Exclude_types, Enforce_types=Enforce_types)
        this_types = mydata.get_this_types(id_hot, id_cold)
    else:
        this_types = None
    comm_world.Barrier()

    this_types = comm_world.bcast(this_types, root=0)
    lmp.scatter_this_types(this_types)
    iloop  += 1
    mydata.this_TE, mydata.this_types = lmp.get_total_energy_types(iloop)
    eatoms = lmp.get_eatoms(iloop, mydata.natoms)

    if iloop % nsteps4updateEREFs == 0:
        mydata.update_EREFs(mydata.this_types, eatoms)
        if rank_world == 0:
            logstr = f"Reference energies for {nstypes} types at {iloop} step are :{mydata.EREFs}"
            print(logstr)

    if rank_world == 0:
        isAccept, iaccept, ireject = mydata.MMC(iaccept, ireject, Temp=Temperature)
    else:
        isAccept = None
    comm_world.Barrier()
    isAccept = comm_world.bcast(isAccept, root=0)
    if not isAccept:
        lmp.scatter_this_types(mydata.last_types)

    if iloop % nsteps4writedata == 0:
        lmp.write_data(iloop)

    if iloop % nsteps4summary == 0 and rank_world == 0:
        SummaryDF.append_to_file(iloop, iaccept, ireject, mydata.last_TE)

    if iloop % nsteps4visual == 0 and rank_world == 0:
        logstr = f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
        print(logstr)
        Logfile.write_to_file(logstr, open_style="a")
    
    if rank_world == 0:
        isValid, energy_checkpoint = mydata.checkpoint(iloop, isValid, energy_checkpoint,
                                                   nsteps4checkpoint=nsteps4checkpoint, tol=tol,
                                                   loopmax=loopmax)
    else:
        isValid = None
        energy_checkpoint = None
    comm_world.Barrier()
    isValid = comm_world.bcast(isValid, root=0)
    energy_checkpoint = comm_world.bcast(energy_checkpoint, root=0)
    comm_world.Barrier()

lmp.write_data(iloop)
lmp.close()
logstr = f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
if rank_world == 0:
    print(logstr)
    Logfile.write_to_file(logstr, open_style="a")
    SummaryDF.append_to_file(iloop, iaccept, ireject, mydata.last_TE)

