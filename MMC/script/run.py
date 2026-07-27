import os
import copy
import numpy as np
import shutil
from mpi4py import MPI
from MMC.read_input import Settings
from MMC.mmc import DFWriter, LOGWriter, PyLMP4MMC, MMC

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

def main():
    finput4mmc = "inputMMC.yaml"
    thissett = Settings.from_file(finput4mmc)
    SummaryDF = DFWriter(thissett.visual["SummaryFile"])
    Logfile = LOGWriter(thissett.visual["LogFile"], screen=thissett.visual["Screen"], log=thissett.visual["Log"])

    DataOut_Path = thissett.visual["DataOut_Path"]
    if rank_world == 0:
        if not os.path.isdir(DataOut_Path):
            os.makedirs(DataOut_Path)
    lmp = PyLMP4MMC(Screen=thissett.PyLAMMPS["Screen"], Log=thissett.PyLAMMPS["Log"])

    infile = thissett.PyLAMMPS["Input4LAMMPS"]
    fdata = thissett.PyLAMMPS["FileName"]

    MMCsetts = copy.deepcopy(thissett.MetropolisMC)
    ntypes = MMCsetts["ntypes"]
    EREFs = MMCsetts["EREFs"]
    ff_elements = MMCsetts["ff_elements"]
    ratio_hot1 = MMCsetts["ratio_hot1"]
    ratio2 = MMCsetts["ratio2"]
    select_style = MMCsetts["select_style"]
    norm = MMCsetts["norm"]
    min_norm = MMCsetts["min_norm"]

    mydata = MMC(ntypes, EREFs=EREFs, ff_elements=ff_elements,
                 ratio_hot1=ratio_hot1, ratio2=ratio2, select_style=select_style,
                 norm=norm, min_norm=min_norm)

    if rank_world == 0:
        shutil.copy(fdata, os.path.join(DataOut_Path, "MMC0.dat"))
    comm_world.Barrier()

    atom_style = thissett.PyLAMMPS["atom_style"]
    loopmax = MMCsetts["LoopMax"]
    ratio_shift = MMCsetts["ratio_shift"]
    Exclude_types = MMCsetts["Exclude_types"]
    Enforce_types = MMCsetts["Enforce_types"]
    Inteval4Enforce = MMCsetts["Inteval4Enforce"]
    Exclude_mid = MMCsetts["Exclude_mid"]
    Nsteps4Checkpoint = MMCsetts["Nsteps4Checkpoint"]
    Temperature = MMCsetts["Temperature"]
    tol = MMCsetts["Tolerance"]
    Nsteps4UpdateEREFs = MMCsetts["Nsteps4UpdateEREFs"]

    visualsett = copy.deepcopy(thissett.visual)
    Nsteps4Visual = visualsett["Nsteps4Visual"]
    Nsteps4Summary = visualsett["Nsteps4Summary"]
    Nsteps4WriteData = visualsett["Nsteps4WriteData"]

    iloop = 0
    iaccept = 0
    ireject = 0
    first_types = [0] * ntypes
    second_types = [0] * ntypes
    first_accept = [0] * ntypes
    second_accept = [0] * ntypes

    lmp.excute_file(infile)
    mydata.last_TE, mydata.last_types, molids = lmp.get_total_energy_types(iloop)
    mydata.natoms = len(mydata.last_types)
    eatoms = lmp.get_eatoms(iloop, mydata.natoms)
    mydata.update_EREFs(mydata.last_types, eatoms, molids=molids, Exclude_mid=Exclude_mid)
    init_energy = mydata.last_TE
    energy_checkpoint = init_energy
    if rank_world == 0:
        logstr = f"loopmax:{loopmax} Temp:{Temperature} convergenc: energy < {tol} in {Nsteps4Checkpoint} steps"
        Logfile.write_to_file(logstr, open_style="w")
        logstr = f"start MMC for fname:{fdata} natoms:{mydata.natoms} "
        Logfile.write_to_file(logstr, open_style="a")
        logstr = f"ratio_hot1:{ratio_hot1} ratio2: {ratio2} select_style: {select_style}"
        Logfile.write_to_file(logstr, open_style="a")

        logstr = f"-- Reference energies of each type at {iloop} step are :{mydata.EREFs} --"
        logstr += "\n" + f"-- first_types:{first_types} second_types:{second_types} --"
        logstr += "\n" + f"-- first_accept:{first_accept} second_accept:{second_accept} --"
        logstr += "\n" + f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
        Logfile.write_to_file(logstr, open_style="a")
        SummaryDF.append_to_file(iloop, iaccept, ireject, init_energy)

    isValid = True
    while isValid:
        if rank_world == 0:
            id_1, id_2, typeid1, typeid2 = mydata.get_select_ids(iloop, mydata.last_types, eatoms, Exclude_types=Exclude_types,
                                                    Enforce_types=Enforce_types, Inteval4Enforce=Inteval4Enforce,
                                                    molids=molids, Exclude_mid=Exclude_mid)

            first_types[typeid1] += 1
            second_types[typeid2] += 1
            this_types = mydata.get_this_types(id_1, id_2)
        else:
            this_types = None
        comm_world.Barrier()

        this_types = comm_world.bcast(this_types, root=0)
        lmp.scatter_this_types(this_types)
        iloop += 1
        mydata.this_TE, mydata.this_types, molids = lmp.get_total_energy_types(iloop)
        eatoms = lmp.get_eatoms(iloop, mydata.natoms)

        if iloop % Nsteps4UpdateEREFs == 0:
            mydata.update_EREFs(mydata.this_types, eatoms)

        if rank_world == 0:
            isAccept, iaccept, ireject = mydata.MMC(iaccept, ireject, Temp=Temperature)
            if isAccept:
                first_accept[typeid1] += 1
                second_accept[typeid2] += 1
        else:
            isAccept = None
        comm_world.Barrier()
        isAccept = comm_world.bcast(isAccept, root=0)
        if not isAccept:
            lmp.scatter_this_types(mydata.last_types)

        if iloop % Nsteps4WriteData == 0:
            lmp.write_data(iloop)

        if iloop % Nsteps4Summary == 0 and rank_world == 0:
            SummaryDF.append_to_file(iloop, iaccept, ireject, mydata.last_TE)

        if iloop % Nsteps4Visual == 0 and rank_world == 0:
            logstr = f"-- Reference energies of each type at {iloop} step are :{mydata.EREFs} --"
            logstr += "\n" + f"-- first_types:{first_types} second_types:{second_types} --"
            logstr += "\n" + f"-- first_accept:{first_accept} second_accept:{second_accept} --"
            logstr += "\n" + f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
            Logfile.write_to_file(logstr, open_style="a")

        if rank_world == 0:
            isValid, energy_checkpoint = mydata.checkpoint(iloop, isValid, energy_checkpoint,
                                                           nsteps4checkpoint=Nsteps4Checkpoint, tol=tol,
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
    logstr = f"-- Reference energies of each type at {iloop} step are :{mydata.EREFs} --"
    logstr += "\n" + f"-- first_types:{first_types} second_types:{second_types} --"
    logstr += "\n" + f"-- first_accept:{first_accept} second_accept:{second_accept} --"
    logstr += "\n" + f"== iloop:{iloop} iaccept:{iaccept} ireject: {ireject} total_energy:{mydata.last_TE} =="
    if rank_world == 0:
        Logfile.write_to_file(logstr, open_style="a")
        SummaryDF.append_to_file(iloop, iaccept, ireject, mydata.last_TE)


if __name__ == '__main__':
    main()
