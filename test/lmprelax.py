from lammps import lammps
infile = "in.relax"
lmp = lammps()
lmp.file(infile)
lmp.close()
