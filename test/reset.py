from mylammps.inputs.data import lmpData
import numpy as np
fname = "FeCr5.dat"
data = lmpData.from_file(fname, "atomic")
data.reset_atom_ids()
data.insert_molecular_id()
data.atoms["molecule-ID"] = np.hstack([np.zeros(len(data.atoms)-6000, dtype=int), -np.ones(6000, dtype=int)])
types = data.atoms["type"].to_numpy()
types[types==2] = 3
data.atoms["type"] = types
data.to_file("FeCr5_reset.dat")