from Modules.Settings.settings import *

def add_gas(structure):
    structure=torch.dstack((structure, torch.zeros((structure.shape[0], structure.shape[1], GAS_HEIGHT), dtype=torch.int8).to(device)))
    
    return structure
 