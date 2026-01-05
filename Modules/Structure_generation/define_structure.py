from Modules.Settings.settings import *
from Modules.Structure_generation.layer_type import layer_type
from Modules.Structure_generation.update_substrate import update_substrate
from Modules.Structure_generation.add_gas import add_gas

def define_structure():
    db={}
    #Initializing Structure
    structure = torch.zeros((GRID_W, GRID_L, 0), dtype=torch.int8).to(device)
    for i in range(len(LAYER_NAMES)):
        db=layer_type(LAYER_NAMES[i],MATERIAL_NAMES[i],LAYER_THICKNESS[i],LAYER_THRESHOLD[i],db)
        structure=update_substrate(LAYER_NAMES[i],structure,db)


    structure=add_gas(structure)
    return structure,db
