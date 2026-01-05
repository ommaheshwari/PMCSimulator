from Modules.Settings.settings import *


# Function for adding new defined layer onto the substrate
def update_substrate(name,structures,db):
  new_layer_thickness=int(db[name]['Thickness']*VOXEL_PER_UNIT)
  layers=list(db.keys())
  new_layer_ = (len(layers))*torch.ones((GRID_W, GRID_L, new_layer_thickness), dtype=torch.int8).to(device)
  subs_update= torch.dstack((structures, new_layer_)).to(device)
  # new_layer_coords= subs_update-structure
  db[name]['z_start']=structures.shape[2]
  db[name]['z_end']=structures.shape[2]+new_layer_thickness -1

  return subs_update
