from Modules.Settings.settings import *



# Function for Layer Definitions
def layer_type(name, material, thickness, mass,db):

  V_th=int(mass)
  props={'Material':material,'Thickness':thickness, 'Mass':mass, 'Voxel_Threshold':V_th, 'z_start':0,'z_end':0}
  if name not in db.keys():
    db[name]=props
  return db