from Modules.Settings.settings import *
from Modules.Surface_voxels.exposed_faces import exposed_faces
def deposition_slots(base_voxels, flag):

    base_voxels[flag==1,0] = base_voxels[flag==1,0]-1
    base_voxels[flag==2,0] = base_voxels[flag==2,0]+1
    base_voxels[flag==3,1] = base_voxels[flag==3,1]-1
    base_voxels[flag==4,1] = base_voxels[flag==4,1]+1
    base_voxels[flag==5,2] = base_voxels[flag==5,2]-1
    base_voxels[flag==6,2] = base_voxels[flag==6,2]+1

    return base_voxels