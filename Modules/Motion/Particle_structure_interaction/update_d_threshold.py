from Modules.Settings.settings import *

from Modules.Surface_voxels.exposed_faces import exposed_faces
from Modules.Motion.Particle_structure_interaction.clear_d_faces import clear_d_faces
from Modules.Motion.Particle_structure_interaction.add_neighbour import add_neighbour
import time as timex

def update_d_threshold(structure,top_voxels,add_voxel, faces,xt,yt,zt):


 
                                                                                 #Exposed faces of voxel to be removed

    top_voxels=torch.cat((top_voxels,add_voxel))                                #Add neighbour to surface voxels

    exp_faces=exposed_faces(add_voxel,structure)    #Exposed faces of neighbour
    
    faces = torch.cat((faces,exp_faces))         

    faces, top_voxels =clear_d_faces(exp_faces, add_voxel, top_voxels, faces, structure, xt, yt,zt)  #Update exposed faces of neighbours of voxel to be removed

    del add_voxel, exp_faces, xt,yt,zt
    torch.cuda.empty_cache()


    return top_voxels, faces