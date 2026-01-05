from Modules.Settings.settings import *

from Modules.Surface_voxels.exposed_faces import exposed_faces
from Modules.Motion.Particle_structure_interaction.clear_faces import clear_faces
import time as timex

def update_threshold(structure,top_voxels, idx, faces,xt,yt,zt):

    # print('structure[2,3,9], up',structure[2,3,9])
    remove_voxel=(top_voxels[idx])                                              #Coordinates of voxel to be removed
    remove_face=faces[idx]   
                                                                                 #Exposed faces of voxel to be removed
    mask = torch.ones(top_voxels.shape[0], dtype=torch.bool)
    mask[idx] = False

    top_voxels=top_voxels[mask,:]                                                   #Remove the voxel from list of surface voxels
    # threshold=threshold[mask]                                                       #Remove threshold of removed voxel from list of threshold
    # hit=hit[mask]                                                                   #Remove number of hits of removed voxel from list of hits
    faces=faces[mask,:]                                                             #Remove faces of removed voxel from list of faces
              
    faces, top_voxels =clear_faces(remove_face, remove_voxel, top_voxels, faces, structure, xt, yt,zt)  #Update exposed faces of neighbours of voxel to be removed


    return  top_voxels, faces