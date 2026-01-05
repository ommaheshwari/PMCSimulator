from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.compare_array import compare_array
from Modules.Surface_voxels.exposed_faces import exposed_faces


def add_neighbour(top_voxels, add_voxel, structure, faces):

    xtl=add_voxel[:,0].type(torch.int).reshape(-1)
    ytl=add_voxel[:,1].type(torch.int).reshape(-1)
    ztl=add_voxel[:,2].type(torch.int).reshape(-1)
    top_voxels=torch.cat((top_voxels,add_voxel))                                #Add neighbour to surface voxels
    # hit=torch.cat((hit,torch.zeros(add_voxel.shape[0]).to(device)))                              #Add 0 hits for neighbour in list of hits
    # new_th=(totalth[0,xtl,ytl,ztl]).type(torch.int).to(device)
    # threshold = torch.cat((threshold,new_th))          #Add new threshold to list of thresholds
    exp_faces=exposed_faces(add_voxel,structure)    #Exposed faces of neighbour
    faces = torch.cat((faces,exp_faces))                                        #Add new exposed faces to list of faces of surface voxels
    return top_voxels, faces