from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.add_neighbour import add_neighbour
from Modules.Motion.Particle_structure_interaction.update_neighbour import update_neighbour
def clear_faces(remove_face, remove_voxel, top_voxels, faces, structure, xt, yt,zt):
    #Check which neighbors are present in structure of the voxel to be removed (remove_face==0 means neighbor is present)

    left_idx=((~remove_face[:,0].type(torch.int8)) & ((remove_voxel[:,0]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    right_idx=((~remove_face[:,1].type(torch.int8)) & ((remove_voxel[:,0]<(structure.shape[0]-1)).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    front_idx=((~remove_face[:,2].type(torch.int8)) & ((remove_voxel[:,1]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    back_idx=((~remove_face[:,3].type(torch.int8)) & ((remove_voxel[:,1]<(structure.shape[1]-1)).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    bottom_idx=((~remove_face[:,4].type(torch.int8)) & ((remove_voxel[:,2]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    top_idx=((~remove_face[:,5].type(torch.int8)) & ((structure[xt,yt,zt+1]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)

    # print('left_idx',left_idx)
    # print('right',right_idx)
    # print('front',front_idx)
    # print('back',back_idx)
    # print('bottom',bottom_idx)
    # print('top ',top_idx)

    if left_idx.shape[0]>0:
        test_voxel=remove_voxel.detach().clone()    
        test_voxel[left_idx,0]=remove_voxel[left_idx,0]-1         #Coordinates of the neighbor voxel
        add_voxel, vox_idx = update_neighbour(left_idx, test_voxel, remove_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:                                
            #If present, edit its neighbor list (faces list) and remove the entry corresponding to the removed voxel
            faces[vox_idx,1]=1

        if add_voxel.shape[0]>0:                #Add any new neighbours
            top_voxels, faces = add_neighbour(top_voxels, add_voxel,  structure, faces)



    if right_idx.shape[0]>0: 
        test_voxel=remove_voxel.detach().clone()
        test_voxel[right_idx,0]=remove_voxel[right_idx,0]+1
        add_voxel, vox_idx = update_neighbour(right_idx, test_voxel, remove_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,0]=1
        if add_voxel.shape[0]>0:
            top_voxels, faces = add_neighbour(top_voxels, add_voxel, structure, faces)


    if front_idx.shape[0]>0:
        test_voxel=remove_voxel.detach().clone()
        test_voxel[front_idx,1]=remove_voxel[front_idx,1]-1
        add_voxel, vox_idx = update_neighbour(front_idx, test_voxel, remove_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,3]=1
        if add_voxel.shape[0]>0:
            top_voxels,faces = add_neighbour(top_voxels, add_voxel, structure, faces)

 
    if back_idx.shape[0]>0:
        test_voxel=remove_voxel.detach().clone()
        test_voxel[back_idx,1]=remove_voxel[back_idx,1]+1
        add_voxel, vox_idx = update_neighbour(back_idx, test_voxel, remove_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,2]=1
        if add_voxel.shape[0]>0:
            top_voxels, faces = add_neighbour(top_voxels, add_voxel,structure, faces)

  
    if bottom_idx.shape[0]>0:
        test_voxel=remove_voxel.detach().clone()
        test_voxel[bottom_idx,2]=remove_voxel[bottom_idx,2]-1
        add_voxel, vox_idx = update_neighbour(bottom_idx, test_voxel, remove_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,5]=1

        if add_voxel.shape[0]>0:
            top_voxels, faces = add_neighbour(top_voxels, add_voxel, structure, faces)



    if top_idx.shape[0]>0:
        test_voxel=remove_voxel.detach().clone()
        test_voxel[top_idx,2]=remove_voxel[top_idx,2]+1
        add_voxel, vox_idx = update_neighbour(top_idx, test_voxel, remove_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,4]=1
        if add_voxel.shape[0]>0:
            top_voxels, faces = add_neighbour(top_voxels, add_voxel, structure, faces)


    return faces, top_voxels