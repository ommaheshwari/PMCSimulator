from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.add_neighbour import add_neighbour
from Modules.Motion.Particle_structure_interaction.update_neighbour import update_neighbour
def clear_d_faces(add_faces, add_voxel, top_voxels, faces, structure, xt, yt,zt):
    #Check which neighbors are present in structure of the voxel to be removed (remove_face==0 means neighbor is present)

    left_idx=((~add_faces[:,0].type(torch.int8)) & ((add_voxel[:,0]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    right_idx=((~add_faces[:,1].type(torch.int8)) & ((add_voxel[:,0]<(structure.shape[0]-1)).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    front_idx=((~add_faces[:,2].type(torch.int8)) & ((add_voxel[:,1]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    back_idx=((~add_faces[:,3].type(torch.int8)) & ((add_voxel[:,1]<(structure.shape[1]-1)).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    bottom_idx=((~add_faces[:,4].type(torch.int8)) & ((add_voxel[:,2]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)
    top_idx=((~add_faces[:,5].type(torch.int8)) & ((structure[xt,yt,zt+1]>0).reshape(-1,))==1).nonzero(as_tuple=False).reshape(-1,)


    if left_idx.shape[0]>0:
        test_voxel=add_voxel.detach().clone()    
        test_voxel[left_idx,0]=add_voxel[left_idx,0]-1         #Coordinates of the neighbor voxel
        _, vox_idx = update_neighbour(left_idx, test_voxel, add_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:                                
            #If present, edit its neighbor list (faces list) and remove the entry corresponding to the removed voxel
            faces[vox_idx,1]=0


    if right_idx.shape[0]>0: 
        test_voxel=add_voxel.detach().clone()
        test_voxel[right_idx,0]=add_voxel[right_idx,0]+1
        _, vox_idx = update_neighbour(right_idx, test_voxel, add_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,0]=0


    if front_idx.shape[0]>0:
        test_voxel=add_voxel.detach().clone()
        test_voxel[front_idx,1]=add_voxel[front_idx,1]-1
        _, vox_idx = update_neighbour(front_idx, test_voxel, add_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,3]=0

 
    if back_idx.shape[0]>0:
        test_voxel=add_voxel.detach().clone()
        test_voxel[back_idx,1]=add_voxel[back_idx,1]+1
        _, vox_idx = update_neighbour(back_idx, test_voxel, add_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,2]=0

  
    if bottom_idx.shape[0]>0:
        test_voxel=add_voxel.detach().clone()
        test_voxel[bottom_idx,2]=add_voxel[bottom_idx,2]-1
        _, vox_idx = update_neighbour(bottom_idx, test_voxel, add_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,5]=0




    if top_idx.shape[0]>0:
        test_voxel=add_voxel.detach().clone()
        test_voxel[top_idx,2]=add_voxel[top_idx,2]+1
        _, vox_idx = update_neighbour(top_idx, test_voxel, add_voxel, top_voxels)         #Check if this neighbor is in surface voxels list already
        if vox_idx.shape[0]>0:
            faces[vox_idx,4]=0



    maskx=(torch.sum(faces,dim=1)!=0)
    top_voxels=top_voxels[maskx]
    faces=faces[maskx,:]

    del left_idx, right_idx, front_idx, bottom_idx, top_idx, back_idx,maskx, add_faces, add_voxel



    return faces, top_voxels