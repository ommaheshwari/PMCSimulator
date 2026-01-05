from Modules.Settings.settings import *

def exposed_faces(top_voxels,structure):

    ef=torch.zeros((top_voxels.shape[0],6)).to(device)
    x=top_voxels[:,0].type(torch. int)
    y=top_voxels[:,1].type(torch. int)
    z=top_voxels[:,2].type(torch. int)

    # x[x>0] & structure[x-1,y,z]==0
    # mask=(x[structure[x[x>0]-1,y[x>0],z[x>0]]==0])
    # print(ef[mask,0])
    # print('x[x>0] & structure[x-1,y,z]==0',x.shape, x[x>0].shape,(structure[x[x>0]-1,y[x>0],z[x>0]]==0).shape)
    


    # print('id',id.shape)
    ix1=(x>0).nonzero().reshape(-1)
    ef[ix1[structure[x[ix1]-1,y[ix1],z[ix1]]==0].reshape(-1),0]=1

    iy1=(y>0).nonzero().reshape(-1)
    ef[iy1[structure[x[iy1],y[iy1]-1,z[iy1]]==0].reshape(-1),2]=1

    iz1=(z>0).nonzero().reshape(-1)
    ef[iz1[structure[x[iz1],y[iz1],z[iz1]-1]==0].reshape(-1),4]=1

    ix2=(x<(structure.shape[0]-1)).nonzero().reshape(-1)
    ef[ix2[structure[x[ix2]+1,y[ix2],z[ix2]]==0].reshape(-1),1]=1    
    
    iy2=(y<(structure.shape[1]-1)).nonzero().reshape(-1)
    ef[iy2[structure[x[iy2],y[iy2]+1,z[iy2]]==0].reshape(-1),3]=1
 
    iz2=(z<(structure.shape[2]-1-(GAS_HEIGHT*VOXEL_PER_UNIT))).nonzero().reshape(-1)
    # print('iz2',iz2)
    # print('structure[x[iz2],y[iz2],z[iz2]+1]==0',structure[x[iz2],y[iz2],z[iz2]+1]==0)
    # print('iz2[structure[x[iz2],y[iz2],z[iz2]+1]==0].nonzero().reshape(-1)',iz2[structure[x[iz2],y[iz2],z[iz2]+1]==0].reshape(-1))
    ef[iz2[structure[x[iz2],y[iz2],z[iz2]+1]==0].reshape(-1),5]=1

    # print(ef[ix1[structure[x[ix1]-1,y[ix1],z[ix1]]==0].nonzero().reshape(-1),0])


    # print(id[structure[x[id],y[id],z[id]+1]==0].nonzero().shape)

    # print('(structure[x[x>0]-1,y[x>0],z[x>0]]==0)',(structure[x[x>0]-1,y[x>0],z[x>0]]==2).nonzero())
    # ef[(x>0)][(structure[x[x>0]-1,y[x>0],z[x>0]]==0),0]=1
    # ef[(structure[x[x>0]-1,y[x>0],z[x>0]]==0),0]=1
    # # ef[x==0,0]=0

    # ef[(x<(structure.shape[0]-1))][(structure[x[(x<(structure.shape[0]-1))]+1,y[(x<(structure.shape[0]-1))],z[(x<(structure.shape[0]-1))]]==0),1]=1
    # # ef[(x>=(structure.shape[0]-1)),1]=0

    # ef[(y>0)][(structure[x[(y>0)],y[(y>0)]-1,z[(y>0)]]==0),2]=1
    # # ef[y==0,2]=0

    # ef[(y<(structure.shape[1]-1))][(structure[x[(y<(structure.shape[1]-1))],y[(y<(structure.shape[1]-1))]+1,z[(y<(structure.shape[1]-1))]]==0),3]=1
    # # ef[(y>=(structure.shape[1]-1)),3]=0


    # ef[(z>0)][(structure[x[(z>0)],y[(z>0)],z[(z>0)]-1]==0),4]=1
    # # ef[z==0,4]=0
    # ef[(structure[x[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],y[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],z[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))]+1]==0),5]=1
    # print('?',ef[(structure[x[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],y[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],z[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))]+1]==0)])
    # ef[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))][(structure[x[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],y[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],z[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))]+1]==0),5]=1
    # print(ef[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))])
    # print(ef[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))][(structure[x[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],y[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))],z[(z<(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT)))]+1]==0),5])
    # # ef[(z>=(structure.shape[2]-(GAS_HEIGHT*VOXEL_PER_UNIT))),5]=0

    # print('ef',ef)
    return ef