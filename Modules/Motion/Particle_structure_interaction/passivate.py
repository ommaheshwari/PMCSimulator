from Modules.Settings.settings import *
from Modules.Surface_voxels.exposed_faces import exposed_faces
def passivate(structure, top_voxels, faces, threshold, totalth):
    # print('top_voxels',top_voxels.dtype)
    # print('structure[top_voxels]',structure[top_voxels[:,0],top_voxels[:,1],top_voxels[:,2]])
    # mask=(structure[top_voxels[:,0],top_voxels[:,1],top_voxels[:,2]]!=2)
    # top_voxels=top_voxels[mask,:]
    # faces=faces[mask,:]
    # print('faces[:,0]==1',faces[:,0]==1)
    avl=top_voxels[faces[:,0]==1,:]
    avr=top_voxels[faces[:,1]==1,:]
    avf=top_voxels[faces[:,2]==1,:]
    avb=top_voxels[faces[:,3]==1,:]
    avbot=top_voxels[faces[:,4]==1,:]
    avt=top_voxels[faces[:,5]==1,:]
    avl[:,0]=avl[:,0]-1
    avr[:,0]=avr[:,0]+1
    avf[:,1]=avf[:,1]-1
    avb[:,1]=avb[:,1]+1
    avbot[:,2]=avbot[:,2]-1
    avt[:,2]=avt[:,2]+1

    add_voxels=torch.cat((avl,avr,avf,avb,avbot,avt),0).unique(dim=0)

    print('totalth.shape',totalth.shape)
    print('add_voxels',add_voxels)
    structure[add_voxels[:,0],add_voxels[:,1],add_voxels[:,2]]=3
    faces=exposed_faces(add_voxels, structure)
    thres=threshold*torch.ones(add_voxels.shape[0]).type(torch.int).to(device)
    # totalth=torch.dstack((totalth, torch.zeros((totalth.shape[0], totalth.shape[1], GAS_HEIGHT), dtype=torch.int)))
    # print('structure.shape',structure.shape)
    print('totalth.shape',totalth.shape)
    totalth[add_voxels[:,0],add_voxels[:,1],add_voxels[:,2]]=threshold
    print('structure>0',structure>0)
    # totalth=totalth[structure>0]


    return structure, add_voxels, faces, thres,totalth
