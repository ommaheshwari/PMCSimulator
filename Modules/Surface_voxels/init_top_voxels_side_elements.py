from Modules.Settings.settings import *


# Function for Identifying top voxels
def init_top_voxels_side_elements(structure,  peri=1,etch=1):
    structure_height=structure.shape[2]-GAS_HEIGHT
    etch_coordinate=int(structure_height-DEPTH)
    peri=torch.tensor(np.array(peri),).to(device)
    print('peri',peri)
    peri=peri[:,peri[0,:]>0]
    peri=peri[:,peri[1,:]>0]
    print('peri',peri)
    peri=peri[:,peri[0,:]<structure.shape[0]]
    # peri=peri[peri[1,:]>0,:]
    peri=peri[:,peri[1,:]<structure.shape[1]]
    print('peri',peri)
    if etch==1:
    # Code for identifying top layer if structure was etched
        top_=torch.argwhere(structure[:, :, structure_height-1] != 0).to(device)
        zeros=torch.argwhere(structure[:, :, structure_height-1] == 0).to(device)

        ztop=(structure_height-1)*torch.ones(top_.shape[0],1).to(device)
        ztop2=(etch_coordinate-1)*torch.ones(zeros.shape[0],1).to(device)
        top_vox1=torch.cat((top_,ztop),1).to(device)
        top_vox2=torch.cat((zeros,ztop2),1).to(device)
        top_vox=torch.cat((top_vox1,top_vox2)).to(device)
        for h in range(etch_coordinate,structure_height):
            zperi=h*torch.ones(peri.shape[1],1).to(device)
            top_peri=torch.cat((torch.t(peri),zperi),1).to(device)

            top_vox=torch.cat((top_vox,top_peri)).to(device)


    else:     # Code for identifying top layer if structure was not etched

        top_=torch.argwhere(structure[:, :, structure_height-1] != 0).to(device)
        ztop=(structure_height-1)*torch.ones(top_.shape[0],1).to(device)
        top_vox=torch.cat((top_,ztop),1).to(device)
    top_vox=torch.unique(top_vox,dim=0).to(device)
    # print('structure[top_vox[:,0],top_vox[:,1], top_vox[:,2]]>0',structure[top_vox[:,0].type(torch.int),top_vox[:,1].type(torch.int), top_vox[:,2].type(torch.int)]>0)
    top_vox=top_vox[structure[top_vox[:,0].type(torch.int),top_vox[:,1].type(torch.int), top_vox[:,2].type(torch.int)]>0]
    return  top_vox.type(torch.int16)
