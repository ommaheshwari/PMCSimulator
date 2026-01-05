from Modules.Settings.settings import *


# Function for Etch
def boolean_etch(structure, mask_coords):
    structure_height=structure.shape[2]-GAS_HEIGHT
    etch_coordinate=int(structure_height-DEPTH*VOXEL_PER_UNIT)
    structure[mask_coords[0], mask_coords[1], etch_coordinate:structure_height] = 0     # Etching Masked voxels
    
    # Code for identifying top layer
    top_=torch.argwhere(structure[:, :, structure_height-1] != 0).to(device)
    zeros=torch.argwhere(structure[:, :, structure_height-1] == 0).to(device)
    ztop=(structure_height-1)*torch.ones(top_.shape[0],1).to(device)
    ztop2=(etch_coordinate-1)*torch.ones(zeros.shape[0],1).to(device)
    top_vox1=torch.cat((top_,ztop),1).to(device)
    top_vox2=torch.cat((zeros,ztop2),1).to(device)
    top_vox=torch.cat((top_vox1,top_vox2)).to(device)
    return structure
