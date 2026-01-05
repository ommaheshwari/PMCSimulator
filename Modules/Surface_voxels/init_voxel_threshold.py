from Modules.Settings.settings import *

def init_voxel_threshold(top_voxelsx, structurex):
    thresholdx=torch.zeros(top_voxelsx.shape[0]).type(torch.int).to(device)

    totalth=torch.zeros((2, structurex.shape[0], structurex.shape[1], structurex.shape[2])).type(torch.int).to(device)
    layers=LAYER_NAMES
    # Create array of voxel thresholds corresponding to top voxels
    xt,yt,zt=(torch.t(top_voxelsx[:])).to(device).type(torch.int)
    all_voxels=torch.argwhere(structurex!=0)
    # xta,yta,zta=(torch.t(all_voxels[:])).to(device).type(torch.int)
    # print(all_voxels.shape)
    k=0
    for i in range (len(layers)):
        thresholdx[structurex[xt,yt,zt]==(i+1)]= LAYER_ETCH_THRESHOLD[i] #Adjusts Threshold of Voxel
        # print('structurex[xta,yta,zta]==(i+1)',(torch.argwhere(structurex[:,:,:]==(i+1))))
        xta,yta,zta=(torch.t((torch.argwhere(structurex[:,:,:]==(i+1))))).to(device).type(torch.int)
        
        totalth[0,xta,yta,zta]= LAYER_ETCH_THRESHOLD[i] #Adjusts Threshold of Voxel
        totalth[1,xta,yta,zta]=LAYER_DEPOSIT_THRESHOLD[i] 
    return thresholdx,totalth