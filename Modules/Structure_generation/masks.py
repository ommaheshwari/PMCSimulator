from Modules.Settings.settings import *


# Function for Mask Types
def iso_circle_mask(structurei, radius):
    # Create a circular mask
    stwidth=structurei.shape[0]
    stlength=structurei.shape[1]
    mask_radius = radius*VOXEL_PER_UNIT  # Radius of the mask
    mask_center = (stwidth // 2, stlength // 2)  # Center of the mask
    mask_coords = disk(mask_center, mask_radius)
    print('mask_radius', mask_radius)
    perimeter = circle_perimeter(mask_center[0], mask_center[1], mask_radius)
    return mask_coords,perimeter, mask_center
def iso_rect_mask(structurer,rlength_,rwidth_):
    # Create a rectangular mask
    rlength=rlength_*VOXEL_PER_UNIT
    rwidth=rwidth_*VOXEL_PER_UNIT
    stwidth=structurer.shape[0]
    stlength=structurer.shape[1]
    # print('stwidth',stwidth,'stlength',stlength)
    # print('rwidth',rwidth,'rlength',rlength)
    rect_start=((stwidth-rwidth)/2, (stlength-rlength)/2)
    # print('rect_start',rect_start)
    rect_end=((stwidth-rwidth)/2+rwidth-1, (stlength-rlength)/2+rlength-1)
    # print('rect_end',rect_end)
    mask_coords = (rectangle(rect_start,rect_end))
    perimeter= (rectangle_perimeter(rect_start,rect_end))
    centre = (torch.tensor(rect_start) + torch.tensor(rect_end))//2
    
    return mask_coords,perimeter, centre.type(torch.int8)

    # Mask with multiple reactangles at different pitch
def pitch_rect_mask(structurep, rlength_, rwidth_, nrows, ncols):
    rlength=rlength_*VOXEL_PER_UNIT
    rwidth=rwidth_*VOXEL_PER_UNIT
    stwidth=structurep.shape[0]
    stlength=structurep.shape[1]
    mwidth=stwidth//nrows   #Width of each small rectangle
    mlength=stlength//ncols  #Length of each small rectangle


    mask_coords_=torch.tensor([[],[]],dtype=int).to(device)

    for i in range(0,nrows):
        for j in range(0,ncols): 

            rect_start=(mwidth*i+(mwidth-rwidth)//2, mlength*j+(mlength-rlength)//2)
            rect_end=(mwidth*i+(mwidth-rwidth)//2+rwidth, mlength*j+(mlength-rlength)//2+rlength)
            mask_coords_temp=torch.from_numpy(np.array(rectangle(rect_start,rect_end),dtype=int)).to(device)
            mask_coords_temp2=mask_coords_temp.reshape(2,mask_coords_temp.shape[1]*mask_coords_temp.shape[2]).to(device)

            mask_coords_=torch.concatenate((mask_coords_,mask_coords_temp2),axis=1).to(device)
    return tuple(mask_coords_)


    # Mask with multiple circles at different pitch
def pitch_circle_mask(structurec, radius, nrows, ncols):
    stwidth=structurec.shape[0]
    stlength=structurec.shape[1]
    mask_radius = radius*VOXEL_PER_UNIT  # Radius of the mask
    mwidth=stwidth//nrows
    mlength=stlength//ncols
    mask_coords_=torch.tensor([[],[]],dtype=int).to(device)
    for i in range(1,nrows+1):
        for j in range(1,ncols+1):    
            mask_center = (mwidth*i-(mwidth // 2), mlength*j-(mlength// 2))  # Center of the mask
            mask_coords_temp=torch.from_numpy(np.array(disk(mask_center, mask_radius),dtype=int)).to(device)
            mask_coords_=torch.concatenate((mask_coords_,mask_coords_temp),axis=1).to(device)
    return tuple(mask_coords_)