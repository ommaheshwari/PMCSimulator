from Modules.Settings.settings import *

# from datetime import datetime
 
# using now() to get current time
# current_time = datetime.datetime.now()

 
def calculate_cd(structure, centre, z_depth):
    # i=0
    # y_cd=torch.zeros(structure.shape[2]-GAS_HEIGHT)
    # for z_depth in range (structure.shape[2]-GAS_HEIGHT):

    # z_bottom = (structure[centre[0], centre[1], :] == 0).nonzero().min()

    # z_top = torch.tensor(sum(LAYER_THICKNESS) - DEPTH - 1)


        # z_depth = (z_top + z_bottom) // 2
 
    # x = (structure[:, centre[1], z_depth] == 0).nonzero()

    # if x.numel() > 0:

    #     x_cd = 1 + x.max() - x.min()

    # else:

    #     x_cd = torch.tensor(0)  # Or handle appropriately
 
    y = (structure[:, :, z_depth] == 0).nonzero()

    if y.numel() > 0:

        y_cd =  y.max()

    else:

        y_cd = torch.tensor(0)  # Or handle appropriately
        # i=i+1
 
    return  y_cd


def metrology_inner_spacer(structure, centre, name):
    # etch_depth = calculate_depth(structure, centre)
    # print('etch_depth',etch_depth)
    # y_cd=calculate_cd(structure, centre)
    # x_etch_cd_middle, y_etch_cd_middle, z_etch_depth_middle=calculate_cd(structure, centre,'middle')
    # x_etch_cd_bottom, y_etch_cd_bottom, z_etch_depth_bottom=calculate_cd(structure, centre,'bottom')

    # if os.path.exists('metrology'+str(current_time)+'.log'):
    #     os.remove('metrology'+str(current_time)+'.log')
    # Format the current time to a string that is safe for use in Windows file names
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f'Experiment_Results/Metrology/{name}_metrology{current_time}.log'
    
    with open(filename, 'a') as f:
    # with open('metrology'+str(current_time)+'.log', 'a') as f:
        # print('Structure Dimension X = ', WIDTH, 'Y = ', LENGTH, 'Z = ',  sum(LAYER_THICKNESS), file = f)
        # print('Mask Thickness',LAYER_THICKNESS[-1], file=f )
        # print('total etch depth = ',etch_depth.item(), file=f)
        for z in range (structure.shape[2]-GAS_HEIGHT):
            z_depth=z
            y_cd=calculate_cd(structure, centre, z_depth)
            print('y_cd = ', y_cd.item(),'at z = ',z_depth, file=f)
        # print('y_cd_top = ', y_etch_cd_top.item(),'at z = ',z_etch_depth_top.item(), file=f)
        
        # print('x_cd_mid = ', x_etch_cd_middle.item(),'at z = ',z_etch_depth_middle.item(), file=f)
        # print('y_cd_mid = ', y_etch_cd_middle.item(),'at z = ',z_etch_depth_middle.item(), file=f)
        # print('x_cd_bot = ', x_etch_cd_bottom.item(),'at z = ',z_etch_depth_bottom.item(), file=f)
        # print('y_cd_bot = ', y_etch_cd_bottom.item(),'at z = ',z_etch_depth_bottom.item(), file=f)
    return 