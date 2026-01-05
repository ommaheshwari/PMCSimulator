from Modules.Settings.settings import *


def plotting(structurex,name,save=0, slice='f',angle=180):
    if slice=='q':
        structurex=structurex[:(structurex.shape[0]//2),:(structurex.shape[1]//2),:].cpu()
    if slice=='h':
        structurex=structurex[:(structurex.shape[0]//2),:,:].cpu()
    if slice=='hy':
        structurex=structurex[:,(structurex.shape[1]//2):,:].cpu()
    else:
        structurex=structurex[:,:,:].cpu()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.zeros(structurex.shape + (4,))  # Create a 4D array for RGBA values
    colors[structurex == 1] = [0.8, 0.8, 0.8, 1]  # Color for the structure == 1
    colors[structurex == 2] = [1, 1, 0, 1]  # Fill layer 2 with green color
    colors[structurex == 3] = [0, 1, 1, 0.0]  # Color for the structure == 3
    colors[structurex == 4] = [0, 150/256, 75/256, 0]  # Color for the structure == 3
    colors[structurex == 5] = [0.1, 1, 0.5, 1]
    colors[structurex == 6] = [1, 0, 1, 1]
    colors[structurex == 7] = [0, 0, 1, 1]
    ax.voxels((structurex.cpu()>0), facecolors=colors, edgecolor='None', alpha=1)  
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticks([0,20,40],[0,10,20])
    ax.set_yticks([0,12,24],[0,6,12])
    ax.tick_params(axis='x',  pad=0)
    ax.tick_params(axis='y',  pad=0)
    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.set_zticks([0, 20, 40, 60, 80],[0, 10, 20, 30, 40])
    plt.axis('scaled')
    plt.draw()
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    if save==1:
        # ax.view_init(azim=angle1, elev=10)
        # plt.savefig('Experiment_Results/Plots/' + str(name) + str('_') + str(current_time) + str('.png'))
        ax.view_init(azim=angle, elev=30)
        plt.savefig(str(name) + str('_') + str(current_time) + str('.png'))
        
    # plt.ion()
    if save==0:
        ax.view_init(azim=angle, elev=30)
        plt.show()
    # plt.pause(5)
    plt.close()
    return
