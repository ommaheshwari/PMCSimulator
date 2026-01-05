from Modules.Settings.settings import *


import numpy as np

def animate(time_, rsnew,spnew, structure, stf, save=0, name='animation',fps=6, dpi=300, angle=180):
    t = time_.cpu()
    df = pd.DataFrame({"time": t ,"x" : rsnew[0,:].flatten().cpu(), "y" : rsnew[1,:].flatten().cpu(), "z" : rsnew[2,:].flatten().cpu(), "sp" : spnew[0,:].flatten().cpu(),"ty" : spnew[1,:].flatten().cpu()})
    # df.to_excel('dftest.xlsx')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-EXTRA_SOURCE_X0,structure.shape[0]+EXTRA_SOURCE_X1)
    ax.set_ylim(-EXTRA_SOURCE_Y0,structure.shape[1]+EXTRA_SOURCE_Y1)
    ax.set_zlim(0,structure.shape[2])
    def update_graph(num):
        temp_num=num*DB_STEPS
        data=df[df['time']==temp_num]
        # data=data[]
        structurefplot=stf.reshape([(TIME_STEPS//DB_STEPS)+1,structure.shape[0],structure.shape[1],structure.shape[2]])[num+1,:,:,:].cpu()
        plt.cla()

        # xx, yy = np.meshgrid(range(-EXTRA_SOURCE,structure.shape[0]+1+EXTRA_SOURCE), range(-EXTRA_SOURCE,structure.shape[1]+1+EXTRA_SOURCE))
        # ax.plot_surface(xx, yy, structure.shape[2]*xx/xx, alpha=0.2,edgecolors='white', lw=2)   
        colors = np.zeros(structurefplot.cpu().shape + (4,))  # Create a 4D array for RGBA values
        colors[structurefplot.cpu() == 1] = [0.8, 0.8, 0.8, 1]  # Color for the structure == 1
        colors[structurefplot.cpu() == 2] = [1, 1, 0, 1]  # Fill layer 2 with green color
        colors[structurefplot.cpu() == 3] = [0, 1, 1, 1]  # Color for the structure == 3
        colors[structurefplot.cpu() == 4] = [0, 150/256, 75/256, 0]  # Color for the structure == 3
        ax.voxels(structurefplot.cpu()>0, facecolors=colors, edgecolor='None', alpha=1)  
        # print('logic',data.sp==3)
        ax.scatter(data.x[data.ty==0], data.y[data.ty==0], data.z[data.ty==0], s=0.1, marker='d',c='r')
        ax.scatter(data.x[data.ty==1], data.y[data.ty==1], data.z[data.ty==1], s=0.1, marker='o',c='k')  
        # ax.scatter(data.x[np.logical_and((data.sp==3).values , (data.ty==0).values)], data.y[np.logical_and((data.sp==3).values , (data.ty==0).values)], data.z[np.logical_and((data.sp==3).values , (data.ty==0).values)], s=10, marker='d',c='b')
        # ax.scatter(data.x[np.logical_and((data.sp==1).values , (data.ty==0)).values], data.y[np.logical_and((data.sp==1).values , (data.ty==0)).values], data.z[np.logical_and((data.sp==1).values , (data.ty==0)).values], s=10,marker='d',c='m')
        # ax.scatter(data.x[np.logical_and((data.sp==2).values , (data.ty==0)).values], data.y[np.logical_and((data.sp==2).values , (data.ty==0)).values], data.z[np.logical_and((data.sp==2).values , (data.ty==0)).values], s=10,marker='d',c='y')
        # ax.scatter(data.x[np.logical_and((data.sp==3).values , (data.ty==1)).values], data.y[np.logical_and((data.sp==3).values , (data.ty==1)).values], data.z[np.logical_and((data.sp==3).values , (data.ty==1)).values], s=10, marker='o',c='b')
        # ax.scatter(data.x[np.logical_and((data.sp==1).values , (data.ty==1)).values], data.y[np.logical_and((data.sp==1).values , (data.ty==1)).values], data.z[np.logical_and((data.sp==1).values , (data.ty==1)).values], s=10,marker='o',c='m')
        # ax.scatter(data.x[np.logical_and((data.sp==2).values , (data.ty==1)).values], data.y[np.logical_and((data.sp==2).values , (data.ty==1)).values], data.z[np.logical_and((data.sp==2).values , (data.ty==1)).values], s=10,marker='o',c='y')
        ax.set_title('time={},total_particles={}'.format(num,len(data.x)))

        ax.set_xlim(-EXTRA_SOURCE_X0,structure.shape[0]+EXTRA_SOURCE_X1)
        ax.set_ylim(-EXTRA_SOURCE_Y0,structure.shape[1]+EXTRA_SOURCE_Y1)
        ax.set_zlim(0,structure.shape[2])
        # ax.set_zlim(0,structure.shape[2])
        ax.view_init(azim=angle, elev=15)
        plt.axis('scaled')
        # plt.show()

    print('TIME_STEPS//DB_STEPS',TIME_STEPS//DB_STEPS)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, int(TIME_STEPS//DB_STEPS), 
                                interval=1e-5, blit=False)
    if save:
        writer = matplotlib.animation.FFMpegWriter(fps=fps)
        current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        ani.save(name+ str('_') + str(current_time) +'.mp4',writer=writer,dpi=dpi)

    plt.show()
    return