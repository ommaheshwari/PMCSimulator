from Modules.Settings.settings import *
import numpy as np
def plotscope(n_ion,n_neutral,nsegments,structure,TIME_STEPS,save=1):
    # plt.figure(figsize=(10,8))
    fig, ax = plt.subplots( nsegments,2,figsize=(5,2), sharex=True)

    s=3
    topcoord=structure.shape[2]
    # segments=np.linspace(0,topcoord,nsegments+1, dtype=int)
    segments=torch.tensor([2, structure.shape[2]-GAS_HEIGHT-2-10])
    # print(TIME_STEPS)
    # print(SCOPE_STEPS)
    # print(np.arange(0,TIME_STEPS,SCOPE_STEPS))
    for i in range(nsegments):
        if nsegments==1:
            ax[0].plot(np.arange(0,TIME_STEPS,SCOPE_STEPS), n_ion[i].cpu(), c = 'b', label = 'ions '+str(segments[i])+' to '+str(segments[i+1]),marker='s',lw=1,ms=s)
            ax[1].plot(np.arange(0,TIME_STEPS,SCOPE_STEPS), (n_neutral[i].cpu())/(26*(segments[i+1]//2-segments[i]//2)), c = 'g', label = 'neutrals '+str(segments[i]//2)+' to '+str(segments[i+1]//2),marker='o',lw=1,ms=s)

            ax[0].legend(fontsize=10)
            ax[1].legend(fontsize=10)
            ax[0].tick_params(which='both',width=1,direction='in',labelsize=8)
            ax[0].tick_params(which='major', length=4)
            ax[0].tick_params(which='minor', length=2)
            ax[1].tick_params(which='both',width=1,direction='in',labelsize=8)
            ax[1].tick_params(which='major', length=4)
            ax[1].tick_params(which='minor', length=2)



        else:
            ax[i, 0].plot(np.arange(0,TIME_STEPS,SCOPE_STEPS), n_ion[i].cpu(), c = 'b', label = 'ions '+str(segments[i])+' to '+str(segments[i+1]),marker='s',lw=1,ms=s)
            ax[i, 1].plot(np.arange(0,TIME_STEPS,SCOPE_STEPS), n_neutral[i].cpu(), c = 'g', label = 'neutrals '+str(segments[i])+' to '+str(segments[i+1]),marker='o',lw=1,ms=s)

            ax[i,0].legend(fontsize=10)
            ax[i,1].legend(fontsize=10)
            ax[i, 0].tick_params(which='both',width=1,direction='in',labelsize=8)
            ax[i, 0].tick_params(which='major', length=4)
            ax[i, 0].tick_params(which='minor', length=2)
            ax[i, 1].tick_params(which='both',width=1,direction='in',labelsize=8)
            ax[i, 1].tick_params(which='major', length=4)
            ax[i, 1].tick_params(which='minor', length=2)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Time Steps')
    # plt.ylabel('No. of Particles')
    # fig.text(0.5, 0.04, 'Time Steps', ha='center', fontsize = 20)
    # fig.text(0.04, 0.5, 'No. of Particles', va='center', rotation='vertical', fontsize=20)
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    if save==1:
        plt.savefig('Experiment_Results/Plots/' + str('scope') + str('_') + str(current_time) + str('.png'))
    else:
        plt.show()




    return