from Modules.Settings.settings import *

def particle_stats(p_active, p_gen, p_absorbed, p_deleted):
    # plt.figure(figsize=(10,8))
    fig, ax = plt.subplots(2, 2,figsize=(25,8))
    s=3
    ax[0, 0].plot(np.arange(0,TIME_STEPS), p_active.cpu(), c = 'b', label = 'Active',marker='s',lw=1,ms=s)
    ax[0, 1].plot(np.arange(0,TIME_STEPS), p_gen.cpu(), c = 'g', label = 'Generated',marker='o',lw=1,ms=s)
    ax[1, 0].plot(np.arange(0,TIME_STEPS), p_absorbed.cpu(), c = 'm', label = 'Absorbed',marker='v',lw=1,ms=s)
    ax[1, 1].plot(np.arange(0,TIME_STEPS), p_deleted.cpu(), c = 'r', label = 'Deleted',marker='d',lw=1,ms=s)
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel('Time Steps')
    # plt.ylabel('No. of Particles')
    fig.text(0.5, 0.04, 'Time Steps', ha='center', fontsize = 20)
    fig.text(0.04, 0.5, 'No. of Particles', va='center', rotation='vertical', fontsize=20)
    plt.show()




    return