from Modules.Settings.settings import *
from scipy.ndimage import uniform_filter
from csv import writer

def create_csv(structure,FLUX,TIME_STEPS,decay_factor,T,tsige,DIR,doe_file,csventry=1):
    
    vel=np.zeros(((2*tsige+1),40))

    for zz in range(2,(3+2*tsige)):
        for xx in range (6,46):

            ss=(structure[xx, :, zz] == 0).nonzero()
            # print(' y = (structure[:, :, :] == 0).nonzero()',ss)

            vel[zz-2][xx-6]=ss.max().item()*0.5e-3
    # print('vel',vel)

    # np.savetxt("foo2.csv", vel, delimiter=",")
    MAX_DEPTH=vel.max()
    MIN_DEPTH=vel.min()
    vel4 = uniform_filter(vel, size=4,mode='nearest')
    # print('vel4',vel4)
    topcol=np.arange(0,0.0205,0.0005)
    leftcol=np.arange(0,(0.001*tsige+0.00050),0.0005)
    vel5=np.zeros((2*tsige+2,42))
    vel5[1:,0]=leftcol
    vel5[0,1:]=topcol
    vel5[1:,1:-1]=vel4
    vel5[1:,-1]=vel4[:,-1]
    file_name="training_tsige_"+str(tsige)+"_flux_"+str(FLUX)+"_steps_"+str(TIME_STEPS)+'_Temp_'+str(T)+'_decay_'+str(decay_factor)+".csv"
    np.savetxt(DIR+file_name, vel5, delimiter=",")



    if csventry==1:
        # List that we want to add as a new row
        List = [T, decay_factor, FLUX, tsige, TIME_STEPS, MAX_DEPTH, MIN_DEPTH,DIR,file_name,0]
        with open(doe_file, 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()

    if csventry==0:
        print("Temperature:", T, "Decay:", decay_factor,"FLUX:", FLUX,"TSIGE:", tsige, "STEPS:", TIME_STEPS,"\n")
        print("MAX_DEPTH:",MAX_DEPTH.round(5),"MIN_DEPTH:", MIN_DEPTH.round(5))
    return