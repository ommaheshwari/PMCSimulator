from Modules.Settings.settings import *
from scipy.ndimage import uniform_filter
from csv import writer
from Modules.Animation.plotting import plotting
def create_csv_sheetrelease(structure,FLUX,TIME_STEPS,decay_factor,T,tsige,DIR,doe_file,csventry=1):
    l= 10
    w=10
    vel=np.zeros((l*VOXEL_PER_UNIT,w*VOXEL_PER_UNIT))
    strx=structure[4:-4,4:-4,51:62]
    plotting(strx,'_',save=0,slice='f',angle=330)
    for yy in range(0,l*VOXEL_PER_UNIT):
        for xx in range (0,w*VOXEL_PER_UNIT):

            ss=(strx[xx, yy, :] == 0).nonzero()
            print(' ss',ss)
            print('ssmax',ss.min())

            vel[yy-0][xx-0]=ss.min().item()*0.5e-3
    print('vel',vel)

    # np.savetxt("foo2.csv", vel, delimiter=",")
    MAX_DEPTH=vel.max()
    MIN_DEPTH=vel.min()
    vel4 = uniform_filter(vel, size=4,mode='nearest')
    # print('vel4',vel4)
    topcol=np.arange(0,(0.001*w)+0.00025,0.00025)
    leftcol=np.arange(0,((0.001*l)+0.000250),0.00025)
    vel5=np.zeros(((l*VOXEL_PER_UNIT)+2,(w*VOXEL_PER_UNIT)+2))
    vel5[1:,0]=leftcol
    vel5[0,1:]=topcol
    vel5[1:-1,1:-1]=vel4
    vel5[1:-1,-1]=vel4[:,-1]
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