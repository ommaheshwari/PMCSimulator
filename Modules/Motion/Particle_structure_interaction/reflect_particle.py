from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.delete_particle import delete_particle

def reflect_particle(flag,reflect,rsnew, vsnew,spnew):
    if flag.shape[0]>0:
        vsnew[0,reflect[flag==1]] = -(vsnew[0,reflect[flag==1]])
        vsnew[0,reflect[flag==2]] = -(vsnew[0,reflect[flag==2]])        
        vsnew[1,reflect[flag==3]] = -(vsnew[1,reflect[flag==3]])
        vsnew[1,reflect[flag==4]] = -(vsnew[1,reflect[flag==4]])
        vsnew[2,reflect[flag==5]] = -(vsnew[2,reflect[flag==5]])
        vsnew[2,reflect[flag==6]] = -(vsnew[2,reflect[flag==6]])

        # if flag==1 or flag==2:
        #     vsnew[0,reflect] = -(vsnew[0,reflect])

        # if flag==3 or flag==4:
        #     vsnew[1,reflect] = -(vsnew[1,reflect])

        # if flag==5 or flag==6:
        #     vsnew[2,reflect] = -(vsnew[2,reflect])

        spnew[3,reflect] = spnew[3,reflect]+1       #Counting number if reflections
        mask = reflect[(spnew[3,reflect] > REFLECTMAX)]
        # print('spnew[3,reflect]',spnew[3,reflect])
        # print('reflect',reflect)
        # print("mask",mask)
        rsnew, vsnew, spnew = delete_particle(rsnew, vsnew, spnew, mask)
        # print('spnew[3,reflect]',spnew[3,reflect])
    return rsnew, vsnew, spnew