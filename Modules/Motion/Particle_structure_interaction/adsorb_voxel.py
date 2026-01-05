from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.update_hitlist import update_hitlist
from Modules.Motion.Particle_structure_interaction.update_threshold import update_threshold
from Modules.Animation.plotting import plotting

def adsorb_voxel(hitstructure, xt,yt,zt,structure, totalth):



    structure[xt,yt,zt,]=ADSORB_ID  
        # print('structure[2,3,9]',structure[2,3,9])  
        # plotting(structure,'step_1',save=0,slice='f')
        # print('structure[xtup,xtup,xtup]',structure[xtup,ytup,ztup])  
        
                                                                                #Remove that voxel from structure
    hitstructure[0,xt,yt,zt]=0                                 #Avoid negative valuesi in db 

    hitstructure[1,xt,yt,zt]=0                                      #Avoid negative valuesi in db        


    totalth[0,xt,yt,zt] = ADSORBED_ETCH_THRESHOLD
    totalth[1,xt,yt,zt] = ADSORBED_DEPOSIT_THRESHOLD
        
    return structure, hitstructure, totalth