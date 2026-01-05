from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.update_hitlist import update_hitlist
from Modules.Motion.Particle_structure_interaction.update_d_threshold import update_d_threshold
from Modules.Motion.Particle_structure_interaction.deposition_slots import deposition_slots
def deposit_voxel(spnew,particleidx,hitstructure,idx, flag, structure, top_voxels,faces,totalth):
    
    depo_slots=deposition_slots(top_voxels[idx,:], flag)
    xt,yt,zt=(torch.t(depo_slots)).to(device).type(torch.int)

    hitinc = spnew[2,particleidx].detach().clone().type(torch.int8) #How many hitcounts increase when this particle hits voxel

    hitstructure = update_hitlist(hitstructure, idx, xt, yt, zt, hitinc,'deposit').type(torch.int8) #update number of hits for this voxel
    # print('hitstructure.type',hitstructure.dtype)
    dhit=hitstructure[1,xt,yt,zt]
    dthreshold=totalth[1,xt,yt,zt].type(torch.int8)
    # print('dthreshold',dthreshold)
    if torch.ge(dhit,dthreshold).nonzero().shape[0]>0:
        updates=torch.ge(dhit,dthreshold).nonzero().reshape(-1)
        # idxupdate=idx[updates].type(torch.int)

        xtup=xt[updates].type(torch.int)
        ytup=yt[updates].type(torch.int)
        ztup=zt[updates].type(torch.int)
        add_voxel=depo_slots[updates,:]
        structure[xt,yt,zt]=DEPOSIT_ID     
                                                                                #Remove that voxel from structure
        hitstructure[1,xtup,ytup,ztup]=dthreshold[updates]                                    #Avoid negative valuesi in db        
        hitstructure[0,xtup,ytup,ztup]=0
        totalth[0,xtup,ytup,ztup] = DEPOSITED_ETCH_THRESHOLD

        top_voxels, faces = update_d_threshold(structure,top_voxels,add_voxel,faces,xtup,ytup,ztup)

        del xtup, ytup, ztup, add_voxel, updates
    del hitinc, dhit, dthreshold, depo_slots

    torch.cuda.empty_cache()
    return structure, top_voxels,hitstructure, faces, totalth