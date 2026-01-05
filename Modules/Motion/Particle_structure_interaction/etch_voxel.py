from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.update_hitlist import update_hitlist
from Modules.Motion.Particle_structure_interaction.update_threshold import update_threshold
from Modules.Animation.plotting import plotting

def etch_voxel(spnew,particleidx, hitstructure,idx, xt,yt,zt,structure, top_voxels,faces,totalth):

    hitinc = spnew[2,particleidx].detach().clone().type(torch.int8) #How many hitcounts increase when this particle hits voxel
    hitstructure = update_hitlist(hitstructure,idx,xt,yt,zt,hitinc,'etch') #update number of hits for this voxel
    threshold=totalth[0,xt,yt,zt]
    hit=hitstructure[0,xt,yt,zt]
    # print('hitstructure.shape',hitstructure.shape)
    # print('idx',idx)
    # print('threshold',threshold)
    # print('hit',hit)
    # print('xt', xt,yt,zt)
    if torch.ge(hit,threshold).nonzero().shape[0]>0:
        updates=torch.ge(hit,threshold).nonzero().reshape(-1)
        idxupdate=idx[updates].type(torch.int)
        xtup=xt[updates].type(torch.int)
        ytup=yt[updates].type(torch.int)
        ztup=zt[updates].type(torch.int)
        # print('idx_update',idxupdate)
        # print('xtup',xtup,ytup,ztup)
        # print('hitstructure[:,xtup,xtup,xtup]',hitstructure[:,xtup,xtup,xtup])
        structure[xtup,ytup,ztup]=0  
        # print('structure[2,3,9]',structure[2,3,9])  
        # plotting(structure,'step_1',save=0,slice='f')
        # print('structure[xtup,xtup,xtup]',structure[xtup,ytup,ztup])  
        
                                                                                #Remove that voxel from structure
        hitstructure[0,xtup,ytup,ztup]=threshold[updates]                                       #Avoid negative valuesi in db 
        # print('threshold_update',threshold[updates])       
        hitstructure[1,xtup,ytup,ztup]=0                                      #Avoid negative valuesi in db        

        top_voxels, faces = update_threshold(structure,top_voxels,idxupdate,faces,xtup,ytup,ztup)
        
    return structure, top_voxels, hitstructure, faces