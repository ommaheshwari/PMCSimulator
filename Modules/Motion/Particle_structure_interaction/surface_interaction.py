from Modules.Settings.settings import *
# from test.check_hit_face import check_hit_face
from Modules.Motion.Particle_structure_interaction.etch_voxel import etch_voxel
from Modules.Motion.Particle_structure_interaction.deposit_voxel import deposit_voxel
from Modules.Motion.Particle_structure_interaction.adsorb_voxel import adsorb_voxel
from Modules.Motion.Particle_structure_interaction.delete_particle import delete_particle
from Modules.Motion.Particle_structure_interaction.reflect_particle import reflect_particle


def surface_interaction(top_voxels, faces, idx,flag, particleidx, rsnew, vsnew, spnew, hitstructure, structure, totalth, pref, petch, pdep, padsorb):
    absorbed_count = torch.tensor([0]).to(device)
    xt,yt,zt=(torch.t(top_voxels[idx,:])).to(device).type(torch.int)  #coordinates of voxel which is hit

    if flag.shape[0]>0: 
        p_reflection = pref[spnew[1,particleidx].type(torch.int), structure[xt,yt,zt].type(torch.int)-1, spnew[0,particleidx].type(torch.int)-1]
        p_etching = petch[spnew[1,particleidx].type(torch.int), structure[xt,yt,zt].type(torch.int)-1, spnew[0,particleidx].type(torch.int)-1]
        p_deposit = pdep[spnew[1,particleidx].type(torch.int), structure[xt,yt,zt].type(torch.int)-1, spnew[0,particleidx].type(torch.int)-1]
        p_adsorb = padsorb[spnew[1,particleidx].type(torch.int), structure[xt,yt,zt].type(torch.int)-1, spnew[0,particleidx].type(torch.int)-1]

        reflection = torch.bernoulli(p_reflection).type(torch.bool)
        etching = torch.bernoulli(p_etching).type(torch.bool)
        deposition = torch.bernoulli(p_deposit).type(torch.bool)
        adsorption = torch.bernoulli(p_adsorb).type(torch.bool)
        del p_reflection, p_etching, p_deposit, p_adsorb



        reflectnowhere=torch.where(reflection==0)[0]
        reflectwhere=torch.where(reflection==1)[0]
        etchhere=torch.where(((~reflection)&(etching))==1)[0]
        adsorbhere=torch.where(((~reflection)&(adsorption))==1)[0]
        deposithere=torch.where(((~reflection)&(deposition))==1)[0]

        del reflection, etching, deposition

        if reflectwhere.shape[0]>0:

            particlereflect=particleidx[reflectwhere]
            flagreflect=flag[reflectwhere]

            rsnew, vsnew, spnew=reflect_particle(flagreflect, particlereflect, rsnew, vsnew, spnew)


        if reflectnowhere.shape[0]>0:
            if etchhere.shape[0]>0:

                particleetch=(particleidx[etchhere])


                idxetch=(idx[etchhere])
                xtetch=(xt[etchhere])
                ytetch=(yt[etchhere])
                ztetch=(zt[etchhere])

                structure, top_voxels, hitstructure, faces = etch_voxel(spnew,particleetch, hitstructure,idxetch, xtetch,ytetch,ztetch,structure, top_voxels,faces,totalth)

                del idxetch, xtetch, ytetch, ztetch, particleetch

            if deposithere.shape[0]>0:

                particledeposit=(particleidx[deposithere])


                idxdeposit=(idx[deposithere])
                # xtetch=(xt[deposithere])
                # ytetch=(yt[deposithere])
                # ztetch=(zt[deposithere])
                flagdeposit=flag[deposithere]
                structure, top_voxels, hitstructure, faces, totalth = deposit_voxel(spnew,particledeposit, hitstructure,idxdeposit, flagdeposit, structure, top_voxels,faces,totalth)

                del idxdeposit, flagdeposit, particledeposit

            if adsorbhere.shape[0]>0:

                xtadsorb=(xt[adsorbhere])
                ytadsorb=(yt[adsorbhere])
                ztadsorb=(zt[adsorbhere])

                structure, hitstructure, totalth = adsorb_voxel(hitstructure, xtadsorb, ytadsorb, ztadsorb,structure, totalth)

                del xtadsorb, ytadsorb, ztadsorb


            particledelete=particleidx[reflectnowhere]

            rsnew, vsnew, spnew = delete_particle(rsnew, vsnew, spnew, particledelete)  
            
            # Remove absorbed particle
            absorbed_count = torch.tensor([particledelete.shape[0]]).to(device)
            del particledelete
            #Function to reflect particle off the structure surface
        del reflectnowhere, reflectwhere, etchhere, deposithere, xt,yt,zt, particleidx
    # print('structure[2,3,9] si',structure[2,3,9])
    torch.cuda.empty_cache()
    return rsnew, vsnew, spnew, structure, top_voxels, hitstructure, faces, absorbed_count, totalth