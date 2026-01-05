from Modules.Settings.settings import *
from Modules.Motion.Generate_particles.generate_particles import generate_particles
from Modules.Motion.Generate_particles.generate_particles_iso import generate_particles_iso
from Modules.Motion.Particle_particle_Interaction.get_deltad2_pairs import get_deltad2_pairs
from Modules.Motion.Particle_particle_Interaction.compute_new_v import compute_new_v
from Modules.Motion.Particle_structure_interaction.surface_interaction import surface_interaction
from Modules.Motion.Particle_structure_interaction.reflect_particle import reflect_particle
from Modules.Motion.Particle_structure_interaction.boundary_particle import boundary_particle
from Modules.Animation.plotting import plotting
from Modules.Motion.percentage_etched import *
from Modules.Motion.Particle_structure_interaction.check_hit import check_hit
import time as timex
from Modules.Motion.Particle_structure_interaction.compare_array import compare_array
from Modules.Motion.Particle_structure_interaction.scope import scope
from Modules.Metrology.create_csv import create_csv
from Modules.Metrology.create_csv_sheetrelease import create_csv_sheetrelease
def motion(structure, top_voxels, faces, totalth, time_steps,db_steps, pref, petch, pdep,padsorb,etch_iteration , ALL_STEPS, N_PARTICLES,FLUX,T,tsige,decay_factor,DIR,csventry,doe_file,nsegments=3,FPLOT=0):
    
    tcount=0
    time=torch.empty((0,),dtype=torch.int8).to(device)
    rsnew=torch.empty((3,0),dtype=torch.float16 ).to(device) #current position of particles
    vsnew=torch.empty((3,0),dtype=torch.float16).to(device) #current velocity of particles
    spnew=torch.empty((5,0)).to(device) #current velocity of particles
    rsf=torch.empty((3,0),dtype=torch.float16).to(device)   #position of particles with time
    vsf=torch.empty((3,0),dtype=torch.float16).to(device)   #velocity of particles with time
    spf=torch.empty((5,0)).to(device) 
    # hit=torch.zeros(top_voxels.shape[0],dtype=torch.int8).to(device)        #Local list of current no. of hits on surface voxels 
    structuref=torch.empty((structure.flatten().shape[0]),dtype=torch.int8).to(device)  #Evolution of structure with time
    hitstructure=torch.zeros((2, structure.shape[0], structure.shape[1],structure.shape[2]),dtype=torch.int).to(device) #Main list of current no. of hits on whole structure
    hitstructuref=torch.zeros((structure.flatten().shape[0]),dtype=torch.int).to(device) #Main list of total no. of hits with time on whole structure
    # Initial_count=initial_count(structure)
    # if os.path.exists('output.log'):
    #     os.remove('output.log')
    p_gen = torch.zeros((0)).to(device)
    p_active = torch.zeros((0)).to(device)
    p_deleted = torch.zeros((0)).to(device)
    count=0
    p_absorbed = torch.zeros((0)).to(device)
    n_ion=torch.zeros((nsegments,0)).to(device)
    n_neutral=torch.zeros((nsegments,0)).to(device)
    for i in range(0,time_steps): #iterate over all time
        absorbed_countout = torch.tensor([0]).to(device)
        torch.cuda.synchronize()
        start=timex.time()

        may_hit=rsnew[:,rsnew[2]<=top_voxels[:,2].max()+1].to(device) #filtering particles above top surface which may hit the surface
        all_voxels=torch.argwhere(structure!=0)
        if may_hit.shape[1]>0:

            i1, _ = compare_array(all_voxels, torch.t(torch.floor(may_hit)))
            del may_hit
            torch.cuda.empty_cache()    
            if i1.shape[0]>0:
                # print('may_hit',may_hit)
                # print('top_voxels',top_voxels)
                rsnewt=torch.t(torch.floor(rsnew.detach().clone())).type(torch.int)
                rsnewtu=torch.unique(rsnewt,dim=0)
                all_voxelst=all_voxels.detach().clone()[i1,:]
                del all_voxels, i1, _
                torch.cuda.empty_cache()  
                torch.cuda.synchronize()
                t1=timex.time()
                pxyzt, __ = compare_array(rsnewtu, all_voxelst)
                del all_voxelst
                torch.cuda.empty_cache()  
                torch.cuda.synchronize()
                t2=timex.time()
                # print('rsnewtu[pxyzt]',rsnewtu[pxyzt])
                # valz,pxyz= torch.topk(((rsnewt.t() == rsnewtu[pxyzt].unsqueeze(-1)).all(dim=1)).int(), 1, 1)
                # pxyz = pxyz[valz!=0]
                pxyz = (torch.where((rsnewt.t() == rsnewtu[pxyzt].unsqueeze(-1)).all(dim=1))[1]).type(torch.int32)
                del rsnewt, rsnewtu, pxyzt
                torch.cuda.empty_cache()                
                torch.cuda.synchronize()
                t3=timex.time()
                # print('rsnew[:,pxyzw]',rsnew[:,pxyzw])
                # print('rsnew[:,pxyz]',rsnew[:,pxyz])
                # print('pxyz',pxyz, pxyz.shape)
                # print('all_voxelst[__,:]',all_voxelst[__,:])
                A=rsnew[:,pxyz]
                B=-vsnew[:,pxyz]*DT
                # print('A',A.shape)
                # print('B',B)
                # print('A+B',A+B)
                torch.cuda.synchronize()
                t4=timex.time()
                indices1, flag, rsnew , pid= check_hit(A,B,top_voxels, faces, rsnew, pxyz,__)
                torch.cuda.synchronize()
                t5=timex.time()
                del pxyz, A, B, __
                torch.cuda.empty_cache()
                # print('flag',flag)
                # print('rsnew[:,pid]',rsnew[:,pid])
                # print('pid',pid)
                # print('top_voxels[indices1,:]',top_voxels[indices1.type(torch.int),:])
                # print('pid.sort',pid.sort()[0])
                # print('pxyz.sort,',pxyz.sort()[0])
                # print('rsnew[:,pid]',rsnew[:,pid])

                # assert (pid.sort()[0]).equal(pxyz.sort()[0])

                rsnew, vsnew, spnew, structure, top_voxels, hitstructure, faces, absorbed_countout, totalth =surface_interaction(top_voxels, faces, indices1.type(torch.int), flag, pid, rsnew,vsnew,spnew, hitstructure, structure,totalth, pref, petch, pdep,padsorb)
                del indices1, flag, pid
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                t6=timex.time()
                # print('top_voxels,', top_voxels.dtype)
                # print('faces',faces)
                # print('comp',t2-t1, 'where',t3-t2, 'checkhit',t5-t4, 'sint',t6-t5)
                # print('checkhit',t5-t4)
        total_gen_particles = torch.tensor([0]).to(device)
        for num_particle in range (len(PARTICLES)):            
            if i%GENERATION_STEPS[num_particle]==0: #generate new particles after every gen_step steps
                r,v,sp, count, end_id, gen_particles =generate_particles_iso(structure,IDENTIFIER_VALUE[num_particle],HIT_WEIGHT[num_particle],N_PARTICLES,T,ION_NEUTRAL_RATIO[num_particle],1-ION_NEUTRAL_RATIO[num_particle],count)
                rsnew=torch.cat((rsnew,r),1)
                vsnew=torch.cat((vsnew,v),1)
                spnew=torch.cat((spnew,sp),1)
                total_gen_particles = gen_particles + total_gen_particles

        # assert (rsnew[2,:]>0).all()


    
        # Deleting particles out of outer boundaries
        rsnew,vsnew, spnew, total_deleted=boundary_particle(rsnew, vsnew, spnew, structure)

        # p_gen = torch.cat((p_gen, total_gen_particles))
        # print('p_absorbed',p_absorbed)
        # print('absorbed_countout',absorbed_countout)
        # p_absorbed = torch.cat((p_absorbed, absorbed_countout))
        # p_deleted = torch.cat((p_deleted,total_deleted))
        # p_deleted = torch.cat((p_deleted,torch.tensor([total_deleted])))
        # Particle-Particle Collision
        # # Array of all particle pairs in simulation  
        # ids = torch.arange(rsnew.shape[1]).to(device)
        # ids_pairs = torch.combinations(ids,2).to(device) #makes pairs of all particles, every timestep. ******NEEDS TO BE IMPROVED******
        # ic = ids_pairs[get_deltad2_pairs(rsnew, ids_pairs) <  2*RADIUS**2] #Array of particles colliding
        # #Update velocity  after colission
        # vsnew[:,ic[:,0]], vsnew[:,ic[:,1]] = compute_new_v(vsnew[:,ic[:,0]], vsnew[:,ic[:,1]], rsnew[:,ic[:,0]], rsnew[:,ic[:,1]])

        #Update particle position 

        rsnew= rsnew + vsnew*DT
        
        # print(hitstructure)
        if i%db_steps==0:
            tcount=tcount+1
            # print(tcount)
            #Add curent positions and velocities to memory
            time=torch.cat((time,i*torch.ones(rsnew.shape[1]).to(device))).to(device)
            rsold=rsf
            vsold=vsf
            spold=spf
            structureold=structuref
            hitstructureold=hitstructuref

            rsf=torch.cat((rsold,rsnew),1)
            vsf=torch.cat((vsold,vsnew),1)
            spf=torch.cat((spold,spnew),1)

            structuref=torch.cat((structureold,structure.flatten()))
            hitstructuref=torch.cat((hitstructureold,hitstructure.flatten()))
            torch.cuda.synchronize()
        stop=timex.time()
        # print('### Iteration',etch_iteration,' Step',i, 'time',stop - start,'Active Particles', rsnew.shape[1])
        print(i)
        if i%SCOPE_STEPS==0:
            # p_active=torch.cat((p_active,torch.tensor([rsnew.shape[1]]).to(device)))
            # plotting(structure,'NSFET_spacer'+str(i),save=1,slice='h',angle=330)
            # plotting(structure,'NSFET_spacer'+str(i),save=1,slice='h',angle=360)
            nion,nneutral=scope(rsnew, spnew, nsegments,structure)
            # print('nion,nneutral',nion,nneutral)
            n_ion = torch.cat((n_ion, nion),1)
            n_neutral = torch.cat((n_neutral, nneutral),1)

        if (i+1) in ALL_STEPS:
            FPLOT=1
            # create_csv_sheetrelease(structure,N_PARTICLES,i+1,decay_factor,T,tsige,DIR,doe_file,csventry)
            if FPLOT:
                plotting(structure,DIR+"training_tsige_"+str(tsige)+"_nump_"+str(FLUX)+"_steps_"+str(i+1)+'_Temp_'+str(T)+'_decay_'+str(decay_factor),save=1,slice='hy',angle=-140)
    # print('Done')


 
    # print('rs vs,',rsnew.dtype, vsnew.dtype, rsf.dtype, vsf.dtype)

    return rsf, vsf,spf,time,structuref,hitstructuref,structure, top_voxels, faces, end_id, p_active, p_gen, p_absorbed, p_deleted, totalth, n_ion, n_neutral


