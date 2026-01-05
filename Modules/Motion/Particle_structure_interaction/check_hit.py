from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.check_cross import check_cross
from Modules.Motion.Particle_structure_interaction.check_eq import check_eq
from Modules.Motion.Particle_structure_interaction.generate_eq import generate_eq
import time as timex
def check_hit(A,B,top_voxels, faces, rsnew, particleidx, allvoxidx):
    # t0=timex.time()
    # pvtensor=torch.cat((particleidx.reshape(-1,1),allvoxidx.reshape(-1,1)),dim=1)
    flag = torch.zeros((particleidx.shape[0])).to(device)
    # print('pvtensor',pvtensor)
    # flag=0
    ## check top
    # print('faces[pid,:] check_hit',faces[particleidx,:])
    # print('top_voxels[faces[:,5]==1,:]',top_voxels[faces[:,5]==1,2].unique())
    # t1=timex.time()
    eqt=generate_eq(top_voxels, faces, A, B, 2, 5)
    # t2=timex.time()
    crosst , pidt = check_eq(A, B, eqt, 1, 2, particleidx)
    # t3=timex.time()
    indices1t, pidtt, intt=check_cross(crosst, top_voxels,faces,5, pidt)
    intt[:,2]=intt[:,2]+1
    del eqt, crosst, pidt
    # t4=timex.time()
    eqr=generate_eq(top_voxels, faces, A, B, 0, 1)
    # t5=timex.time()
    crossr , pidr = check_eq(A, B, eqr, 1, 0, particleidx)
    # t6=timex.time()
    indices1r, pidrr, intr=check_cross(crossr, top_voxels,faces,1, pidr)
    intr[:,0]=intr[:,0]+1
    del eqr, crossr, pidr
    # t7=timex.time()
    eql=generate_eq(top_voxels, faces, A, B, 0, 0)
    # t8=timex.time()
    crossl , pidl = check_eq(A, B, eql, 0, 0, particleidx)
    # t9=timex.time()
    
    indices1l, pidll, intl=check_cross(crossl, top_voxels,faces,0, pidl)
    del eql, crossl, pidl
    # t10=timex.time()
    eqf=generate_eq(top_voxels, faces, A, B, 1, 2)
    # t11=timex.time()
    crossf , pidf = check_eq(A, B, eqf, 0, 1, particleidx)
    # t12=timex.time()
    indices1f, pidff, intf=check_cross(crossf, top_voxels,faces,2, pidf)
    del eqf, pidf, crossf
    # t13=timex.time()
    eqb=generate_eq(top_voxels, faces, A, B, 1, 3)
    # t14=timex.time()
    crossb , pidb = check_eq(A, B, eqb, 1, 1, particleidx)
    # t15=timex.time()
    indices1b, pidbb, intb=check_cross(crossb, top_voxels,faces,3, pidb)
    intb[:,1]=intb[:,1]+1
    del eqb, crossb, pidb
    # t16=timex.time()
    eqbot=generate_eq(top_voxels, faces, A, B, 2, 4)
    # t17=timex.time()
    crossbot , pidbot = check_eq(A, B, eqbot, 0,2, particleidx)
    # t18=timex.time()
    indices1bot,pidbotbot, intbot=check_cross(crossbot, top_voxels,faces,4, pidbot)
    del eqbot, crossbot, pidbot
    # t19=timex.time()
    # print('indices1t',indices1t)
    # print('indices1r',indices1r)
    # print('indices1l',indices1l)
    # print('indices1f',indices1f)
    # print('indices1b',indices1b)
    # print('indices1bot',indices1bot)
    # print('stacked_indices',torch.cat((indices1t, indices1r, indices1l, indices1f, indices1b,indices1bot),0))
    # print('stacked_pid',torch.cat((pidtt,pidrr,pidll,pidff,pidbb,pidbotbot),0))
    # t5=timex.time()
    indices1=torch.cat((indices1t, indices1r, indices1l, indices1f, indices1b,indices1bot),0)
    del indices1t, indices1r, indices1l, indices1f, indices1b,indices1bot
    # t6=timex.time()
    pid=torch.cat((pidtt,pidrr,pidll,pidff,pidbb,pidbotbot),0).to(device)
    # t7=timex.time()
    inters=torch.cat((intt,intr,intl,intf,intb,intbot),0).to(device)
    del intt,intr,intl,intf,intb,intbot
    # t8=timex.time()
    flag=torch.cat((6*(torch.ones(pidtt.shape[0])),2*(torch.ones(pidrr.shape[0])),1*(torch.ones(pidll.shape[0])), 
                    3*(torch.ones(pidff.shape[0])),4*(torch.ones(pidbb.shape[0])),5*(torch.ones(pidbotbot.shape[0])),)).to(device)
    del pidtt,pidrr,pidll,pidff,pidbb,pidbotbot
    # t9=timex.time()
    # print('inters.T',inters.T)
    # print('rsnew[:,pid]',rsnew[:,pid])
    rsnew[:,pid]=inters.T

    del inters
    torch.cuda.empty_cache()

    # t20=timex.time()
    # print('total',t20-t0, t2-t1, t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7,t9-t8,t10-t9,t11-t10,t12-t11,t13-t12,t14-t13,t15-t14,t16-t15,t17-t16,t18-t17,t19-t18)
    # print('chlgya')
    # unique, inverse = torch.unique(pid, sorted=False, return_inverse=True)
    # perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    # inverse, perm = inverse.flip([0]), perm.flip([0])
    # perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    # pid=pid[perm]
    # indices1=indices1[perm]
    # flag=flag[perm]
    # if indices1t.shape[0]>0:
    #     # flag=6
    #     # print('hitidx',particleidx)
    #     # print('vox idx',indices1t)
    #     # print('rsnew[:,pidtt]',rsnew[:,pidtt])
    #     # print('top_voxels[indices1t,:]',top_voxels[indices1t,:])
    #     # print('top_voxels',top_voxels)
    #     rsnew[2,pidtt]=top_voxels[indices1t,2]+1
    #     # print('rsnew[:,particleidx] after',rsnew[:,particleidx])
    #     # return indices1t, flag, rsnew


    # #Right
    # # crossr = check_eq(A, B, top_voxels[faces[:,1]==1,0].unique()+1, 1, 0)
    # # indices1r=check_cross(crossr, top_voxels,1,0)
    # if indices1r.shape[0]>0:
    #     # flag=2  
    #     rsnew[0,pidrr]=top_voxels[indices1r,0]+1
    #     # return indices1r, flag, rsnew
    
    # # crossl = check_eq(A, B, top_voxels[faces[:,0]==1,0].unique(), 0, 0)
    # # indices1l=check_cross(crossl, top_voxels,0,0)
    # if indices1l.shape[0]>0:
    #     # flag=1  
    #     rsnew[0,pidll]=top_voxels[indices1l,0]
    #     # return indices1l, flag, rsnew
    

    # # crossf = check_eq(A, B, top_voxels[faces[:,2]==1,1].unique(), 0, 1)

    # # indices1f=check_cross(crossf, top_voxels,0,1)
    # if indices1f.shape[0]>0:
    #     # flag=3  
    #     rsnew[1,pidff]=top_voxels[indices1f,1]
    #     # return indices1f, flag, rsnew

    # # crossb = check_eq(A, B, top_voxels[faces[:,3]==1,1].unique()+1, 1, 1)
    # # indices1b=check_cross(crossb, top_voxels,1,1)
    # if indices1b.shape[0]>0:
    #     # flag=4  
    #     rsnew[1,pidbb]=top_voxels[indices1b,1]+1
    #     # return indices1b, flag, rsnew
    

    # # crossbot = check_eq(A, B, top_voxels[faces[:,4]==1,2].unique(), 0,2)
    # # indices1bot=check_cross(crossbot, top_voxels,0,2)
    # if indices1bot.shape[0]>0:
    #     # flag=5  
    #     rsnew[2,pidbotbot]=top_voxels[indices1bot,2]

    return indices1, flag, rsnew, pid