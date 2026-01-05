from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.compare_array import compare_array
import time as timex
def check_cross(cross, top_voxels,faces, fv, pid):
    new_cross=torch.floor(torch.t(cross))
    # print('cross.type', cross.dtype)
    # print('new_corss', new_cross)
    # print('arr1.argsort()',torch.sort(new_cross, dim=0, stable=True))
    # new_cross=new_cross_u[new_cross_u[:,0].sort()[1],:]
    # print('sorted',new_cross_u[new_cross_u[:,0].sort()[1],:])
    # new_cross=torch.unique(new_cross, dim=0)

    # indices1, indices2 = compare_array(top_voxels, new_cross)
    # print('top_voxels',top_voxels)
    # print('new_cross',new_cross.unsqueeze(-1).shape)
    # print('pid shape',pid.shape)
    # print('top_)vox.shape',top_voxels.shape)
    # torch.cuda.synchronize()
    # t0=timex.time()
    # print('torch.where',torch.where((top_voxels.T == new_cross.unsqueeze(-1)).all(dim=1)))
    # valzx2 = (torch.where((new_cross.T == top_voxels.unsqueeze(-1)).all(dim=1))[1]).type(torch.int)
    # inxx3= 
    # print('new_cross[valzx2]',new_cross[valzx2])
    # torch.cuda.synchronize()
    # t1=timex.time()
    # valzx,inxx= torch.topk(((top_voxels.T == new_cross.unsqueeze(-1)).all(dim=1)).int(), 1, 1)
    # print('new_cross.shape',new_cross.shape)
    valzx,inxx = (torch.where((top_voxels.T == new_cross.unsqueeze(-1)).all(dim=1)))
    # print('inxx.shape',inxx.shape)
    # torch.cuda.synchronize()
    # t2=timex.time()
    del new_cross
    # torch.cuda.synchronize()
    # t3=timex.time()
    valzx = ((valzx.reshape(-1))).type(torch.int32)

    # valzx=(valzx!=0).reshape(-1).type(torch.bool)
    torch.cuda.synchronize()
    # t4=timex.time()
    inxx = ((inxx.reshape(-1))).type(torch.int16)
    
    
    # if inxx.shape[0]>0:
    #     print('maxinxx',torch.max(inxx))
    # print('valzx2',pid[valzx2])
    # print('valzx3',pid[valzx3])
    # print('valzx1',pid[valzx])
    # assert(pid[valzx3]).equal(pid[valzx])
    # assert(pid[valzx2]).equal(pid[valzx])
    # torch.cuda.synchronize()
    # t5=timex.time()
    pid=pid[valzx]
    
    # torch.cuda.synchronize()
    # t6=timex.time()
    # inxx = inxx[valzx]
    # torch.cuda.synchronize()
    # t7=timex.time()

    # print('inxx1',inxx)

    # print('inxx2',inxx2)
    # assert(inxx).equal(inxx2)

    intersection=(cross.T)[valzx]

    del valzx, cross
    
    ic=(faces[(inxx).type(torch.int),fv]==1)
    pid=pid[ic]
    inxx=inxx[ic]
    # print('inxx',inxx.dtype)
    # print('pidtpye', pid.dtype)
    intersection=intersection[ic]
    # t6=timex.time()
    # print(t3-t2)
    # print(t2-t1, t3-t2,t4-t3,t5-t4,t6-t5)

    # print(t6-t1, tx-t1,t2-tx, t3-t2,t4-t3,t5-t4,t6-t5)
    del ic
    torch.cuda.empty_cache()

    return inxx, pid, intersection