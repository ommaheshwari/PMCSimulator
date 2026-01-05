from Modules.Settings.settings import *
# import time as timex
def check_eq(A, B, eqn, ad, vec, particleidx):
    
    eq=eqn
    # eq=eqn.repeat(A.shape[1],1)
    # sqn=torch.le(eq,A[vec].reshape(-1,1))
    # # print('sqn',sqn)
    # # eq2=eq.detach().clone()
    # eq[sqn]=-5
    # print('eq2',eq2)
    # print('eq[sqn]=-1',eq[sqn.type(bool)]=-1)
    # print('eq[sqn==False]=-5',sqn[sqn==False]=-5)
    # print('eq', eq)
    # print('A[vec]',A[vec].reshape(-1,1))
    # print('(eq-A[vec])',(eq-(A[vec]).reshape(-1,1)))
    t= (eq-(A[vec]).reshape(-1,1))/((B[vec]).reshape(-1,1))
    # print('t',t)
    # print('treshape',t.reshape(eqn.shape[0],A.shape[1]))
    # print('torch.einsum',torch.einsum('bp,qr->bpqr', t, B))
    # print(torch.einsum('qr,pq->rpq', t,B))
    cross=A+torch.einsum('qr,pq->rpq', t,B)
    # print('crosshere',cross)

    # cross=A+torch.einsum('bp,qr->bpqr', t, B) 
    # cross=A+t*B
    # print('cross',cross, cross.shape)
    # print('cross[:,:,vec,:]',cross[:,:,vec,:])
    if ad==1:
        cross[:,vec,:]=cross[:,vec,:]-1
    #     print('ad')
    # print('cross',cross)
    # starttime=timex.time()
    crossed=cross.permute(1,0,2).reshape(3,-1)
    # endtime=timex.time()
    # crossed = torch.hstack([cross[i,:,:] for i in range(cross.shape[0])])
    # endtime2=timex.time()
    # print('time2', endtime-starttime)
    # print('time1', endtime2-endtime)
    # print('rehsaped cross',cross.reshape(3,-1))
    # x=cross.reshape(-1,A.shape[1])[::3,:].reshape(1,-1)
    # y=cross.reshape(-1,A.shape[1])[1::3,:].reshape(1,-1)
    # z=cross.reshape(-1,A.shape[1])[2::3,:].reshape(1,-1)
    # crossed=torch.cat((x,y,z),0)
    # print('crossed2',crossed2)
    # print('crossed',crossed)
    pid=particleidx.repeat(eq.shape[1])
    # print('pid',pid)
    del t, cross, particleidx, A,B,eq
    torch.cuda.empty_cache()
    mask=(crossed>0).all(0)
    crossed=crossed[:,mask]
    pid=pid[mask]
    del mask
    # print(torch.where((crossed>0), dim=0))
    return crossed, pid