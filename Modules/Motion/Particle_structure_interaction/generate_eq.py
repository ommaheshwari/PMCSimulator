from Modules.Settings.settings import *

def generate_eq(top_voxels, faces, A, B, vec, checkface):
    
    eqn=top_voxels[faces[:,checkface]==1,vec].unique().type(torch.int16)
    # print('checkface',checkface)
    # print('top_voxels',top_voxels[faces[:,checkface]==1,:])
    mask=B[vec]>0
#     print('mask',mask)
#     print('~mask',~mask)


    if checkface==1 or checkface==3 or checkface==5:
            # print('eqn^',eqn)
            eqn=eqn+1
            eq=eqn.repeat(A.shape[1],1)
            # print('eq rep^', eq)
            sqn=torch.lt(eq,A[vec].reshape(-1,1))
            sqp=torch.gt(eq,(A+B)[vec].reshape(-1,1))
            # print('sqp^',sqp)
            # print('sqn^',sqn)
            eq[sqn]=-5
            eq[sqp]=-5
            eq[~mask,:]= -5
    else:
            # print('eqn',eqn)
            eq=eqn.repeat(A.shape[1],1)
            # print('eq rep', eq)
            sqn=torch.gt(eq,A[vec].reshape(-1,1))
            sqp=torch.lt(eq,(A+B)[vec].reshape(-1,1))
            # print('sqp',sqp)
            # print('sqn',sqn)
            eq[sqn]=-5
            eq[sqp]=-5
        #     print('eq b', eq)
            eq[mask,:]= -5
    # print('eq',eq)
    # eq2=eq.detach().clone()
    del mask, sqn, sqp, eqn
    torch.cuda.empty_cache()
    # print('eq',eq.shape)
    return eq