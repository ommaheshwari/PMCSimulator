from Modules.Settings.settings import *


def scope(rsnew, spnew, nsegments,structure):
    # topcoord=GAS_HEIGHT+torch.sum(LAYER_THICKNESS)-1
    topcoord=structure.shape[2]
    nion=torch.zeros((nsegments,1)).to(device)
    nneutral=torch.zeros((nsegments,1)).to(device)
    segments=torch.tensor([2, structure.shape[2]-GAS_HEIGHT-2-10])
    nsegments=1
    # print('segments,',segments)
    for i in range(nsegments):
        maska=(segments[i]<=rsnew[2]).nonzero().reshape(-1)
        # print('segments[i]',segments[i],segments[i+1])
        # print('rsnew[2,maska]',rsnew[2,maska])
        maskb=maska[(rsnew[2,maska]<=segments[i+1]).nonzero()].reshape(-1)
        # print('maska',maska)
        # print('maskb',maskb)
        # print('rsnew.shape',rsnew.shape)
        # print('spnew.shape',spnew.shape)
        totalp=spnew[1,maskb]
        # print('totalp',totalp)
        # print("totalp.shape[0]",totalp.shape[0])
        # print('(totalp==0).nonzero()',(totalp==0).nonzero().shape[0])
        nion[i]=(totalp==0).nonzero().shape[0]
        nneutral[i]=totalp.shape[0]-nion[i]
        # print('nion[i]',nion)
        # print('nneutral[i]',nneutral)
    return nion, nneutral

# def scope(rsnew, spnew, nsegments,structure):
#     # topcoord=GAS_HEIGHT+torch.sum(LAYER_THICKNESS)-1
#     topcoord=structure.shape[2]
#     nion=torch.zeros((nsegments,1)).to(device)
#     nneutral=torch.zeros((nsegments,1)).to(device)
#     segments=torch.linspace(0,topcoord,nsegments+1, dtype=int)
#     # print('segments,',segments)
#     for i in range(nsegments):
#         maska=(segments[i]<=rsnew[2]).nonzero().reshape(-1)
#         # print('segments[i]',segments[i],segments[i+1])
#         # print('rsnew[2,maska]',rsnew[2,maska])
#         maskb=maska[(rsnew[2,maska]<=segments[i+1]).nonzero()].reshape(-1)
#         # print('maska',maska)
#         # print('maskb',maskb)
#         # print('rsnew.shape',rsnew.shape)
#         # print('spnew.shape',spnew.shape)
#         totalp=spnew[1,maskb]
#         # print('totalp',totalp)
#         # print("totalp.shape[0]",totalp.shape[0])
#         # print('(totalp==0).nonzero()',(totalp==0).nonzero().shape[0])
#         nion[i]=(totalp==0).nonzero().shape[0]
#         nneutral[i]=totalp.shape[0]-nion[i]
#         # print('nion[i]',nion)
#         # print('nneutral[i]',nneutral)
#     return nion, nneutral