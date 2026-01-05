from Modules.Settings.settings import *

def delete_particle(rsnew, vsnew, spnew, particleidx):
    mask = torch.ones(rsnew.shape[1], dtype=torch.bool)
    mask[particleidx] = False
    # print('deleteidx',particleidx)
    # print('mask',mask)
    rsnew=rsnew[:,mask]
    vsnew=vsnew[:,mask]
    spnew=spnew[:,mask]  
    return rsnew, vsnew, spnew