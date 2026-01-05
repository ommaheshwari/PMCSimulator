from Modules.Settings.settings import *

def boundary_particle(rsnew,vsnew,spnew,structure1):
# Deleting particles out of outer boundaries
    initial_count = rsnew.shape[1]


    rsz0=((rsnew[2]<=structure1.shape[2]+0.5).nonzero())[:,0].to(device)
    rsnew=rsnew[:,rsz0]
    vsnew=vsnew[:,rsz0]
    spnew=spnew[:,rsz0]

    rsz1=((rsnew[2]>=(0)).nonzero())[:,0].to(device)
    rsnew=rsnew[:,rsz1]
    vsnew=vsnew[:,rsz1]
    spnew=spnew[:,rsz1]

    if PERIODIC_X==1:
        rsx0=((rsnew[0]>structure1.shape[0]+EXTRA_SOURCE_X1).nonzero())[:,0].to(device)
        rsnew[0,rsx0]=rsnew[0,rsx0]-(structure1.shape[0]+EXTRA_SOURCE_X1)
        rsx1=((rsnew[0]<(0-EXTRA_SOURCE_X0)).nonzero())[:,0].to(device)
        rsnew[0,rsx1]=rsnew[0,rsx1]+(structure1.shape[0]+EXTRA_SOURCE_X1)
    else: 
        rsx0=((rsnew[0]<=structure1.shape[0]+EXTRA_SOURCE_X1).nonzero())[:,0].to(device)
        rsnew=rsnew[:,rsx0]
        vsnew=vsnew[:,rsx0]
        spnew=spnew[:,rsx0]
        rsx1=((rsnew[0]>=(0-EXTRA_SOURCE_X0)).nonzero())[:,0].to(device)
        rsnew=rsnew[:,rsx1]
        vsnew=vsnew[:,rsx1]
        spnew=spnew[:,rsx1]
    if PERIODIC_Y==1:
        rsy0=((rsnew[1]>structure1.shape[1]+EXTRA_SOURCE_Y1).nonzero())[:,0].to(device)
        rsnew[1,rsy0]=rsnew[1,rsy0]-(structure1.shape[1]+EXTRA_SOURCE_Y1)
        rsy1=((rsnew[1]<(0-EXTRA_SOURCE_Y0)).nonzero())[:,0].to(device)
        rsnew[1,rsy1]=rsnew[1,rsy1]+(structure1.shape[1]+EXTRA_SOURCE_Y1)
    else:
        rsy0=((rsnew[1]<=structure1.shape[1]+EXTRA_SOURCE_Y1).nonzero())[:,0].to(device)
        rsnew=rsnew[:,rsy0]
        vsnew=vsnew[:,rsy0]
        spnew=spnew[:,rsy0]
        rsy1=((rsnew[1]>=(0-EXTRA_SOURCE_Y0)).nonzero())[:,0].to(device)
        rsnew=rsnew[:,rsy1]
        vsnew=vsnew[:,rsy1]
        spnew=spnew[:,rsy1]
        
    final_count =rsnew.shape[1]
    total_deleted = torch.tensor([initial_count- final_count]).to(device)
    return rsnew, vsnew, spnew, total_deleted