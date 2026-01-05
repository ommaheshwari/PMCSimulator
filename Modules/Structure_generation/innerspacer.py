from Modules.Settings.settings import *
from Modules.Structure_generation.add_gas import add_gas


def innerspacer(tspr, L, W, tsi, tsige, Nsheet, Lextra, Wextra, Hextra, Htop, open=0):
    tspr=tspr  * VOXEL_PER_UNIT 
    L = L  * VOXEL_PER_UNIT 
    W = W  * VOXEL_PER_UNIT 
    tsi = tsi   * VOXEL_PER_UNIT 
    tsige = tsige   * VOXEL_PER_UNIT 
    Lextra = Lextra  * VOXEL_PER_UNIT 
    Wextra = Wextra  * VOXEL_PER_UNIT 
    Hextra = Hextra  * VOXEL_PER_UNIT 
    Htop = Htop * VOXEL_PER_UNIT
    Wdevice=W+2*Wextra
    Ldevice=tspr+L+Lextra
    Hdevice=2*Hextra+Nsheet*(tsi+tsige)+Htop

    structure = torch.zeros((Wdevice, Ldevice, Hdevice), dtype=torch.int8).to(device)
    x0sige = torch.zeros(Nsheet).to(device)
    x1sige = torch.zeros(Nsheet).to(device)
    y0sige = torch.zeros(Nsheet).to(device)
    y1sige = torch.zeros(Nsheet).to(device)
    z0sige = torch.zeros(Nsheet).to(device)
    z1sige = torch.zeros(Nsheet).to(device)

    structure[:,:,:]=0

    #Spacer
    structure[0:Wdevice,Lextra:Lextra+tspr, 0:Hdevice]=3
    # structure[0:Wdevice,Ldevice-tspr:Ldevice, 0:Hdevice]=3
    #PolyGate
    structure[0:Wdevice,Lextra+tspr:Ldevice, 0:Hdevice]=4
    #Base
    structure[0:Wdevice,0:Ldevice, 0:Hextra]=4

    #Sheets
    if open==0:
        sigeblock=2
    else:  
        sigeblock=0


    for i in range(Nsheet):
        x1=Wextra
        x2=W+Wextra
        y1=Lextra
        y2=y1+tspr+L
        z3=Hextra+i*(tsige+tsi)
        z4=z3+tsige
        z1=z4
        z2=z1+tsi

        #Si
        structure[x1:x2,y1:y2, z1:z2]=1
        #SiGe
        structure[x1:x2,y1:y2, z3:z4]=sigeblock
        x0sige[i]=x1
        x1sige[i]=x2
        y0sige[i]=y1
        y1sige[i]=y2
        z0sige[i]=z3
        z1sige[i]=z4
    # structure[x1:x2,y1:y2, z2:z2+Hextra]=sigeblock
    structure=add_gas(structure)
    return structure, x0sige.type(torch.int), x1sige.type(torch.int), y0sige.type(torch.int), y1sige.type(torch.int), z0sige.type(torch.int), z1sige.type(torch.int)
