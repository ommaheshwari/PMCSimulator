##main file
from Modules.Settings.settings import *
from Modules.Settings.import_functions import *
from Modules.Metrology.metrology_inner_spacer import *
plt.style.use(['science', 'notebook'])
                   
import numpy as np


def main(FLUX,T,DECAY,MAX_STEPS, ALL_STEPS,TSIGE,DIR,FPLOT,csventry,doe_file):
    N_PARTICLES=int((FLUX/(17+GAS_HEIGHT))*(7+TSIGE+GAS_HEIGHT))
    tspr=6
    L=6
    W=10
    tsi=5
    tsige=TSIGE
    Nsheet=1
    Wextra=1
    Lextra=1   #For Inner Spacer Only
    Hextra=1
    Htop=0 #For Inner Spacer Only
    structure, x0sige, x1sige, y0sige, y1sige, z0sige, z1sige = innerspacer(tspr, L, W, tsi, tsige, Nsheet, Lextra, Wextra, Hextra, Htop, open=0)
    # plotting(structure,'Experiment_Results//Plots//NSFET_spacer_'+str(TSIGE),save=0,slice='h',angle=330)

    all_voxels=torch.argwhere(structure!=0)
    totalexpfaces=exposed_faces(all_voxels,structure)
    mask=((totalexpfaces==1).any(1))
    top_voxels=all_voxels[mask,:]

    faces=exposed_faces(top_voxels,structure).type(torch.bool)

    threshold,totalth=init_voxel_threshold(top_voxels, structure)


    coords=torch.arange(0,(z1sige[0]-z0sige[0])//2-1)

    end_value=1
    start_value=6
    end_x=coords[-1]+1
    start_x=coords[0]-1
    decay_factor = DECAY

    decay_rate = np.log(end_value / start_value) / (end_x - start_x)
    values = (start_value * np.exp(decay_rate * decay_factor * (coords - start_x))).type(torch.int)
    thres=torch.clip(values, end_value, start_value)




    for i in range(len(z0sige)):
    # print(totalth[0,x0sige[0]:x1sige[0],y0sige[0]:y1sige[0],zsige[0]:zsige[0]+int(tsige/2)])
        totalth[0,x0sige[i]:x1sige[i],y0sige[i]:y1sige[i],z0sige[i]:z0sige[i]+int((z1sige[i]-z0sige[i])//2)-1]=thres
        totalth[0,x0sige[i]:x1sige[i],y0sige[i]:y1sige[i],z0sige[i]+int((z1sige[i]-z0sige[i])//2)+1:z1sige[i]]=thres.flip(0)

    # A =1.1725368117075551e-05
    # B =0.024345473620953826
    # print('T',T)


    # def custom_sigmoid(x):
    #     x0 = 550  # midpoint
    #     k = 0.03   # steepness
    #     L = 0.99  # upper asymptote
    #     lower_asymptote = 0.3
        
    #     return (L - lower_asymptote) / (1 + np.exp(-k * (x - x0))) + lower_asymptote

    # x = np.linspace(400, 700, 1000)
    # pe = custom_sigmoid(T)
    # pe=0.6
    # pe=0.95*A*np.exp(B*T)/(A*np.exp(B*650))
    # # print("pe",pe)
    # pe=min(pe,0.99)
    # print("pe now",pe)



    def custom_sigmoid(x):
        x0 = 638  # midpoint
        k = 0.05   # steepness
        L = 120  # upper asymptote
        lower_asymptote = 10
        
        return (L - lower_asymptote) / (1 + np.exp(-k * (x - x0))) + lower_asymptote

    pe = custom_sigmoid(T)/120
    # # STEP1
    P_REFLECTION_ = torch.tensor([[[0.0],   #Si, ion    # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                                [0.0], #SiGe
                                [0.0], #Spacer
                                [0.0], #BOXs
                                [0.0], #Deposited
                                [0.0]], #Adsorbed  
                                [[0.9],   #Si, neutral    # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species   #PR, neutral
                                [1-pe],
                                [1.0],
                                [1.0],
                                [1.0],
                                [1.0]]]).to(device)   #PR, neutral


    pref_, petch_, pdep_,padsorb_=probabilities(pref=P_REFLECTION_)


    # pref_, petch_, pdep_,padsorb_=probabilities()
    nsegments=5
    rsnew,vsnew,spnew,time_,stf,stfhit,structure,top_voxels, faces, end_id,p_active, p_gen, p_absorbed, p_deleted, totalth, n_ion, n_neutral=motion(structure,top_voxels,faces, totalth, MAX_STEPS, 10000000, pref_, petch_, pdep_,padsorb_, 1, ALL_STEPS, N_PARTICLES,FLUX,T,tsige,decay_factor,DIR,csventry,doe_file,nsegments,FPLOT)

    # plotscope(n_ion,n_neutral,1,structure,MAX_STEPS,save=1)
    plotting(structure,'Experiment_Results//Plots//NSFET_spacer_'+str(TSIGE),save=1,slice='h',angle=330)

    return

T=650
DECAY=1.8
N_P=200
TSIGE=10
STEPS=1500
DOE_1="Training/DOE_ISE/"
FPLOT=1
CSVENTRY=1
current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
doe_file = f'doex_sh_test_{current_time}.csv'
tr=0


main(N_P,T,DECAY,STEPS,[1250, 1500],TSIGE,DOE_1,FPLOT,CSVENTRY,doe_file)