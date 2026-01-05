from Modules.Settings.settings import *

def generate_particles_iso(structure,species,hit_weight,N_PARTICLES,T,pion=1,pneutral=0, count=0,):
    ## 0> ion, 1>neutral
    start = count

 
    K=1.380649e-23
    m=35.5*1.67377e-27
    # T=600
    V_NEUTRAL= ((3*K*(T+273)/m)**(1/2))*1e9 #nm/s
    # velocity_dist_ion=torch.distributions.Normal(0,-velocity*0.1) #Deciding Angular Distribution
    # velocity_distz_ion=torch.distributions.Normal(velocity,-velocity*0.1) #Deciding Energy Distribution
    velocity_dist_neutral=torch.distributions.Normal(0,V_NEUTRAL) 



    # x,y= np.indices([20,20])
    x, y = np.meshgrid(range(-EXTRA_SOURCE_X0,structure.shape[0]+EXTRA_SOURCE_X1), range(-EXTRA_SOURCE_Y0,structure.shape[1]+EXTRA_SOURCE_Y1))
    xdist=torch.distributions.uniform.Uniform(-EXTRA_SOURCE_X0,structure.shape[0]+EXTRA_SOURCE_X1)
    ydist=torch.distributions.uniform.Uniform(-EXTRA_SOURCE_Y0,structure.shape[1]+EXTRA_SOURCE_Y1)
    zdist=torch.distributions.uniform.Uniform(0,structure.shape[2]-GAS_HEIGHT)
    ztop=torch.distributions.uniform.Uniform(structure.shape[2]-GAS_HEIGHT+1,structure.shape[2])
    yleft=torch.distributions.uniform.Uniform(-EXTRA_SOURCE_Y0,0+1e-3)
    yright=torch.distributions.uniform.Uniform(structure.shape[1]-1e-3,structure.shape[1]+EXTRA_SOURCE_Y1)
    xfront=torch.distributions.uniform.Uniform(structure.shape[0]-1e-3,structure.shape[0]+EXTRA_SOURCE_X1)
    xback=torch.distributions.uniform.Uniform(-EXTRA_SOURCE_X0,0+1e-3)
    

    # rion = torch.tensor(np.array([x.ravel(),y.ravel(),x.ravel()])).to(device)+torch.rand(1)[0]
    # rion[2]=structure.shape[2]

    
    # n_particles = (structure.shape[0]+EXTRA_SOURCE_X0+EXTRA_SOURCE_X1) * (structure.shape[1]+EXTRA_SOURCE_Y0+EXTRA_SOURCE_Y1)
    n_particles = N_PARTICLES
    print('n_particles',n_particles)
    r = torch.zeros((3,n_particles)).to(device)
    v = torch.zeros((3,n_particles)).to(device)

    # print('n_particles', n_particles)
    sp=  torch.zeros((5,n_particles), dtype=torch.int8).to(device) #Index for species [species, type, hit_weight, reflections_count]
    sp[0] = species*torch.ones((1,n_particles))
    sp[1] = torch.multinomial(torch.tensor([pion,pneutral]), n_particles, replacement=True)
    print('sp1',sp[1])
    sp[2] = hit_weight*torch.ones((1,n_particles))
    # sp[4] = torch.arange(start, start+n_particles).type(torch.int64)

    
    ## IONS Initialization
    n_ions=(sp[1]==0).nonzero().shape[0]
    print('n_ions',n_ions)
    ion_idx=torch.t((sp[1]==0).nonzero())[0]
    v[2,ion_idx] = -torch.sqrt(2*ENERGY.sample((n_ions,))*Q/MASS[species-1]).to(device)*torch.cos(ANGLE_POLAR.sample((n_ions,))).to(device)*1e9
    v[0,ion_idx] = -torch.sqrt(2*ENERGY.sample((n_ions,))*Q/MASS[species-1]).to(device)*torch.sin(ANGLE_POLAR.sample((n_ions,))).to(device)*torch.sin(ANGLE_AZIMUTHAL.sample((n_ions,))).to(device)*1e9
    v[1,ion_idx] = -torch.sqrt(2*ENERGY.sample((n_ions,))*Q/MASS[species-1]).to(device)*torch.sin(ANGLE_POLAR.sample((n_ions,))).to(device)*torch.cos(ANGLE_AZIMUTHAL.sample((n_ions,))).to(device)*1e9
    r[2,ion_idx] = ztop.sample((n_ions,)).to(device)
    r[1,ion_idx] = ydist.sample((n_ions,)).to(device)
    r[0,ion_idx] = xdist.sample((n_ions,)).to(device)

    ## NEUTRALS Initialization

    n_neutrals=(sp[1]==1).nonzero().shape[0]
    if n_neutrals>0:
        print('n_neutrals',n_neutrals)
        neutral_idx=torch.t((sp[1]==1).nonzero())[0]
        neutral_gen_ratio= NEUTRAL_GEN_RATIO
        gen_posi=torch.multinomial(torch.tensor(neutral_gen_ratio), n_neutrals, replacement=True)
        neutrals_top=neutral_idx[torch.t((gen_posi==0).nonzero())[0]]
        neutrals_l=neutral_idx[torch.t((gen_posi==1).nonzero())[0]]
        neutrals_r=neutral_idx[torch.t((gen_posi==2).nonzero())[0]]
        neutrals_f=neutral_idx[torch.t((gen_posi==3).nonzero())[0]]
        neutrals_b=neutral_idx[torch.t((gen_posi==4).nonzero())[0]]
        v[:,neutral_idx] = velocity_dist_neutral.sample((3,n_neutrals)).to(device)
        r[0,neutrals_top] = xdist.sample((neutrals_top.shape[0],)).to(device)
        r[1,neutrals_top] = ydist.sample((neutrals_top.shape[0],)).to(device)
        r[2,neutrals_top] = ztop.sample((neutrals_top.shape[0],)).to(device)

        r[0,neutrals_l] = xdist.sample((neutrals_l.shape[0],)).to(device)
        r[1,neutrals_l] = yleft.sample((neutrals_l.shape[0],)).to(device)
        r[2,neutrals_l] = zdist.sample((neutrals_l.shape[0],)).to(device)

        r[0,neutrals_r] = xdist.sample((neutrals_r.shape[0],)).to(device)
        r[1,neutrals_r] = yright.sample((neutrals_r.shape[0],)).to(device)
        r[2,neutrals_r] = zdist.sample((neutrals_r.shape[0],)).to(device)

        r[0,neutrals_f] = xfront.sample((neutrals_f.shape[0],)).to(device)
        r[1,neutrals_f] = ydist.sample((neutrals_f.shape[0],)).to(device)
        r[2,neutrals_f] = zdist.sample((neutrals_f.shape[0],)).to(device)

        r[0,neutrals_b] = xback.sample((neutrals_b.shape[0],)).to(device)
        r[1,neutrals_b] = ydist.sample((neutrals_b.shape[0],)).to(device)
        r[2,neutrals_b] = zdist.sample((neutrals_b.shape[0],)).to(device)

    end = start+n_particles-1

    start = end +1
    # print('spnew.dtype',sp.dtype)
    return r, v, sp, start, end, n_particles