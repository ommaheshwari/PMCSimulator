from Modules.Settings.settings import *

def generate_particles(structure,species,hit_weight,pion=1,pneutral=0, count=0):
    ## 0> ion, 1>neutral
    start = count

 

    # velocity_dist_ion=torch.distributions.Normal(0,-velocity*0.1) #Deciding Angular Distribution
    # velocity_distz_ion=torch.distributions.Normal(velocity,-velocity*0.1) #Deciding Energy Distribution
    velocity_dist_neutral=torch.distributions.Normal(0,V_NEUTRAL) 



    # x,y= np.indices([20,20])
    x, y = np.meshgrid(range(-EXTRA_SOURCE,structure.shape[0]+EXTRA_SOURCE), range(-EXTRA_SOURCE,structure.shape[1]+EXTRA_SOURCE))
    r = torch.tensor(np.array([x.ravel(),y.ravel(),x.ravel()])).to(device)+torch.rand(1)[0]
    r[2]=structure.shape[2]
    n_particles=r.shape[1]
    # print('n_particles', n_particles)
    sp=  torch.zeros((5,n_particles), dtype=torch.int8).to(device) #Index for species [species, type, hit_weight, reflections_count]
    sp[0] = species*torch.ones((1,n_particles))
    sp[1] = torch.multinomial(torch.tensor([pion,pneutral]), n_particles, replacement=True)

    sp[2] = hit_weight*torch.ones((1,n_particles))
    # sp[4] = torch.arange(start, start+n_particles).type(torch.int64)

    v = torch.zeros((3,n_particles)).to(device)
    # print('MASS[sp[sp[1]==0][0]]',MASS[species-1])
    v[2,torch.t((sp[1]==0).nonzero())[0]] = -torch.sqrt(2*ENERGY.sample(((sp[1]==0).nonzero().shape[0],))*Q/MASS[species-1]).to(device)*torch.cos(ANGLE_POLAR.sample(((sp[1]==0).nonzero().shape[0],))).to(device)*1e9
    v[0,torch.t((sp[1]==0).nonzero())[0]] = -torch.sqrt(2*ENERGY.sample(((sp[1]==0).nonzero().shape[0],))*Q/MASS[species-1]).to(device)*torch.sin(ANGLE_POLAR.sample(((sp[1]==0).nonzero().shape[0],))).to(device)*torch.sin(ANGLE_AZIMUTHAL.sample(((sp[1]==0).nonzero().shape[0],))).to(device)*1e9
    v[1,torch.t((sp[1]==0).nonzero())[0]] = -torch.sqrt(2*ENERGY.sample(((sp[1]==0).nonzero().shape[0],))*Q/MASS[species-1]).to(device)*torch.sin(ANGLE_POLAR.sample(((sp[1]==0).nonzero().shape[0],))).to(device)*torch.cos(ANGLE_AZIMUTHAL.sample(((sp[1]==0).nonzero().shape[0],))).to(device)*1e9


    # v[2,torch.t((sp[1]==0).nonzero())[0]] = velocity_distz_ion.sample(((sp[1]==0).nonzero().shape[0],)).to(device)
    # v[0,torch.t((sp[1]==0).nonzero())[0]] = velocity_dist_ion.sample(((sp[1]==0).nonzero().shape[0],)).to(device)
    # v[1,torch.t((sp[1]==0).nonzero())[0]] = velocity_dist_ion.sample(((sp[1]==0).nonzero().shape[0],)).to(device)

    # print(v[:,torch.t((sp[1]==1).nonzero())[0]].shape)
    # print(velocity_dist_neutral.sample((3,(sp[1]==1).nonzero().shape[0])).to(device).shape)
    v[:,torch.t((sp[1]==1).nonzero())[0]] = velocity_dist_neutral.sample((3,(sp[1]==1).nonzero().shape[0])).to(device)
    print('torch.t((sp[1]==1).nonzero())[0]',torch.t((sp[1]==1).nonzero())[0])
    # v[0,(sp[1]==0)] = velocity_dist_ion.sample((n_particles,))
    # v[1,(sp[1]==0)] = velocity_dist_ion.sample((n_particles,))
    # v[:,(sp[1]==1)] = velocity_dist_neutral.sample((n_particles,3))
    end = start+n_particles-1

    start = end +1
    # print('spnew.dtype',sp.dtype)
    return r, v, sp, start, end, n_particles