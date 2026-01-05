from Modules.Settings.module_import import *
from Modules.Settings.flux_control import flux_control

plt.style.use(['science', 'notebook'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

#GLOBAL VARIABLES

# STRUCTURE VARIABLES
Q = 1.602e-19
SI_MASS = 28
RESOLUTION = 0.25
VOXEL_PER_UNIT = int(1/RESOLUTION)
WIDTH = 40 #Width (nm)
LENGTH = 100 #Length (nm)
GRID_W = WIDTH * VOXEL_PER_UNIT #Voxel Grid Size
GRID_L = LENGTH * VOXEL_PER_UNIT #Voxel Grid Size
GAS_HEIGHT = 3  #height of gas over top layer
DEPTH = 120 #Boolean etch depth
WIDTH_OPENING = WIDTH*0.6
LENGTH_OPENING = LENGTH*0.6
RADIUS_OPENING = 20

# LAYER VARIABLES
LAYER_NAMES = ['Si','SiGe','Spacer','BOX'] 
MATERIAL_NAMES = ['Si','SiGe','Spacer','BOX']
LAYER_THICKNESS = [360, 120] #Not used for NSFET
LAYER_ETCH_THRESHOLD = [6, 1, 40,1000,30] #Adjusts Threshold of Voxel
DEPOSIT_ID = 5
LAYER_DEPOSIT_THRESHOLD = [25, 2000, 10000,10000,10000] #Adjusts Threshold of Voxel
# DEPOSITION_THRESHOLD = 1
DEPOSITED_ETCH_THRESHOLD = 1


ADSORB_ID = 6
ADSORBED_ETCH_THRESHOLD = 25
ADSORBED_DEPOSIT_THRESHOLD = 10

# PARTICLE VARIABLES


PARTICLES = ['TCl4']
IDENTIFIER_VALUE = [1]
MASS = [35*1.66e-27]
PERIODIC_X=1
PERIODIC_Y=0
REFLECTMAX=5

EXTRA_SOURCE_X0 = 0
EXTRA_SOURCE_X1 = 0
EXTRA_SOURCE_Y0 = 5
EXTRA_SOURCE_Y1 = 0
NEUTRAL_GEN_RATIO=[0.0,1.0,0.0,0.0,0.0] #[top, left, right, front, back] generation percentage (Sum must be 1)
#PARTICLE AGGREGATION
particle_aggregation=1
# total_steps=1980000/5
# TIME_STEPS = int(total_steps/particle_aggregation)
# TIME_STEPS = 1600
DB_STEPS = 25000000     #Store Data in DB after every -- steps
SCOPE_STEPS = 10000000
HIT_WEIGHT = [particle_aggregation,particle_aggregation] #Related to Particle Aggregation


ENERGY = torch.distributions.Normal(700,100) #eV
# ANGLE_POLAR = torch.distributions.Normal(0,5*3.14159/180) #rad
ANGLE_POLAR = torch.distributions.Normal(0*3.14159/180,3*3.14159/180) #rad

# ANGLE_POLAR = torch.distributions.Normal(0,1e-20) #rad
ANGLE_AZIMUTHAL = torch.distributions.Normal(0,3.14159) #rad
# ANGLE_AZIMUTHAL = torch.distributions.Normal(0,1e-20) #rad

# VELOCITY = [-10000,-10000]
# T=300
# K=1.380649e-23
# m=35.5*1.67377e-27
# V_NEUTRAL= ((3*K*T/m)**(1/2))*1e9 #nm/s

# V_NEUTRAL= 10000*1e9 #nm/s
ION_NEUTRAL_RATIO = [0.0]
GENERATION_STEPS = [1]
# N_PARTICLES = 100
REACTIONS = {"Sputter":{},"Reflection":{},"Chemical_Etch":{}}


                             #Cl  
P_REFLECTION = torch.tensor([[[0.0],   #Si, ion    # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                             [0.0], #SiGe
                             [0.0], #Spacer
                             [0.0], #BOX
                             [0.0], #Deposited
                             [0.0]], #Adsorbed  
                             [[0.9],   #Si, neutral    # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species   #PR, neutral
                             [0.1],
                             [1.0],
                             [1.0],
                             [1.0],
                             [1.0]]]).to(device)   #PR, neutral
P_ETCH = torch.tensor([[[0.00],       # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]],
                        [[1.0],       # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]]]).to(device)
P_DEPOSIT = torch.tensor([[[0.0],       # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]],
                        [[0.0],       # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]]]).to(device)

P_ADSORB = torch.tensor([[[0.00],       # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]],
                        [[0.0],       # DIMENSIONS: [0]: Ion/Neutral [1]: Material [2]: Species
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0]]]).to(device)


# MOTION VARIABLES
DT = 1e-13          #sec
RADIUS = 0.03
# FLUX = 1E18 # flux/cm2/sec
STIME = 1 # Simulation time in seconds
# TIME_STEPS = flux_control(FLUX, max(GENERATION_STEPS), STIME, DT, VOXEL_PER_UNIT)


#DATABASE VARIABLES

#VOXEL DB 
VOXEL_DB_NAME='voxel_db2'
STORE_STATE = 1
STORE_HITS = 0
#PARTICLE DB
PARTICLE_DB_NAME='particle_db2'
STORE_POSITION = 1
STORE_VELOCITY = 0
STORE_TYPE = 0
STORE_REFLECTIONS = 1
N_I_ID = ['Ion','Neutral']

#ANIMATION
# ANIMATION_NAME = 'animation_'+str(TIME_STEPS)+'circle_mask'


# RUN TO RUN VARIABILITY
VARIABILITY = 0
if VARIABILITY == 0:
    torch.manual_seed(10)
