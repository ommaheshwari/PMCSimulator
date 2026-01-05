from Modules.Settings.settings import *

def initial_count(init_structure):
    Initial_count=torch.empty(len(LAYER_NAMES))
    for i in range(len(LAYER_NAMES)):
        count=0
        count= (init_structure.flatten()==i+1).nonzero()
        Initial_count[i]=count.shape[0]
    print(Initial_count)
    return Initial_count

def percentage_etched(Initial_count, structure):
    Layer_count=torch.empty(len(LAYER_NAMES))
    for i in range(len(LAYER_NAMES)):
        count=0
        count= (structure.flatten()==i+1).nonzero()
        Layer_count[i]=count.shape[0]

    percentage=100*(Initial_count-Layer_count)/(Initial_count)

    return percentage

