from Modules.Settings.settings import *


def update_hitlist(hitstructure, idx, xt, yt, zt, hitinc,process):
    # print('hit[idx] before', hit[idx])
    if process=='etch':
        i=0
        hitstructure[i,xt,yt,zt]=hitstructure[i,xt,yt,zt]+hitinc     #updating the number of hits of voxel in whole structure

    else:
        i=1
        hitstructure[i,xt,yt,zt]=hitstructure[i,xt,yt,zt]+hitinc     #updating the number of hits of voxel in whole structure
        
    return hitstructure

