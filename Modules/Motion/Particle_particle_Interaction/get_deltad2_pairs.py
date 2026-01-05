from Modules.Settings.settings import *


def get_deltad2_pairs(r, ids_pairs): #find distance between eaach particle pair
    dx = torch.diff(torch.stack([r[0][ids_pairs[:,0]], r[0][ids_pairs[:,1]]]).T).squeeze()
    dy = torch.diff(torch.stack([r[1][ids_pairs[:,0]], r[1][ids_pairs[:,1]]]).T).squeeze()
    dz = torch.diff(torch.stack([r[2][ids_pairs[:,0]], r[2][ids_pairs[:,1]]]).T).squeeze()

    return dx**2 + dy**2 + dz**2