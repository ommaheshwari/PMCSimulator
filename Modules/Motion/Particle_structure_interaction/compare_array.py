from Modules.Settings.settings import *

def compare_array(array1, array2):
    _, idx, counts = torch.cat([array1, array2], dim=0).unique(
    dim=0, return_inverse=True, return_counts=True)
    # print('counts.gt(1)',counts.gt(1))
    # print('torch.where(counts.gt(1))[0]',torch.where(counts.gt(1))[0])
    # print('idx',idx)

    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    # print('mask',mask)
    mask1 = mask[:len(array1)]  # tensor([ True, False,  True], device='cuda:0')
    mask2 = mask[len(array1):]  # tensor([ True, False, False,  True], device='cuda:0')
    # print('mask1',mask1)
    # assert top_voxels[mask1].equal(new_cross[mask2])
    indices1 = torch.arange(len(mask1)).to(device)[mask1]  # tensor([0, 2])
    indices2 = torch.arange(len(mask2)).to(device)[mask2]  # tensor([0, 3])
    # print('i1',indices1,'i2',indices2)
    del _, idx, counts, mask, mask1, mask2, array1, array2
    torch.cuda.empty_cache()

    return indices1, indices2