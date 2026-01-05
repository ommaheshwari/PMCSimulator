from Modules.Settings.settings import *
from Modules.Motion.Particle_structure_interaction.compare_array import compare_array
from Modules.Surface_voxels.exposed_faces import exposed_faces


def update_neighbour(face_idx, test_voxel, remove_voxel, top_voxels):


    test_voxel=torch.unique(test_voxel[face_idx,:], sorted=False, dim=0)
    vox_idx1,test_idx1 = compare_array(remove_voxel,test_voxel)
    mask1 = torch.ones(test_voxel.shape[0], dtype=torch.bool)
    mask1[test_idx1] = False
    test_voxel=test_voxel[mask1,:]
    vox_idx,test_idx = compare_array(top_voxels,test_voxel)
    mask = torch.ones(test_voxel.shape[0], dtype=torch.bool)
    mask[test_idx] = False
    add_voxel=test_voxel[mask,:]


    return add_voxel, vox_idx
