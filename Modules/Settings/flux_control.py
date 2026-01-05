
def flux_control(flux, gen_steps, stime, DT, VOXEL_PER_UNIT):
     ## needs area factor somewhere
    max_steps_per_sec=1/DT
    voxels_per_cm2 = 1E14*VOXEL_PER_UNIT*VOXEL_PER_UNIT
    max_steps_per_particle = 1/(flux*DT/voxels_per_cm2)
    map_time = max_steps_per_particle/max_steps_per_sec
    steps_per_sec = gen_steps/map_time
    steps = steps_per_sec * stime
    return int(steps)