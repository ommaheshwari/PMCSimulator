from Modules.Settings.settings import *

def time_db_voxel(init_dfx, name,  stfhit, structure):
    
    for ti in range ((TIME_STEPS//DB_STEPS)):
        stfr=stfhit.reshape((TIME_STEPS//DB_STEPS)+1,structure.shape[0],structure.shape[1],structure.shape[2])[ti+1,:,:,:].cpu().numpy()
        # print(stfr)
        if STORE_STATE==1:
            init_dfx['state_{}'.format(ti)]=1-(stfr.ravel()/init_dfx['Threshold'])
        if STORE_HITS==1:
            init_dfx['hits_{}'.format(ti)]=1-(stfr.ravel())
    table_name = name # table and file name

    conn = sq.connect('{}.sqlite'.format(table_name)) # creates file
    init_dfx.to_sql(table_name, conn, if_exists='replace', index=False) # writes to file
    conn.close() # good practice: close connection
    return

def time_db_particle(name, time, rsnew, vsnew, spnew, end_id ):
    t = time.cpu()
    dfx=pd.DataFrame()
    df = pd.DataFrame({"time": t ,"x" : rsnew[0,:].flatten().cpu(), "y" : rsnew[1,:].flatten().cpu(), "z" : rsnew[2,:].flatten().cpu(), "vx" : vsnew[0,:].flatten().cpu(), "vy" : vsnew[1,:].flatten().cpu(), "vz" : vsnew[2,:].flatten().cpu(), "sp" : spnew[0,:].flatten().cpu(),"ty" : spnew[1,:].flatten().cpu(),"refl" : spnew[3,:].flatten().cpu(), "id" :spnew[4,:].flatten().cpu() })
    dfx['ID'] = torch.arange(0, int(end_id+1))
    for tp in range ((TIME_STEPS//DB_STEPS)):
        temp_num=tp*DB_STEPS
        data=df[df['time']==temp_num]

        if STORE_POSITION == 1:
            dfx.loc[dfx.index[data.id.values.astype(np.uint64)], 'pos_{}'.format(temp_num)] = [str((data.x.values[pp], data.y.values[pp], data.z.values[pp])) for pp in range(data.x.values.shape[0])]
        if STORE_VELOCITY == 1:
             dfx.loc[dfx.index[data.id.values.astype(np.uint64)], 'v_{}'.format(temp_num)]=[str((data.vx.values[vpp], data.vy.values[vpp], data.vz.values[vpp])) for vpp in range(data.vx.values.shape[0])]
        if STORE_TYPE == 1:
             dfx.loc[dfx.index[data.id.values.astype(np.uint64)], 'type_{}'.format(temp_num)]=[str((PARTICLES[p-1], N_I_ID[p-1])) for p in np.array(data.sp, dtype=int)]
        if STORE_REFLECTIONS == 1:
             dfx.loc[dfx.index[data.id.values.astype(np.uint64)], 'refl_{}'.format(temp_num)]=data.refl.values
            

    table_name = name # table and file name
    conn = sq.connect('{}.sqlite'.format(table_name)) # creates file
    dfx.to_sql(table_name, conn, if_exists='replace', index=False) # writes to file
    conn.close() # good practice: close connection
    return
