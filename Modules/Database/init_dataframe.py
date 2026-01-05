from Modules.Settings.settings import *



def init_dataframe(structured,db):
    structured=structured.cpu()
    #DB
    name_list=[]
    mat_list=[]
    lay_list=[]
    mass_list=[]
    vth_list=[]
    layers=LAYER_NAMES
    shp = structured.shape
    x,y,z= np.indices(shp)

    for i in range(0,structured.ravel().shape[0]):
        if structured.ravel()[i]==0:
            mat='Air'
            lay='Etched'
            vth=0
            mass=0
        elif structured.ravel()[i]==1:
            lay=layers[0]
            mat=db[layers[0]]['Material']
            vth=db[layers[0]]['Voxel_Threshold']
            mass=db[layers[0]]['Mass']
            
        elif structured.ravel()[i]==2:
            lay=layers[1]
            mat=db[layers[1]]['Material']
            vth=db[layers[1]]['Voxel_Threshold']
            mass=db[layers[1]]['Mass']
        elif structured.ravel()[i]==3:
            lay=layers[2]
            mat=db[layers[2]]['Material']
            vth=db[layers[2]]['Voxel_Threshold']
            mass=db[layers[2]]['Mass']        
        elif structured.ravel()[i]==4:
            lay=layers[3]
            mat=db[layers[3]]['Material']
            vth=db[layers[3]]['Voxel_Threshold']
            mass=db[layers[3]]['Mass']        

        n1='V_'+str(lay)+'_'+str(mat)+'_'+str(x.ravel()[i])+'_'+str(y.ravel()[i])+'_'+str(z.ravel()[i])
        mat_list.append(mat)
        lay_list.append(lay)
        vth_list.append(vth)
        mass_list.append(mass)
        name_list.append(n1)


    df=pd.DataFrame(np.c_[name_list,x.ravel(), y.ravel(), z.ravel(), structured.ravel(), lay_list,mat_list,mass_list,np.array(vth_list),structured.ravel()/structured.ravel()], \
                                    columns=((['Voxel_name','x','y','z','val','Layer_Name','Material','Mass','Threshold','State t=0'])))
    df['State t=0'] = df['State t=0'].replace('nan', 0)
    df = df.astype({'Voxel_name': 'string', 'x': 'float', 'y': 'float', 'z': 'float','val': 'float','Layer_Name':'string','Material':'string','Mass':'float','Threshold':'float','State t=0':'float'})


    return df