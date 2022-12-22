# -*- coding: utf-8 -*-

import numpy as np
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import pandas as  pd

from pymatgen import Structure
from pymatgen import Composition
from pymatgen.analysis import structure_analyzer, structure_matcher
import joblib
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import json

from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils import *


#%% querying material project database to get cif files and material properties

#please use your own api key for querying MP.org
mat_api_key = '5iSAlXJeOTGq30v2'

sup_prop = ['formation_energy_per_atom']
semi_prop = ['Powerfactor']

num_ele=3
num_sites = 20

def get_data (mat_api_key, nsites=41, least_ele = 1, most_ele= 6):

    mpdr = MPDataRetrieval(mat_api_key)
    #e_above_hull and formation energy per atom are the upper limit to get stable compounds
    df=mpdr.get_dataframe(criteria={
                                    'nsites':{'$lt':nsites},
                                    'e_above_hull':{'$lt':-1},
                                   
                                    'nelements': {'$gt':least_ele,'$lt':most_ele} ,
                                    
                                     },properties=['material_id','formation_energy_per_atom','density','spacegroup.crystal_system','volume','energy_per_atom','icsd_ids','band_gap','pretty_formula','e_above_hull','elements','cif','spacegroup.number'])
    
    
    df['ind'] = np.arange(0,len(df),1)
    
    #load the thermoelectric calculations dataset from the csv file
    
    df_m = pd.read_csv('df_power.csv',index_col=0)
    
    #df_m = np.log10(np.abs(df_m))
    df_m = df_m.dropna()
    
    #select compounds that has both ground state properties and BTE calculations
    i = df.index.intersection(df_m.index)
    

    df_in = pd.concat([df.iloc[:,:],df_m.loc[i,:]],1)
    
#    df_in = df_in.dropna()
    
    
    df_in['Seebeck'] = np.abs(df_in['Seebeck'] )
    return df, df_in

#df1,df_in1 = get_data (mat_api_key, nsites=21, least_ele = 4, most_ele= 6)


#%%

#function for featurizing crystal representation using cif files from MP.org
def  crystal_represent(df,num_ele=3,num_sites=20):

    Element= joblib.load('./files/element.pkl')
       
    E_v = np_utils.to_categorical(np.arange(0,len(Element),1))

    
    
    
    elem_embedding_file = './files/atom_init.json'
    with open(elem_embedding_file) as f:
    			elem_embedding = json.load(f)
    elem_embedding = {int(key): value for key, value
    						  in elem_embedding.items()}
    feat_cgcnn = []
    
    for key,value in elem_embedding.items():
        feat_cgcnn.append(value)
        
    feat_cgcnn = np.array(feat_cgcnn)
   
    #start featurization
    
    ftcp = []


    
    for idx in range(len(df)): #46382
        
        crystal = Structure.from_str(df['cif'][idx],fmt="cif")
        
        latt = crystal.lattice

        ui, ux, uy = np.unique(crystal.atomic_numbers,return_index=True,return_inverse= True)
        z_sorted=np.array(crystal.atomic_numbers)

        z_u = z_sorted[np.sort(ux)]
     
        onehot = np.zeros((num_ele,len(E_v)))
        onehot[:len(z_u),:] = E_v[z_u-1,:]
        fc1 =np.zeros((num_sites,3))
        fc1_ind = np.zeros((num_sites,num_ele))
        #Fourier space, 1.2 is used at the maximum distance
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.2)
        zs = []
        coeffs = []
        fcoords = []
        coords = []
        occus = []
        
     
        for site in crystal:
           
            for sp, occu in site.species.items():
                zs.append(sp.Z)
                c= feat_cgcnn[sp.Z-1,:]
                
                coeffs.append(c)
                fcoords.append(site.frac_coords)
                
                occus.append(occu)
                coords.append(site.coords)
                
            
        zs = np.array(zs)
        coeffs = np.array(coeffs)
        
        coeffs_crsytal = np.zeros((num_ele,feat_cgcnn.shape[1]))
        
        coeffs_crsytal [:len(z_u),:] = coeffs[np.sort(ux),:]
        coords = np.array(coords)
        
        fcoords = np.array(fcoords)
        
        fc1[:fcoords.shape[0],:]= fcoords
        occus = np.array(occus).reshape(-1,1)
        
        abc1 = np.asarray(latt.abc)
        ang1 = np.asarray(latt.angles)
      
        for i in range(len(z_u)):
             fc1_ind[np.where(z_sorted==z_u[i]),i]=1
      
        crys_list = np.concatenate((abc1.reshape(-1,1),
                                ang1.reshape(-1,1),fc1.T),axis=1)
        
        crys_list1 = np.zeros((num_ele, crys_list.shape[1]))
        crys_list1 [:crys_list.shape[0],:] = crys_list
        
        #real space represeatnion
        atom_list = np.concatenate((onehot,crys_list1,fc1_ind.T,np.zeros((num_ele,1)),coeffs_crsytal),axis=1)
        
        atom_list = atom_list.T
    
        hkl_list = []        
        ftcp_list = []
        i=0
  
        for hkl, g_hkl, ind, _ in sorted(
                    recip_pts, key=lambda i: (abs(i[0][0])+abs(i[0][1])+abs(i[0][2]), -i[0][0], -i[0][1], -i[0][2])):
            
                       # Force miller indices to be integers.
            hkl = [int(round(i)) for i in hkl]
            
            i+=1
                   
            
            if g_hkl != 0 and i < 61:
                hkl_list.append(hkl)

    
                # Vectorized computation of g.r for all fractional coords and
                
                g_dot_r = np.dot(fcoords, np.transpose([hkl])).reshape(-1,1)
                
    
              
                # Vectorized computation.
                f_hkl = np.sum(np.sin(2*np.pi*(occus * np.pi * g_dot_r*coeffs)), axis=0)
    #            z_hkl = np.sum(occus*g_dot_r*zs,axis=0)
                
              
                f_hkl1  = np.insert(f_hkl,1,g_hkl)
                f_hkl1 = np.concatenate((np.zeros((atom_list.shape[0]
                -coeffs.shape[1]-1,1)),f_hkl1.reshape(-1,1) ))
                
                ftcp_list.append(f_hkl1)              

        #Fourier space representations        
        ftcp_list = np.stack(ftcp_list,axis=1) 
        ftcp_list = np.squeeze(ftcp_list,axis=2)
        
        ftcp_list = np.concatenate((atom_list,ftcp_list),axis=1)
             
        ftcp.append(ftcp_list)  
    
        
    return ftcp


def atomic_represent(df, num_ele=3, num_sites=20):

    crystal_list = df['pretty_formula'].tolist()
    Element = joblib.load('./files/element.pkl')

    df1 = pd.read_csv('files/atomic_features.csv')

    E_v = np_utils.to_categorical(np.arange(0, len(Element), 1))


    elem_embedding_file = './files/atom_init.json'
    with open(elem_embedding_file) as f:
        elem_embedding = json.load(f)
    elem_embedding = {int(key): value for key, value
                      in elem_embedding.items()}
    feat_cgcnn = []

    for key, value in elem_embedding.items():
        feat_cgcnn.append(value)

    feat_cgcnn = np.array(feat_cgcnn)

    # start featurization

    atomic = []

    for x in range(len(crystal_list)):

        crystal = Composition(crystal_list[x])
        size = len(crystal.elements)

        z_u = np.array([crystal.elements[i].number for i in range(size)])

        onehot = np.zeros((num_ele, len(E_v)))
        onehot[:len(z_u), :] = E_v[z_u - 1, :]

        coeffs_crsytal = np.zeros((num_ele, feat_cgcnn.shape[1]))
        for i in range(len(z_u)):
            coeffs_crsytal[i, :] = feat_cgcnn[z_u[i] - 1, :]

        dic = crystal.get_el_amt_dict()
        ratio_ = np.array([dic[crystal.elements[i].symbol] for i in range(size)])
        ratio_ /= crystal._natoms
        ratio = np.zeros((num_ele, 1))
        ratio[:len(z_u), 0] = ratio_
        ratio = ratio.reshape(1, num_ele)
        ratio1 = ratio * crystal._natoms

        # real space represeatnion
        atom_list = np.concatenate(
            ((onehot.T * ratio).T, ratio1.T, ratio.T, np.zeros((num_ele, 1)), (coeffs_crsytal.T * ratio).T), axis=1)
        atom_list = atom_list.T

        atomic.append(atom_list)

    return atomic


def crystal_represent_2(df, num_ele=3, num_sites=20):
    Element = joblib.load('./files/element.pkl')

    E_v = np_utils.to_categorical(np.arange(0, len(Element), 1))

    df1 = pd.read_csv('files/atomic_features.csv')



    elem_embedding_file = './files/atom_init.json'
    with open(elem_embedding_file) as f:
        elem_embedding = json.load(f)
    elem_embedding = {int(key): value for key, value
                      in elem_embedding.items()}
    feat_cgcnn = []

    for key, value in elem_embedding.items():
        feat_cgcnn.append(value)

    feat_cgcnn = np.array(feat_cgcnn)

    # start featurization

    ftcp = []

    for idx in range(len(df)):  # 46382

        crystal = Structure.from_str(df['cif'][idx], fmt="cif")

        latt = crystal.lattice

        ui, ux, uy = np.unique(crystal.atomic_numbers, return_index=True, return_inverse=True)
        z_sorted = np.array(crystal.atomic_numbers)

        z_u = z_sorted[np.sort(ux)]

        onehot = np.zeros((num_ele, len(E_v)))
        onehot[:len(z_u), :] = E_v[z_u - 1, :]
        fc1 = np.zeros((num_sites, 3))
        fc1_ind = np.zeros((num_sites, num_ele))
        # Fourier space, 1.2 is used at the maximum distance
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.2)
        zs = []
        coeffs = []
        fcoords = []
        coords = []
        occus = []

        for site in crystal:

            for sp, occu in site.species.items():
                zs.append(sp.Z)
                c = feat_cgcnn[sp.Z - 1, :]

                coeffs.append(c)
                fcoords.append(site.frac_coords)

                occus.append(occu)
                coords.append(site.coords)

        zs = np.array(zs)
        coeffs = np.array(coeffs)

        coeffs_crsytal = np.zeros((num_ele, feat_cgcnn.shape[1]))

        coeffs_crsytal[:len(z_u), :] = coeffs[np.sort(ux), :]
        coords = np.array(coords)

        fcoords = np.array(fcoords)

        fc1[:fcoords.shape[0], :] = fcoords
        occus = np.array(occus).reshape(-1, 1)

        abc1 = np.asarray(latt.abc)
        ang1 = np.asarray(latt.angles)

        for i in range(len(z_u)):
            fc1_ind[np.where(z_sorted == z_u[i]), i] = 1

        crys_list = np.concatenate((abc1.reshape(-1, 1),
                                    ang1.reshape(-1, 1), fc1.T), axis=1)

        crys_list1 = np.zeros((num_ele, crys_list.shape[1]))
        crys_list1[:crys_list.shape[0], :] = crys_list

        # calculate the ratio of each element in a compound
        ratio = np.sum(fc1_ind, axis=0)
        ratio = ratio / np.sum(ratio)
        ratio = ratio.reshape(1, 3)

        # added elemental features (Raymond)
        ele_feature = np.zeros((num_ele, 56))
        for i, x in enumerate(z_u):
            array = df1[df1['Number'] == x].to_numpy()[:, 1:]
            if array.shape == (1, 56):
                ele_feature[i, :] = array

        # real space represeatnion
        atom_list = np.concatenate((onehot, crys_list1, fc1_ind.T, np.zeros((num_ele, 1)), coeffs_crsytal), axis=1)
        #         atom_list = np.concatenate((onehot,ratio.T,np.zeros((num_ele,1)),coeffs_crsytal),axis=1)

        atom_list = atom_list.T

        ftcp.append(atom_list)

    return ftcp


def plot_confusion_matrix(model_, x_data, y_data, file_name):
    y_test_result = model_.predict(x_data)
    y_test_result1 = []
    for i in y_test_result:
        if i > 0.5:
            y_test_result1.append(1)
        else:
            y_test_result1.append(0)

    #     y_data1 = []
    #     for i in y_data[:,0:1]:
    #         if i >0.5:
    #             y_data1.append(1)
    #         else:
    #             y_data1.append(0)
    #     print(y_data[:,0:1])

    cm = metrics.confusion_matrix(y_data, y_test_result1, normalize='true')
    cm_ = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])
    #     print(cm)
    #     print(cm_)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5.5)

    SMALL_SIZE = 12
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 25

    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    y_label = ['Sythesizable', 'Not Sythesizable']
    x_label = ['Have ICSD ID', 'Not Have ICSD ID']

    cm2 = metrics.confusion_matrix(y_data, y_test_result1)

    im, cbar = heatmap(np.transpose(cm_) * 100, y_label, x_label, ax=ax,
                       cmap="YlGn", cbarlabel="Percentage of all True Condition samples")
    ax.set_ylabel('Predicted Condition')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('True Condition')

    texts = annotate_heatmap(im, valfmt="{x:.1f} %")

    fig.tight_layout()
    fig.savefig('model_result/{}.png'.format(file_name))
    plt.show()
    cm2 = metrics.confusion_matrix(y_data, y_test_result1)


    print('Test Precision:', cm2[1, 1] / (cm2[1, 1] + cm2[0, 1]))
    print('Test Recall:', cm[1, 1])
#     return cm2[1,1]/(cm2[1,1]+cm2[0,1]),cm[1,1]