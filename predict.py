import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import  optimizers
from pymatgen import Composition
import pandas as pd
from utils import *
import featurizer
from featurizer import *
import numpy as np
import joblib
import argparse
import sys
import warnings
import gzip
import os.path



warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SC_model'
                                             '')
parser.add_argument('--crystal', default=0)
parser.add_argument('--type', choices=['formula', 'cif', 'quaternary_cif'],
                    default='formula')

def main():
    args = parser.parse_args(sys.argv[1:])

    if args.type == 'cif':
        print('---------Loading Model and Predicting---------------')
        model = tf.keras.models.load_model('./trained_models/SC_realspace')
        predict_list = [args.crystal]
        predict_crystals = crystal_represent_3(predict_list)
        a = np.stack(predict_crystals, axis=0)
        X = a
        X = pad(X, 2)
        y_result = model.predict(X)
        print('')

        print('---------Printing Result---------------')
        crystal_ = Structure.from_file(args.crystal)
        formula = crystal_.composition.reduced_formula
        print('The SC for CIF[{}] is {}'.format(formula,y_result[0]))
        print('')

    if args.type == 'quaternary_cif':
        print('---------Loading Model and Predicting---------------')
        print('Warning: This is the SC model for quaternary crystal structures (CIF) with number of sites lower than 40')

        if not os.path.isfile('trained_models/SC_quaternary/variables/variables.data-00000-of-00001'):
            with open('trained_models/SC_quaternary/variables/variables.data-00000-of-00001.gz.001', 'rb') as f:
                file_content = f.read()
            with open('trained_models/SC_quaternary/variables/variables.data-00000-of-00001.gz.002', 'rb') as f:
                file_content += f.read()
            weight = gzip.decompress(file_content)
            with open('trained_models/SC_quaternary/variables/variables.data-00000-of-00001', 'wb') as f:
                f.write(weight)

        model = tf.keras.models.load_model('./trained_models/SC_quaternary')
        predict_list = [args.crystal]
        predict_crystals = crystal_represent_3(predict_list, num_ele=4, num_sites=40)
        a = np.stack(predict_crystals, axis=0)
        X = a
        X = pad(X, 2)
        y_result = model.predict(X)
        print('')

        print('---------Printing Result---------------')
        crystal_ = Structure.from_file(args.crystal)
        formula = crystal_.composition.reduced_formula
        print('The SC for CIF[{}] is {}'.format(formula,y_result[0]))
        print('')


    elif args.type == 'formula':

        print('---------Loading Model and Predicting---------------')
        model = tf.keras.models.load_model('./trained_models/SC_atomic')
        predict_list = [args.crystal]
        predict_crystals = atomic_represent2(predict_list)
        a = np.stack(predict_crystals, axis=0)
        X = a
        X = pad(X, 2)
        y_result = model.predict(X)
        print('')

        print('---------Printing Result---------------')
        print('The atomic SC for [{}] is {}'.format(predict_list[0],y_result[0]))
        print('')


def atomic_represent2(crystal_list, num_ele=3, num_sites=20):

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


def crystal_represent_3(cif_list, num_ele=3, num_sites=20):
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

    for idx in range(len(cif_list)):  # 46382

        crystal = Structure.from_file(cif_list[idx])

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
        ratio = ratio.reshape(1, num_ele)

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

if __name__ == '__main__':
    main()