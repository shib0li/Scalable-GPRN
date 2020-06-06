import numpy as np
from hdf5storage import loadmat
from hdf5storage import savemat
from sklearn.preprocessing import StandardScaler
import os

def process(domain, save=False):

    data = {}
    data['dname'] = domain
    
    if domain == 'jura':
        filename = 'data/jura/jura.mat'
        raw = loadmat(filename, squeeze_me=True, struct_as_record=False, mat_dtype=True)['jura']

        X_all = raw[:, 0:2] # X, Y
        Y_all = raw[:, [2, -2, -1]] # Cd, Ni, Zn
        N_all = X_all.shape[0]
        
        N_train = 249
        N_test = 100
        
        DList = [3,1,1] 

        print('Xall shape:', X_all.shape)
        print('Yall shape:', Y_all.shape)
    else:
        print('No valid dataset found... program terminated...')
        return

    scaler = StandardScaler()
    scaler.fit(X_all)
    N_X_all = scaler.transform(X_all)
    X_mean = scaler.mean_
    X_std = scaler.scale_

    scaler.fit(Y_all)
    N_Y_all = scaler.transform(Y_all)
    Y_mean = scaler.mean_
    Y_std = scaler.scale_

    perm = np.random.permutation(N_all)

    N_X_all = N_X_all[perm]
    N_Y_all = N_Y_all[perm]

    X_train = N_X_all[0:N_train, :]
    X_test = N_X_all[-N_test:, :]

    Y_train = N_Y_all[0:N_train, :]
    Y_test = N_Y_all[-N_test: , :]

    data['N_all'] = N_all
    data['N_train'] = N_train
    data['N_test'] = N_test

    data['X_all'] = X_all
    data['Y_all'] = Y_all
    data['DList'] = DList

    data['X_train'] = X_train
    data['X_test'] = X_test
    data['X_mean'] = X_mean
    data['X_std'] = X_std

    data['Y_train'] = Y_train
    data['Y_test'] = Y_test
    data['Y_mean'] = Y_mean
    data['Y_std'] = Y_std
    data['Y_test_ground'] = Y_test * Y_std + Y_mean
    
    if save==True:
        if not os.path.exists('processed'):
            os.makedirs('processed')
        savemat('processed/' + data['dname'] + '.mat', data)
        print('saved to', 'processed/' + data['dname'] + '.mat')
        
    return data