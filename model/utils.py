
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load your data from the pickle file
    import pickle
    ct1, ct2, ct3, data, data0, CATALYST_IDX, TEMP, SUP_IDX, mol = pickle.load(open("./Data/ocm_all_final.pickle", "rb"))
    
    # Prepare inflows and outflows
    inflows = np.array(data0[:, 1:8])  # flows
    outflows = np.array(data0[:, -7:])  # flows
    contactTime = data0[:, 0]
    
    return inflows, outflows, contactTime, CATALYST_IDX, TEMP, SUP_IDX, mol

def preprocess_data(inflows, outflows, TEMP, data):
    # Scale inflows and outflows
    scaler = StandardScaler().fit(data)
    inflowsScaled = scaler.transform(inflows)
    outflowsScaled = scaler.transform(outflows)
    
    # Scale temperature
    t_scaler = StandardScaler().fit(TEMP.reshape([-1, 1]))
    TEMP_SC = t_scaler.transform(TEMP.reshape([-1, 1]))

    return inflowsScaled, outflowsScaled, TEMP_SC

def create_train_test_splits(contactTime):
    ct1 = [i for i, v in enumerate(contactTime) if v == 0.38]
    ct2 = [i for i, v in enumerate(contactTime) if v == 0.5]
    ct3 = [i for i, v in enumerate(contactTime) if v == 0.75]

    np.random.shuffle(ct1)
    np.random.shuffle(ct2)
    np.random.shuffle(ct3)

    ct1Tr = ct1[:int(0.8 * len(ct1))]
    ct1Te = ct1[int(0.8 * len(ct1)):]
    ct2Tr = ct2[:int(0.8 * len(ct2))]
    ct2Te = ct2[int(0.8 * len(ct2)):]
    ct3Tr = ct3[:int(0.8 * len(ct3))]
    ct3Te = ct3[int(0.8 * len(ct3)):]

    return ct1Tr, ct1Te, ct2Tr, ct2Te, ct3Tr, ct3Te
