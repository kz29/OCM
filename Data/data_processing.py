import datetime
import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import StepLR
import pickle

# Set device for PyTorch (use GPU if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Debugging tool
from IPython.core.debugger import set_trace

# Reading data from CSV and converting it to a pandas DataFrame
path = 'data/OCM-data.csv'
df = pd.read_csv(path)

# Calculating the flows from yields and removing data points where oxygen flow is non-positive
df = df.assign(FCH4out=(df['CH4_inflow'] * (1 - df[columns_list].sum(axis=1) / 100)))
df = df.assign(FARout=df['Ar_inflow'])
df = df.assign(FC2H6out=(0.5 * df['CH4_inflow'] * df['C2H6yield'] / 100))
df = df.assign(FC2H4out=(0.5 * df['CH4_inflow'] * df['C2H4yield'] / 100))
df = df.assign(FCOout=df['CH4_inflow'] * df['Coyield'] / 100)
df = df.assign(FCO2out=df['CH4_inflow'] * df['CO2yield'] / 100)

# Calculation of oxygen flow (FO2out) based on yields and CH4 inflow
calc = (0.25 * df['C2H6yield'] / 100 + 0.25 * df['C2H4yield'] / 100 + 1.5 * df['Coyield'] / 100 + 2 * df['CO2yield'] / 100)
prod = df['CH4_inflow'] * calc
df = df.assign(FO2out=(df['O2_inflow'] - prod))

# Remove rows where oxygen flow (FO2out) is less than or equal to 0
df = df.loc[df['FO2out'] > 0]

# Adjusting M1_mol% and calculating mol values for missing data
df["M1_mol"] = df["M1_mol%"]
tmp = df[df["M1_mol"] > 0]
MOL = (tmp["M2_mol"] + tmp["M3_mol"])
MOLP = (tmp["M2_mol%"] + tmp["M3_mol%"])
MOLM1 = 100 - MOLP
df["M1_mol"][df["M1_mol"] > 0] = MOLM1 * (MOL / MOLP)

# Create an index for each catalyst atom (M1, M2, M3)
allCatalAtoms = set(df.M1).union(set(df.M2)).union(set(df.M3))
catAtomMap = {}
for i, a in enumerate(allCatalAtoms):
    catAtomMap[a] = i

# Grouping indexes of metals M1, M2, M3 based on the catalyst they form
CATALYST_IDX = np.array([[catAtomMap[e] for e in r] for r in df[['M1', 'M2', 'M3']].values])

TEMP = df['Temperature (K)'].values

# Extract unique values from the 'Support ' column and create a mapping from support type to index
supp = set(df['Support '])
SupMap = {s: i for i, s in enumerate(supp)}
SUP_IDX = np.array([SupMap[s] for s in df['Support '].values])

# Extract columns related to molar values for metals M1, M2, M3
mol = df[['M1_mol', 'M2_mol', 'M3_mol']].values

# Extract catalyst names
Name = df['Name']

# Initialize lists for various flow values and other relevant columns (all initialized to 0)
FC2H6in = FC2H4in = FCOin = FCO2in = FC2in = FC3in = FC3O2in = FC2H2in = FC3O4 = FC2H8in = FC3O3 = [0]*9271

# Extract values for M1, M2, M3 metals from the DataFrame
M1 = df.M1.values.tolist()
M2 = df.M2.values.tolist()
M3 = df.M3.values.tolist()

# Data for calculations and additional features
data0 = np.column_stack((
    CT, Ar_F, CH4_F, O2_F, FC2H6in, FC2H4in, FCOin, FCO2in, FC2in, FC3in, FC3O2in, 
    FC2H2in, FC3O4, FC2H8in, FC3O3, M1, M2, M3, M1_m, M2_m, M3_m, T, FARout, 
    FCH4out, FO2out, FC2H6out, FC2H4out, FCOout, FCO2out
))

# Extract other required lists related to molecular percentages and metal compositions
M1_mp = df['M1_mol%'].values.tolist()
M2_mp = df['M2_mol%'].values.tolist()
M3_mp = df['M3_mol%'].values.tolist()
M2_m = df['M2_mol'].values.tolist()
M3_m = df['M3_mol'].values.tolist()
M1_m = df['M1_mol'].values.tolist()

# Extract temperature (in Kelvin)
T = df['Temperature (K)'].values.tolist()

# Extract inflows for different gases
Ar_F = df['Ar_inflow'].values.tolist()
CH4_F = df['CH4_inflow'].values.tolist()
O2_F = df['O2_inflow'].values.tolist()

# Calculate pressure from flow rate
PAr = [x/y for x, y in zip(Ar_F, tF)]  # Ensure tF is defined elsewhere

# Extract contact time (in seconds)
CT = df['Contact time (tau)'].values.tolist()

# Extract output values: conversions and yields for different compounds
xCH4 = df['CH4_conversion'].values.tolist()
yC2H6 = df['C2H6yield'].values.tolist()
yC2H4 = df['C2H4yield'].values.tolist()
yC2 = df['C2y'].values.tolist()
yCO = df['Coyield'].values.tolist()
yCO2 = df['CO2yield'].values.tolist()

# Extract output flows for different gases
FCH4out = df['FCH4out'].values.tolist()
FARout = df['FARout'].values.tolist()
FC2H6out = df['FC2H6out'].values.tolist()
FC2H4out = df['FC2H4out'].values.tolist()
FCOout = df['FCOout'].values.tolist()
FCO2out = df['FCO2out'].values.tolist()
FO2out = df['FO2out'].values.tolist()

# Split data based on unique contact times
t1, t2, t3 = ([data0[CT == i] for i in np.unique(CT)])

# Remove the first column from the datasets
t1 = np.delete(t1, 0, 1)
t2 = np.delete(t2, 0, 1)
t3 = np.delete(t3, 0, 1)

# Combine the inflow data with output data
data = np.column_stack((
    Ar_F + (FARout), CH4_F + (FCH4out), O2_F + (FO2out), FC2H6in + (FC2H6out), 
    FC2H4in + (FC2H4out), FCOin + (FCOout), FCO2in + (FCO2out)
))

# Pickle the data into a .pickle file for further use
pickle.dump([t1, t2, t3, data, data0, CATALYST_IDX, TEMP, SUP_IDX, mol], 
            open("./ocm_all_final.pickle", "wb"))
