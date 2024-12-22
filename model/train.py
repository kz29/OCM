
import torch
import torch.optim as optim
from tqdm.notebook import tqdm
from ocm_model import OCMmodel
from utils import load_data, preprocess_data, create_train_test_splits
import numpy as np

# Hyperparameters
FEATURE_EMBED = 64
input_size = 7
learning_rate = 1e-3
batch_size = 64
num_epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
inflows, outflows, contactTime, CATALYST_IDX, TEMP, SUP_IDX, mol = load_data()
inflowsScaled, outflowsScaled, TEMP_SC = preprocess_data(inflows, outflows, TEMP, data)

# Create train/test splits
ct1Tr, ct1Te, ct2Tr, ct2Te, ct3Tr, ct3Te = create_train_test_splits(contactTime)

# Convert data to torch tensors
xInflow = torch.tensor(inflowsScaled).float().cuda()
yOutflow = torch.tensor(outflowsScaled).float().cuda()
xCatalistMol = torch.tensor(mol).float().cuda()
xTEMP = torch.tensor(TEMP_SC).float().reshape([-1, 1]).cuda()

# Setup model and optimizer
model = OCMmodel(input_size, FEATURE_EMBED).cuda()
criterion = torch.nn.MSELoss()
optimAdam = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimAdam, step_size=1000, gamma=0.1)

# Training loop
train_losses = []
TrE = []
TsE = []

def evalModel(xInflow, xCatAtom, xCatMol, xTemp, xSup, ct, y_tr):
    model.eval()
    pred = model(xInflow, xCatAtom, xCatMol, xTemp, xSup, ct)
    target = y_tr 
    train = criterion(pred, target).item()
    preds = pred.cpu().data.numpy()
    targets = target.cpu().data.numpy()
    model.train()
    return np.array([train])

for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    k = 0
    np.random.shuffle(ct1Tr)
    np.random.shuffle(ct2Tr)
    np.random.shuffle(ct3Tr)
    
    for it in range(len(ct1Tr) // batch_size):
        # Training logic
        pass

    # Calculate train and test evaluations
    trEval, tsEval = computeTrainAndTest(t1, t2, t3)

    TrE.append(trEval)
    TsE.append(tsEval)
    train_losses.append(running_loss / k)
    
    print(f'Epoch {epoch}, Train Loss: {running_loss / k}, Train Eval: {trEval}, Test Eval: {tsEval}')
    scheduler.step()
