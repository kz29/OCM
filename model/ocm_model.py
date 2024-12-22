import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the OCMmodule class
class OCMmodule(nn.Module):
    
    def __init__(self, input_size, output_size, FEATURE_EMBED):
        super(OCMmodule, self).__init__()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(7 + FEATURE_EMBED, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 7)
       
    def forward(self, X, globEmbed):
        # Concatenate input features with global embedding
        X = torch.cat([X, globEmbed], 1)
        
        # Pass through fully connected layers with SELU activations
        X = F.selu(self.fc1(X))
        X = F.selu(self.fc2(X))       
        X = F.selu(self.fc3(X))
        X = self.fc4(X)

        return X

# Define the OCMmodel class
class OCMmodel(nn.Module):
    
    def __init__(self, input_channel, FEATURE_EMBED):
        super(OCMmodel, self).__init__()

        # Initialize embed dimension and other parameters
        self.embed_dim = FEATURE_EMBED
        self.layers = nn.ModuleList()
        output_channel = input_channel
        
        # Define embedding layers
        self.catalystAtomEmbedding = nn.Linear(self.embed_dim, CATALIST_ATOMS, bias=False)
        self.temperatureEmbedding = nn.Linear(1, self.embed_dim)
        self.supportEmbedding = nn.Linear(self.embed_dim, SUP_ID, bias=False)
        self.globFC = nn.Linear(self.embed_dim * 3, self.embed_dim)

        # Define the OCMmodule
        self.module = OCMmodule(input_channel + self.embed_dim, output_channel, FEATURE_EMBED).to(device)
        self.layers.append(self.module)
        
        # Initialize signDelta (used for scaling)
        self.signDelta = None

    def forward(self, inflow, xCatalistAtom, xCatalistMol, xTEMP, xSupport, time):
        # Forward pass through the model
        
        # Input embedding for catalyst, temperature, and support
        x = inflow
        
        # Embedding for catalyst atoms and molecules
        catalystEmbedding = (self.catalystAtomEmbedding.weight[xCatalistAtom.reshape([-1]), :] * xCatalistMol.reshape([-1, 1])).reshape(-1, 3, self.embed_dim)
        catalystEmbedding = torch.sum(catalystEmbedding, 1)
        
        # Embedding for temperature
        tempEmbed = self.temperatureEmbedding(xTEMP)  
        
        # Embedding for support
        supportEmbed = self.supportEmbedding.weight[xSupport]
    
        # Concatenate all embeddings to form global embedding
        globEmb = torch.cat([catalystEmbedding, tempEmbed, supportEmbed], 1)
        globEmb = self.globFC(globEmb)

        # Initialize history of x
        self.xHist = [x]
        
        if self.signDelta is None:
            # Define signDelta used for scaling
            self.signDelta = torch.tensor([[0.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]]).to(device)
        
        # Iterate over the specified number of time steps
        for i in range(time):
            # Compute the delta (change) using the OCMmodule
            delta = self.module(x, globEmb)
            
            # Apply signDelta for scaling
            delta = self.signDelta * torch.abs(delta)
            
            # Update x with the computed delta
            x = x + delta
            self.xHist.append(x)   
        
        return x
