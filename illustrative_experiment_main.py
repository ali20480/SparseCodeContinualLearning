"""
Illustrative Experiment on the Usefulness of Sparse Coding Networks for Continual Learning
"""


import matplotlib.pyplot as plt

plt.close('all')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.utils import shuffle


digits = load_digits()
DATA = digits.data
Y = digits.target
DATA, Y = shuffle(DATA, Y, random_state = 42)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class Supervised_model(nn.Module):
    def __init__(self, input_size, code_size, n_class, criterion, criterion_class, batch_size, lambda_reg = 0.9, N_iter = 50, eta_c = 1e-3, lambda_c = 0.01, weight_init = "default"):
        super(Supervised_model, self).__init__()
        self.criterion = criterion
        self.criterion_class = criterion_class
        self.batch_size = batch_size

        self.input_size = input_size
        self.code_size = code_size
        self.n_class = n_class

        self.Dictionary = nn.Linear(self.input_size, self.code_size, bias = False)
        self.relu=nn.ReLU()
        
        self.lambda_reg = lambda_reg
        self.N_iter = N_iter
        self.eta_c = eta_c
        self.lambda_c = lambda_c
        
        if weight_init == "orthogonal":
            nn.init.orthogonal_(self.Dictionary.weight) 
        elif weight_init == "xavier":
            nn.init.xavier_uniform_(self.Dictionary.weight)
        elif weight_init == "default":
            pass
        


    def forward(self, input, label):
        
        # Feed in the whole sequence
        batch_size, input_dim = input.shape

        #sample randomly the input - 
        input_x = input[:, :].float()


        code = self.Dictionary(input_x) 
        Npopulation = code.shape[1] // self.n_class
        code_ = code.reshape(code.shape[0], self.n_class, Npopulation)
        out_ = code_.sum(axis = 2)
        out_class = F.log_softmax(out_,dim=1)
        
        pred_ = out_class.argmax(axis=1)
        
        L1 = torch.tensor([0])
        
        L2 =  self.criterion_class(out_class.float().to(device), label.long().to(device))
        
        loss_ = L2
        
        return loss_, pred_, out_, L1, L2
    
    

class Supervised_LISTA(nn.Module):
    def __init__(self, input_size, code_size, n_class, criterion, criterion_class, batch_size, lambda_reg = 0.9, N_iter = 50, eta_c = 1e-3, lambda_c = 0.01, weight_init = "default"):
        super(Supervised_LISTA, self).__init__()
        self.criterion = criterion
        self.criterion_class = criterion_class
        self.batch_size = batch_size

        self.input_size = input_size
        self.code_size = code_size
        self.n_class = n_class

        self.Dictionary = nn.Linear(self.code_size, self.input_size, bias = False)
        self.relu=nn.ReLU()
        
        self.lambda_reg = lambda_reg
        self.N_iter = N_iter
        self.eta_c = eta_c
        self.lambda_c = lambda_c
        
        if weight_init == "orthogonal":
            nn.init.orthogonal_(self.Dictionary.weight)
        elif weight_init == "xavier":
            nn.init.xavier_uniform_(self.Dictionary.weight)
        elif weight_init == "default":
            pass
        


    def forward(self, input, label):
        
        # Feed in the whole sequence
        batch_size, input_dim = input.shape

        #sample randomly the input - 
        input_x = input[:, :].float()

        code = torch.zeros(batch_size, self.code_size)
        Phi = self.Dictionary.weight.t()
        Phi_t = Phi.t()
        for i in range(self.N_iter):
            code = code - self.eta_c * torch.mm(torch.mm(code, Phi) - input_x, Phi_t)
            code = self.relu((code - self.eta_c * self.lambda_c)) - self.relu(-(code - self.eta_c * self.lambda_c)) 
            
        
        reprojection = self.Dictionary(code) 
        Npopulation = code.shape[1] // self.n_class
        code_ = code.reshape(code.shape[0], self.n_class, Npopulation)
        out_ = code_.sum(axis = 2)
        out_class = F.log_softmax(out_,dim=1)
        
        pred_ = out_class.argmax(axis=1)
        
        L1 = self.criterion(reprojection.float().to(device), input_x.float().to(device))
        
        L2 =  self.criterion_class(out_class.float().to(device), label.long().to(device))
        
        loss_ = self.lambda_reg * L1 + (1 - self.lambda_reg) * L2
        
        return loss_, pred_, out_, L1, L2



"""
1) Continual learning on the Sparse Coding network
"""    
    

code_size = 2000
n_class = 10 
input_size = 64 

    
criterion = nn.MSELoss()
criterion_2 = nn.NLLLoss()

Nepochs = 2
batch_size = 8

model = Supervised_LISTA(input_size, code_size, n_class, criterion, criterion_2, batch_size) 
learning_rate = 1e-3

optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(model, train_loader, valid_loader, optimizer_1, epochs, batch_size):
    
    train_loss_array = []
    valid_loss_array = []
    reproj_loss_arr = []
    class_loss_arr = []
    Acc = []
    for e in range(epochs):
        train_loss_sum = 0
        valid_loss_sum = 0
        optimizer = optimizer_1
            
        for i, (images, label) in enumerate(train_loader):
            optimizer.zero_grad()        
            train_loss, out, code, l1, l2 = model(images, label)
            
            train_loss_sum += train_loss.data.cpu().numpy() 
            
            train_loss.sum().backward(retain_graph=True)
            optimizer.step()
            reproj_loss_arr.append(l1.cpu().detach().numpy())
            class_loss_arr.append(l2.cpu().detach().numpy())

            
        
        train_loss_now = train_loss_sum.item() / (i+1) 
        train_loss_array.append(train_loss_now)
        

        print("Epoch: " + str(e) + " Loss: " + str(train_loss_now))
        
        acc = 0
        incr = 0
        for i, (images, label) in enumerate(valid_loader):      
            valid_loss, out, code, l1, l2 = model(images, label)
            
            valid_loss_sum += valid_loss.data.cpu().numpy()
            inferred = out.cpu().detach().numpy()
            acc = acc + np.mean((inferred == label.cpu().detach().numpy()).astype(int))
            incr += 1
            
        total_acc = acc / incr
        valid_loss_now = valid_loss_sum.item() / (i+1) 
        valid_loss_array.append(valid_loss_now)
        

        print("Validation loss: " + str(valid_loss_now) + " Acc: " + str(total_acc))
        Acc.append(total_acc)
        
    return np.array(train_loss_array), np.array(valid_loss_array), np.array(reproj_loss_arr), np.array(class_loss_arr), np.array(Acc)




Ntrain = int(DATA.shape[0] * 0.8)
X = DATA[:Ntrain]
mu_t = np.mean(X, axis = 0)
std_t = np.std(X, axis = 0)
std_t[std_t == 0] = 1
y = Y[:Ntrain]
idx = np.argsort(y)
y = y[idx]
X = X[idx]
Xv = DATA[Ntrain:]
yv = Y[Ntrain:]
idx = np.argsort(yv)
Xv = Xv[idx]
yv = yv[idx]

Train_Loss = 0
Valid_Loss = 0
Reproj_Loss = 0
Class_Loss = 0
Acc_arr = 0
Final_Accuracies = []

for i in range(n_class):
    print("------------CLASS: " + str(i))
    idx_sel = np.argwhere(y == i)[:,0]
    X_t = X[idx_sel]
    y_t = y[idx_sel]
    
    idx_sel = np.argwhere(yv <= i)[:,0]
    X_tv = Xv[idx_sel]
    y_tv = yv[idx_sel]
    
    train_data = TensorDataset(torch.from_numpy(X_t), torch.from_numpy(y_t))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
    
    valid_data = TensorDataset(torch.from_numpy(X_tv), torch.from_numpy(y_tv))
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)
    
    train_loss_arr, valid_loss_arr, reproj_loss_arr, class_loss_arr, acc_arr = train(model, train_loader, valid_loader, optimizer_1, Nepochs, batch_size) 
    
    Final_Accuracies.append(acc_arr[-1])
    
    if i == 0:
        Train_Loss = train_loss_arr
        Valid_Loss = valid_loss_arr
        Reproj_Loss = reproj_loss_arr
        Class_Loss = class_loss_arr
        Acc_arr = acc_arr
    else:
        Train_Loss = np.concatenate((Train_Loss, train_loss_arr))
        Valid_Loss = np.concatenate((Valid_Loss, valid_loss_arr))
        Reproj_Loss = np.concatenate((Reproj_Loss, reproj_loss_arr))
        Class_Loss = np.concatenate((Class_Loss, class_loss_arr))
        Acc_arr = np.concatenate((Acc_arr, acc_arr))
        
Final_Accuracies_SCN = np.array(Final_Accuracies) * 1 

"""
2) Continual Learning on the baseline model
"""

criterion = nn.MSELoss()
criterion_2 = nn.NLLLoss()

Nepochs = 2
batch_size = 8 

model = Supervised_model(input_size, code_size, n_class, criterion, criterion_2, batch_size)
learning_rate = 0.1 * 1e-3

optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)


digits = load_digits()
DATA = digits.data
Y = digits.target
DATA, Y = shuffle(DATA, Y, random_state = 42)


Ntrain = int(DATA.shape[0] * 0.8)
X = DATA[:Ntrain]
mu_t = np.mean(X, axis = 0)
std_t = np.std(X, axis = 0)
std_t[std_t == 0] = 1
y = Y[:Ntrain]
idx = np.argsort(y)
y = y[idx]
X = X[idx]
Xv = DATA[Ntrain:]
yv = Y[Ntrain:]
idx = np.argsort(yv)
Xv = Xv[idx]
yv = yv[idx]

Train_Loss = 0
Valid_Loss = 0
Reproj_Loss = 0
Class_Loss = 0
Acc_arr = 0
Final_Accuracies = []

for i in range(n_class):
    print("------------CLASS: " + str(i))
    idx_sel = np.argwhere(y == i)[:,0]
    X_t = X[idx_sel]
    y_t = y[idx_sel]
    
    idx_sel = np.argwhere(yv <= i)[:,0]
    X_tv = Xv[idx_sel]
    y_tv = yv[idx_sel]
    
    train_data = TensorDataset(torch.from_numpy(X_t), torch.from_numpy(y_t))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
    
    valid_data = TensorDataset(torch.from_numpy(X_tv), torch.from_numpy(y_tv))
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)
    
    train_loss_arr, valid_loss_arr, reproj_loss_arr, class_loss_arr, acc_arr = train(model, train_loader, valid_loader, optimizer_1, Nepochs, batch_size) 
    
    Final_Accuracies.append(acc_arr[-1])
    
    if i == 0:
        Train_Loss = train_loss_arr
        Valid_Loss = valid_loss_arr
        Reproj_Loss = reproj_loss_arr
        Class_Loss = class_loss_arr
        Acc_arr = acc_arr
    else:
        Train_Loss = np.concatenate((Train_Loss, train_loss_arr))
        Valid_Loss = np.concatenate((Valid_Loss, valid_loss_arr))
        Reproj_Loss = np.concatenate((Reproj_Loss, reproj_loss_arr))
        Class_Loss = np.concatenate((Class_Loss, class_loss_arr))
        Acc_arr = np.concatenate((Acc_arr, acc_arr))
        
Final_Accuracies_Baseline = np.array(Final_Accuracies) * 1

class_idx = np.linspace(0, 9, 10)
plt.figure(1)
plt.plot(Final_Accuracies_SCN, '.-', label = "Sparse Coding Network")
plt.plot(Final_Accuracies_Baseline, '.-', label = "Baseline Model (Without Sparse Coding behavior)")
plt.xticks(class_idx, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.ylim([0,1.2])
plt.grid("on")
plt.legend()
plt.xlabel("Task Index", fontsize = 15)
plt.ylabel("Accuracy", fontsize = 15)

