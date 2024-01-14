# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2024-01-13 07:22:49
# @Last Modified by:   Your name
# @Last Modified time: 2024-01-14 02:09:59
import torch
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, TensorDataset
from coefficients_interpolation import k_1_func, k_2_func, r_func
from scipy.misc import derivative
from scipy.interpolate import RegularGridInterpolator

#--- Fixing the Random Seed ---

np.random.seed(1234)
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1234)

# --- CUDA support ---

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#--- the deep neural network ---
    
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        #--- parameters ----

        self.depth = len(layers) - 1
        
        #--- set up layer order dict ---

        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        #--- deploy layers ---

        self.layers = torch.nn.Sequential(layerDict)
        
        #--- Initialize using Xavier ---

        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
        
    def forward(self, x):
        out = self.layers(x)
        return out
    
#--- the physics-guided neural network ---
    
class PhysicsInformedNN():
    def __init__(self, X_t0_train, X_x0_train, X_x1_train, X_y0_train, X_y1_train, X_f_train, layers):
        
        self.X_t0_train = X_t0_train
        self.X_x0_train = X_x0_train
        self.X_x1_train = X_x1_train
        self.X_y0_train = X_y0_train
        self.X_y1_train = X_y1_train
        self.X_f_train = X_f_train

        
        #--- unzip data ---
        
        self.time_step = 14
        self.update_train_data(self.time_step)
        
        self.layers = layers
        
        #--- deep neural networks ---

        self.dnn = DNN(layers).to(device)
        
        #--- optimizers: using the same settings ---

        self.optimizer = torch.optim.LBFGS(
           self.dnn.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=40,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(),
            lr=5e-6
        )

        self.iter = 0
        self.time_step = 14
        
    def update_train_data(self, t):
        
        # X_t0_train = self.X_t0_train[(self.X_t0_train[:, 2] >= t-10) & (self.X_t0_train[:, 2] < t) ]
        X_x0_train = self.X_x0_train[(self.X_x0_train[:, 2] >= t-self.time_step) & (self.X_x0_train[:, 2] < t) ]
        X_x1_train = self.X_x1_train[(self.X_x1_train[:, 2] >= t-self.time_step) & (self.X_x1_train[:, 2] < t) ]
        X_y0_train = self.X_y0_train[(self.X_y0_train[:, 2] >= t-self.time_step) & (self.X_y0_train[:, 2] < t) ]
        X_y1_train = self.X_y1_train[(self.X_y1_train[:, 2] >= t-self.time_step) & (self.X_y1_train[:, 2] < t) ]
        X_f_train = self.X_f_train[(self.X_f_train[:, 2] >= t-self.time_step) & (self.X_f_train[:, 2] < t) ]
        
        self.x_t0, self.y_t0, self.t_t0, self.k1_t0, self.k2_t0, self.r_t0 = self.unzip(X_t0_train)
        self.x_x0, self.y_x0, self.t_x0, self.k1_x0, self.k2_x0, self.r_x0 = self.unzip(X_x0_train)
        self.x_x1, self.y_x1, self.t_x1, self.k1_x1, self.k2_x1, self.r_x1 = self.unzip(X_x1_train)
        self.x_y0, self.y_y0, self.t_y0, self.k1_y0, self.k2_y0, self.r_y0 = self.unzip(X_y0_train)
        self.x_y1, self.y_y1, self.t_y1, self.k1_y1, self.k2_y1, self.r_y1 = self.unzip(X_y1_train)
        self.x_f, self.y_f, self.t_f, self.k1_f, self.k2_f, self.r_f = self.unzip(X_f_train)
        self.k1_x_f = torch.tensor(X_f_train[:, 6:7]).float().to(device)
        self.k1_y_f = torch.tensor(X_f_train[:, 7:8]).float().to(device)
        self.k2_x_f = torch.tensor(X_f_train[:, 8:9]).float().to(device)
        self.k2_y_f = torch.tensor(X_f_train[:, 9:10]).float().to(device)
        
        
    def unzip(self, train_data):
        x = torch.tensor(train_data[:, 0:1], requires_grad=True).float().to(device)
        y = torch.tensor(train_data[:, 1:2], requires_grad=True).float().to(device)
        t = torch.tensor(train_data[:, 2:3], requires_grad=True).float().to(device)
        k1 = torch.tensor(train_data[:, 3:4]).float().to(device)
        k2 = torch.tensor(train_data[:, 4:5]).float().to(device)
        r = torch.tensor(train_data[:, 5:6]).float().to(device)
        return x, y, t, k1, k2, r
    
    #--- dnn for p1 ---    
        
    def net_p1(self, x, y, t):  
        p1 = self.dnn(torch.cat([x, y, t], dim=1))[:, 0:1]

        return p1
    
    #--- dnn for p2 ---

    def net_p2(self, x, y, t):
        p2 = self.dnn(torch.cat([x, y, t], dim=1))[:, 1:2]

        return p2
    
        
    
    def net_f(self, x, y, t, k1, k2, r, k1_x, k1_y, k2_x, k2_y):
        """ 
        The pytorch autograd version of calculating residual

        """
        p1 = self.net_p1(x, y, t)
        p2 = self.net_p2(x, y, t)
        
        p1_x = torch.autograd.grad(
            p1, x, 
            grad_outputs=torch.ones_like(p1),
            retain_graph=True,
            create_graph=True
        )[0]
        p1_t = torch.autograd.grad(
            p1, t, 
            grad_outputs=torch.ones_like(p1),
            retain_graph=True,
            create_graph=True
        )[0]
        p1_xx = torch.autograd.grad(
            p1_x, x, 
            grad_outputs=torch.ones_like(p1_x),
            retain_graph=True,
            create_graph=True
        )[0]
        p1_y= torch.autograd.grad(
            p1, y, 
            grad_outputs=torch.ones_like(p1),
            retain_graph=True,
            create_graph=True
        )[0]
        p1_yy = torch.autograd.grad(
            p1_y, y, 
            grad_outputs=torch.ones_like(p1_y),
            retain_graph=True,
            create_graph=True
        )[0]
        p2_x = torch.autograd.grad(
            p2, x, 
            grad_outputs=torch.ones_like(p2),
            retain_graph=True,
            create_graph=True
        )[0]
        p2_t = torch.autograd.grad(
            p2, t, 
            grad_outputs=torch.ones_like(p2),
            retain_graph=True,
            create_graph=True
        )[0]
        p2_xx = torch.autograd.grad(
            p2_x, x, 
            grad_outputs=torch.ones_like(p2_x),
            retain_graph=True,
            create_graph=True
        )[0]
        p2_y= torch.autograd.grad(
            p2, y, 
            grad_outputs=torch.ones_like(p2),
            retain_graph=True,
            create_graph=True
        )[0]
        p2_yy = torch.autograd.grad(
            p2_y, y, 
            grad_outputs=torch.ones_like(p2_y),
            retain_graph=True,
            create_graph=True
        )[0]        

        f_1 = p1_t - (k1_x * p1_x + k1 * p1_xx + k1_y * p1_y + k1 * p1_yy) + r * (p1 - p2)
        f_2 = p2_t - (k2_x * p2_x + k2 * p2_xx + k2_y * p2_y + k2 * p2_yy) - r * (p1 - p2)
        return f_1**2  + f_2**2 
    
    def net_bound(self, x, y, t, k1_y0, k2_y0):

        p1 = self.net_p1(x, y, t)
        p2 = self.net_p2(x, y, t)
        p1_y= torch.autograd.grad(
            p1, y, 
            grad_outputs=torch.ones_like(p1),
            retain_graph=True,
            create_graph=True
        )[0]
        p2_y= torch.autograd.grad(
            p2, y, 
            grad_outputs=torch.ones_like(p2),
            retain_graph=True,
            create_graph=True
        )[0]
        return (k1_y0 / 1e-4 * p1_y) ** 2 + (k2_y0 / 1e-7  * p2_y) ** 2 



    def loss_func(self):
        self.optimizer.zero_grad()
        
        p1_pred_t0 = self.net_p1(self.x_t0, self.y_t0, self.t_t0)
        p2_pred_t0 = self.net_p2(self.x_t0, self.y_t0, self.t_t0)
        p1_pred_x0 = self.net_p1(self.x_x0, self.y_x0, self.t_x0)
        p2_pred_x0 = self.net_p2(self.x_x0, self.y_x0, self.t_x0)
        p1_pred_x1 = self.net_p1(self.x_x1, self.y_x1, self.t_x1)
        p2_pred_x1 = self.net_p2(self.x_x1, self.y_x1, self.t_x1)
        f_pred = self.net_f(self.x_f, self.y_f, self.t_f, self.k1_f, self.k2_f, self.r_f, self.k1_x_f, self.k1_y_f, self.k2_x_f, self.k2_y_f)
        loss_p1_t0 = torch.mean((1 - p1_pred_t0)** 2)
        loss_p1_x0 = torch.mean((2 * 1 - p1_pred_x0)** 2)
        loss_p1_x1 = torch.mean((1 - p1_pred_x1)** 2)
        loss_p2_t0 = torch.mean((1 - p2_pred_t0)** 2)
        loss_p2_x0 = torch.mean((2 * 1 - p2_pred_x0)** 2)
        loss_p2_x1 = torch.mean((1 - p2_pred_x1)** 2)
        loss_y0 = torch.mean(self.net_bound(self.x_y0, self.y_y0, self.t_y0, self.k1_y0, self.k2_y0))
        loss_y1 = torch.mean(self.net_bound(self.x_y1, self.y_y1, self.t_y1, self.k1_y1, self.k2_y1))
        loss_f = torch.mean(f_pred)
        loss = loss_f + loss_p1_t0 + loss_p1_x0*1.5 + loss_p1_x1*1.5 + loss_p2_t0 + loss_p2_x0*1.5 + loss_p2_x1*1.5 + loss_y0 + loss_y1
        
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d,   Loss: %.5e, \n \n Loss_f: %.5e,    Loss_p1_t0: %.5e,   Loss_p1_x0: %.5e, \n \n Loss_p1_x1: %.5e,   Loss_p2_t0: %.5e, \n \n Loss_p2_x0: %.5e,   Loss_p2_x1: %.5e, \n \n Loss_y0: %.5e,   Loss_y1: %.5e \n \n' % (self.iter, loss.item(), loss_f.item(), loss_p1_t0.item(), loss_p1_x0.item(), loss_p1_x1.item(), loss_p2_t0.item(), loss_p2_x0.item(), loss_p2_x1.item(), loss_y0.item(), loss_y1.item())
            )
        return loss
    
    def train(self):

        self.dnn.train()
                
        #--- Backward and optimize ---

        self.optimizer.zero_grad()
        self.optimizer.step(self.loss_func)
        
    def train_Adam(self, epoch):
        self.dnn.train()
        for i in range(epoch) :
            self.optimizer_Adam.zero_grad()
            self.optimizer_Adam.step(self.loss_func)
        print("Adam done--------------------------------------")
            

    def train_epoch(self):
        num_epoch = 2
        self.dnn.train()
        for epoch in range(num_epoch):
            for t in np.linspace(self.time_step, 7000, int(7000/self.time_step), dtype=int):
                self.update_train_data(t)
                self.train_Adam(100)
                self.train()


    def predict(self, test_data):
        
        x, y, t, k1, k2, r = self.unzip(test_data)
        k1_x = torch.tensor(test_data[:, 6:7]).float().to(device)
        k1_y = torch.tensor(test_data[:, 7:8]).float().to(device)
        k2_x = torch.tensor(test_data[:, 8:9]).float().to(device)
        k2_y = torch.tensor(test_data[:, 9:10]).float().to(device)

        self.dnn.eval()

        p1 = self.net_p1(x, y, t) * 1e+7
        f = self.net_f(x, y, t, k1, k2, r, k1_x, k1_y, k2_x, k2_y) * 1e+7
        p2 = self.net_p2(x, y, t) * 1e+7

        p1 = p1.detach().cpu().numpy()
        p2 = p2.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

        return p1, p2, f
    

def load_ref_solution(field, t):
    L_x, L_y = 1.0, 1.0
    n_x, n_y = 100, 100

    x = np.linspace(0, L_x, n_x + 1)
    y = np.linspace(0, L_y, n_y + 1)

    field_vector = np.loadtxt(f"./solutions/{field}/{field}_{t}.txt", dtype=float)
    field_matrix = np.reshape(field_vector, (n_x + 1, n_y + 1)).T
    field_func = RegularGridInterpolator((x, y), field_matrix)

    return field_func

#--- the number of train data ---
    
N_b_l = 100000
N_b_r = 100000
N_b_b = 100000
N_b_t = 100000
N_c = 1000
N_f = 300000

#--- Generate training data ---

def generate_data(lb, rb, N):
    train_data = lb + (rb - lb) * lhs(3, N)
    k_1 = k_1_func((train_data[:, 0:1], train_data[:, 1:2])).reshape(-1, 1)
    k_2 = k_2_func((train_data[:, 0:1], train_data[:, 1:2])).reshape(-1, 1)
    r = r_func((train_data[:, 0:1], train_data[:, 1:2])).reshape(-1, 1)
    k_1_x = derivative(lambda x: k_1_func((x, train_data[:, 1:2])), x0=train_data[:, 0:1], dx=1e-5) 
    k_1_y = derivative(lambda y: k_1_func((train_data[:, 0:1], y)), x0=train_data[:, 1:2], dx=1e-5)
    k_2_x = derivative(lambda x: k_2_func((x, train_data[:, 1:2])), x0=train_data[:, 0:1], dx=1e-5)
    k_2_y = derivative(lambda y: k_2_func((train_data[:, 0:1], y)), x0=train_data[:, 1:2], dx=1e-5)
    train_data = np.hstack([train_data, k_1, k_2, r, k_1_x, k_1_y, k_2_x, k_2_y])

    return train_data


t_max = 7000
X_t0_train = generate_data(np.array([0, 0, 0]), np.array([1, 1, 0]), N_c)
X_x0_train = generate_data(np.array([0, 0, 0]), np.array([0, 1, t_max]), N_b_l)
X_x1_train = generate_data(np.array([1, 0, 0]), np.array([1, 1, t_max]), N_b_r)
X_y0_train = generate_data(np.array([0, 0, 0]), np.array([1, 0, t_max]), N_b_b)
X_y1_train = generate_data(np.array([0, 1, 0]), np.array([1, 1, t_max]), N_b_t)
X_f_train = generate_data(np.array([0, 0, 0]), np.array([1, 1, t_max]), N_f)
X_f_train = np.vstack([X_f_train, X_t0_train, X_x0_train, X_x1_train, X_y0_train, X_y1_train])

#--- Delete rows containing NaN values ---

nan_rows = np.any(np.isnan(X_f_train), axis=1)
X_f_train = X_f_train[~nan_rows]

#--- Network structure ---

layers = [3] + [250]*2 + [2]

#--- Construct model ---

model = PhysicsInformedNN(X_t0_train, X_x0_train, X_x1_train, X_y0_train, X_y1_train, X_f_train, layers)

#--- training model ---

# model.train_Adam(1000)
# model.train()
model.train_epoch()
# model = torch.load('./model/test_model.pth')

torch.save(model, './model/test_model.pth') 

#--- Calculate relative error ---

def relative_error(model, t, N_test):
    
    test_data = generate_data(np.array([0, 0, t]), np.array([1, 1, t]), N_test)
    p1, p2, f = model.predict(test_data)
    
    #--- Reference solution ---
    
    p1_func = load_ref_solution("p_1", t)
    p2_func = load_ref_solution("p_2", t)
    p1_ref = p1_func((test_data[:, 0:1], test_data[:, 1:2])).reshape(-1, 1)
    p2_ref = p2_func((test_data[:, 0:1], test_data[:, 1:2])).reshape(-1, 1)
    
    print(f"t is {t}")
    print(np.linalg.norm(p1 - p1_ref) / np.linalg.norm(p1_ref))
    print(np.linalg.norm(p2 - p2_ref) / np.linalg.norm(p2_ref))
    print("----------------------------------------------") 



for t in np.linspace(140, 7000, 50, dtype=int) :
    
    relative_error(model, t, N_test=10000)