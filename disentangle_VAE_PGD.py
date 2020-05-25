#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from art.attacks import FastGradientMethod,ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.utils import load_cifar10,load_mnist
from torch.autograd import Variable
from art.utils import load_mnist

# In[26]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)




# In[27]:


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)


# In[28]:


class cnn_model(nn.Module):
    def __init__(self,nc=1,number_classes=10):
        super(cnn_model,self).__init__()
        self.conv1 = nn.Conv2d(nc,6,5,padding=2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,number_classes)
    
    def forward(self,x):
        c1 = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        c2 = F.max_pool2d(F.relu(self.conv2(c1)),(2,2))
        f1 = self.flatten(c2)
        f2 = F.relu(self.fc1(f1))
        f3 = F.relu(self.fc2(f2))
        f4 = self.fc3(f3)
        return f4





class VAE(nn.Module):
    def __init__(self,z_dim=20,nc=1):
        super(VAE,self).__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc,32,5,padding=2)
        self.conv2 = nn.Conv2d(32,64,5,stride=1,padding=2)
        self.flatten = Flatten()
        self.encode_fc1 = nn.Linear(64*7*7,1024)
        self.encode_fc2 = nn.Linear(1024,2*z_dim)

        self.decode_fc1 = nn.Linear(z_dim,1024)
        self.decode_fc2 = nn.Linear(1024,784)
    
    def encode(self,x):
        h1 = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        h2 = F.max_pool2d(F.relu(self.conv2(h1)),(2,2))
        f1 = self.flatten(h2)
        f2 = F.relu(self.encode_fc1(f1))
        f3 = self.encode_fc2(f2)
        return f3 
    
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        return mu + eps*std
    
    def decode(self,x):
        decode_f1 = F.relu(self.decode_fc1(x))
        decode_f2 = self.decode_fc2(decode_f1)
        #conv_transposed1 = self.conv_Transposed1(decode_f2.view(-1,64,1,1))
        #conv_transposed2 = self.conv_Transposed2(conv_transposed1)
        #conv_transposed3 = self.conv_Transposed3(conv_transposed2)
        return decode_f2.view(-1,1,28,28)
    
    def forward(self,x):
        distribution = self.encode(x)
        mu = distribution[:,:self.z_dim]
        logvar = distribution[:,self.z_dim:]
        z = self.reparameterize(mu,logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar




class Ensemble(nn.Module):
    def __init__(self,modelA,modelB):
        super(Ensemble,self).__init__()
        self.modelA = modelA
        self.modelB = modelB
    
    def forward(self,x):
        x_recon,mu,logvar = self.modelA(x)
        y_pred = self.modelB(x_recon)
        return x,x_recon,mu,logvar,y_pred


# In[31]:


class VAE_encode_classifier(nn.Module):
    def __init__(self,z_dim=20,nc=1,number_classes=10):
        # encode layer
        super(VAE_encode_classifier,self).__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.maxpool1 = nn.MaxPool2d(2,stride=2)
        self.maxpool2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = nn.Conv2d(64,256,7,1)
        self.flatten = Flatten()
        self.encode_fc1 = nn.Linear(256,100)
        self.encode_fc2 = nn.Linear(100,2*z_dim)
        
        #classifier layer
        self.classifier_fc1 = nn.Linear(z_dim,100)
        self.classifier_fc2 = nn.Linear(100,number_classes)

    def encode(self,x):
        h1 = F.relu(self.conv1(x))
        m1 = self.maxpool1(h1)
        h2 = F.relu(self.conv2(m1))
        m2 = self.maxpool2(h2)
        h3 = F.relu(self.conv3(m2))
        f1 = self.flatten(h3)
        f2 = self.encode_fc1(f1)
        f3 = self.encode_fc2(f2)
        return f3
   
        
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
     
    def classifier(self,x):
        classifier_f1 = self.classifier_fc1(x)
        classifier_f2 = self.classifier_fc2(decode_f1)
        return classifier_f2
    
    def forward(self,x):
        distribution = self.encode(x)
        mu = distribution[:,:self.z_dim]
        logvar = distribution[:,self.z_dim:]
        z = self.reparameterize(mu,logvar)
        y_pred = self.classifier(z)
        return y_pred,mu,logvar


# In[51]:


def loss_function(recon_x,x,mu,logvar,beta=1):
    BCE = nn.CrossEntropyLoss()(recon_x,x)
    KLD = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE + beta*KLD



def beta_vae_loss_function(recon_x,x,mu,logvar,beta=1):
    BCE = F.binary_cross_entropy_with_logits(recon_x.view(-1,784),x.view(-1,784),reduction='sum')
    KLD = - 0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE + beta*KLD

def classification_loss_function(y,yp):
    return nn.CrossEntropyLoss()(yp,y)


def disentangle_beta_loss(model_output,y,gamma=1000,C_max=25,C_stop_iter=1e5):
    global global_iter
    gl = global_iter
    batch_size = x.size(0)
    BCE = F.binary_cross_entropy(recon_x,x.view(-1,784),reduction='sum').div(batch_size)
    #mse = F.mse_loss(recon_x,x.view(-1,784),reduction='sum')
    KLD = -0.5 * (1+logvar-mu.pow(2)-logvar.exp())
    total_KLD = KLD.sum(1).mean(0,True)
    #C = torch.clamp(C_max/C_stop_iter*global_iter,0,C_max.data[0])
    C = min(C_max/C_stop_iter*global_iter,C_max)
    beta_vae_loss = BCE + gamma * (KLD-C).abs()
    return beta_vae_loss


def custom_loss(model_output,y):
    global c
    C = c
    gamma = 1000
    x = model_output[0]
    recon_x = model_output[1]
    mu = model_output[2]
    logvar = model_output[3]
    yp = model_output[4]

    batch_size = x.size(0)
    #x,recon_x,mu,logvar,yp = model_output
    BCE = F.binary_cross_entropy_with_logits(recon_x.view(-1,784),x.view(-1,784),reduction='sum').div(batch_size)
    KLD = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    total_KLD = KLD.sum(1).mean(0,True)
    ENTROPY = F.cross_entropy(yp,y,reduction='sum').div(batch_size)
    return ENTROPY + BCE + gamma*(total_KLD-C).abs() 



# In[52]:


def fgsm(model, X, y, optimizer,epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    classifier = PyTorchClassifier(
    model=model_concetenate,
    loss = custom_loss,
    optimizer=optimizer,
    input_shape=(1,28,28),
    nb_classes=10,
    device_type='gpu'
    )
    attack = FastGradientMethod(classifier=classifier,eps=epsilon)
    x_adv = attack.generate(X.numpy(),y=y.numpy())
    return torch.Tensor(x_adv)

def pgd_linf(model, X, y, optimizer,epsilon=0.1):
    """ Construct pgd adversarial examples on the examples X"""
    classifier = PyTorchClassifier(
    model=model_concetenate,
    loss = custom_loss,
    optimizer=optimizer,
    input_shape=(1,28,28),
    nb_classes=10,
    device_type='gpu'
    )
    attack = ProjectedGradientDescent(classifier=classifier,eps=epsilon,max_iter=10)
        
    X_adv = attack.generate(X.numpy(),y.numpy())
    return torch.Tensor(X_adv)



# In[9]:


def save_checkpoint(model,filename,verbose=1):
    torch.save(model.state_dict(), filename)
    if verbose == 1:
        print("===> saved checkpoint '{}' ".format(filename))


# In[53]:


def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, attack, beta,global_iter,opt=None):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    batch_idx = 0
    for X,y in loader:
        #X,y = X.to(device), y.to(device)
        if attack == None:
            X_adv = X
        else:
            X_adv = attack(model,X,y,opt,epsilon=0.1)
        
        X_adv = X_adv.to(device)
        y = y.to(device)
        model_output = model(X_adv)
        yp = model_output[-1]
        loss = custom_loss(model_output,y)
        #loss = disentangle_beta_loss(yp,y,mu,logvar,beta,global_iter=global_iter)
        #loss = nn.CrossEntropyLoss()(yp,y)
        #loss = loss_function(yp,y,mu,logvar,beta)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# In[54]:

C_max = 25
C_max = Variable(torch.FloatTensor([C_max]).cuda())
C_stop_iter = 1e5
global_iter = 0

beta_list = np.linspace(3,3,1).tolist()
b = 0
epochs = 30
file_path='./change_beta_FGSM_result/beta_'
attack = fgsm
c = torch.clamp(C_max/C_stop_iter*global_iter,0,C_max.data[0])

for beta in beta_list:
    b = beta
    train_error_path = file_path+str(beta)+'_training_err.txt'
    train_loss_path = file_path+str(beta)+'_training_loss.txt'
    test_error_path = file_path+str(beta)+'_test_err.txt'
    test_loss_path = file_path+str(beta)+'_test_loss.txt'
    train_clean_data_error = file_path+str(beta)+'_clean_train_err.txt'
    test_clean_data_error = file_path+str(beta)+'_clean_test_err.txt'

    
    
    modelA = VAE(z_dim=20,nc=1).to(device)
    modelB = cnn_model(nc=1,number_classes=10).to(device)
    model_concetenate = Ensemble(modelA,modelB).to(device)
    opt = optim.Adam(model_concetenate.parameters(),lr=5e-4)
    for epoch in range(epochs):
        total_loss, total_err = 0.,0.
        batch_idx = 0

        for X,y in train_loader:
            
            global_iter = batch_idx + epoch*600
            X_adv = attack(model_concetenate,X,y,opt,epsilon=0.1)
            X = X.to(device)
            X_adv = X_adv.to(device)
            y = y.to(device)
            model_output = model_concetenate(X_adv)
            yp = model_output[-1]
            c = torch.clamp(C_max/C_stop_iter*global_iter,0,C_max.data[0])
            loss = custom_loss(model_output,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_err = (yp.max(dim=1)[1] != y).sum().item()

            if batch_idx % 10 == 0:
                f1 = open(train_error_path,'a')
                f2 = open(train_loss_path,'a')
                f5 = open(train_clean_data_error,'a')
                
                
                f1.write('\n'+str(batch_err))
                f2.write('\n'+str(loss.item()))
                clean_data_model_output = model_concetenate(X)
                clean_data_yp = clean_data_model_output[-1]
                clean_batch_err = (clean_data_yp.max(dim=1)[1] != y).sum().item()
                f5.write('\n'+str(clean_batch_err))
                
                f1.close()
                f2.close()
                f5.close()
            batch_idx = batch_idx + 1


        print('In Epoch %d ==========================>'%(epoch))
        test_total_err,test_total_loss = epoch_adversarial(test_loader,model_concetenate,attack,beta,global_iter)
        clean_test_total_err,clean_test_total_loss = epoch_adversarial(test_loader,model_concetenate,None,beta,global_iter)
        print("test error on adverasrial attack is "+str(test_total_err))
        print('clean data test error is ' +str(clean_test_total_err))
        
        f3 = open(test_error_path,'a')
        f4 = open(test_loss_path,'a')
        f6 = open(test_clean_data_error,'a')
    
        f3.write('\n'+str(test_total_err))
        f4.write('\n'+str(test_total_loss))
        f6.write('\n'+str(clean_test_total_err))
        
        f3.close()
        f4.close()
        f6.close()
    
    del modelA, modelB, model_concetenate







