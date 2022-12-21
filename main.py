import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


from pathlib import Path
import argparse

## reading data
print("reading data")

parser = argparse.ArgumentParser(description='portfolio contruction with neural networks')
parser.add_argument('--model', metavar='M', type=str, default='dnn',
                    help='type of model: feedforward, lstm')


args = parser.parse_args()


## define network
class dnn(nn.Module):

    def __init__(self):
        super(dnn, self).__init__()
        self.w = torch.nn.Linear(30, 64)
        self.w1 = torch.nn.Linear(64, 64)
        self.act = F.leaky_relu
        self.w2 = torch.nn.Linear(64, 1)
    def forward(self, x):
        x = self.act(self.w(x))
        x = self.act(self.w1(x))
        x = self.w2(x)
        return torch.sigmoid(x)

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.RNN(input_size=1, hidden_size=50)
        self.classifier= nn.Linear(50, 1)
        
    def forward(self, x):
        _, (x) =self.lstm(x)
        x = self.classifier(x.squeeze(0))
        return torch.sigmoid(x)


    
def train_dnn(X,Y, criterion, f, opt, sigma, lbd, minibatch):
  train_loss = 0
  for itr in range(1):
    x, y = minibatch(X, Y, sigma)
    y_pred = f(x)
    loss = criterion(y_pred, y, lbd, x, sigma)  
    opt.zero_grad()  
    loss.backward()
    opt.step()
  return 0 


def train_lstm(X,Y, criterion, f, opt, sigma, lbd, minibatch):
  train_loss = 0
  for itr in range(1):
    x, y = minibatch(X, Y, sigma)
    y_pred = f(x.transpose(1, 0).unsqueeze(2))

    loss = criterion(y_pred, y, lbd, x, sigma)  
    opt.zero_grad()  
    loss.backward()
    opt.step()
  return 0 


if args.model == 'lstm':
    Net = LSTM
    train = train_lstm
elif args.model == 'dnn':
    Net = dnn
    train = train_dnn
else:
    print('type of net unsupported')


data = pd.read_csv('../data/sp500.csv')
symbol_list = list(np.load('../data/symbol_list.npy'))
pruned_data = data[data['symbol'].isin(symbol_list)]
X = torch.tensor(list(pruned_data['open'])).view(-1, len(symbol_list))
RX = torch.log(X[1:] / X[:-1]) 
symbol_list = np.sort(symbol_list)



## minibatch sampling

S = 64
sigma = 0.5
def minibatch(datax, datay, sigma=0):
  L = len(datax)
  idx = torch.randint(L-1, (S,)).cuda()
  x, y, r = datax[idx], datay[idx], RTNtrain[idx]
  if sigma>0:
    return x + (r.abs() ** 0.5) * (x) * torch.empty_like(x).normal_(0, sigma), y
  return x, y



def sharpe(X):
  X = np.array(X)
  ri = (X[1:] / X[:-1]) -1
  return ((ri).mean() - 0.00004)/(np.std(ri) + 1e-5) * np.sqrt(252)

def wealth_gain(x, y):
    return x * y  + (1 - x) * 0.00004

def criterion(x, y, lbd, X, sigma):
  r = ((X[:, 10:] - X[:, 9:-1])/(X[:, 9:-1]))
  return -(wealth_gain(x,y) -  (sigma ** 2) * lbd * (r.abs().mean(dim=1).unsqueeze(1)) *  (x **2)).mean()



## logging purpose
m = 1
file_name = args.model + "_results/aug_" + str(m) + ".csv"
not_done = True
while not_done and m < 1000:
    my_file = Path(file_name)
    if my_file.is_file():
        m = m + 1
        file_name = args.model + "_results/aug_" + str(m) + ".csv"
    else:
        file = open(file_name, 'w+')
        not_done = False
        break

    

for k in range(len(symbol_list)):



    ## processing data
    print("processing data")

    prices = X[:,k][-1000:] / X[:,k][-1000:].std()

    input_size = 30
    prices = torch.tensor(prices).float()
    fullX = torch.stack([prices[i-input_size: i] for i in range(input_size, len(prices))])
    fullX = fullX[:-1]#.transpose(0, 1)
    Y = ((prices[1:] - prices[:-1]) / prices[:-1])[input_size:]#.transpose(0, 1)
    fullRTN = ((prices[1:] - prices[:-1]) / prices[:-1])#.transpose(0, 1)
    fullRTN = torch.stack([fullRTN[i-input_size: i] for i in range(input_size, len(fullRTN))])#.transpose(2, 1).transpose(0, 1)


    fullX = fullX.cuda()
    Y = Y.cuda().unsqueeze(1)
    fullRTN = fullRTN.cuda()


    num_test = 2000
    Xtrain = fullX[:800]
    Ytrain = Y[:800]
    Xtest = fullX[800:]
    Ytest = Y[800:]

 
        
    if args.model == 'lstm':
        Xtest = Xtest.transpose(1,0).unsqueeze(2)
        
    RTNtrain = fullRTN[:800]
    meanout =  Ytrain.mean(dim=0)

    
    
    ## training
    
    losses = []
    STEP = 2500    
    f_aug = Net().cuda()
    optimizer = torch.optim.Adam(f_aug.parameters(), lr=1e-3, weight_decay=1e-5)

    for i in range(STEP):
      loss = train(Xtrain,Ytrain,criterion,f_aug,optimizer, sigma=0.5, lbd=0.1, minibatch=minibatch)


    ## training finished
    print("training finished")

    portfolio = f_aug(Xtest).squeeze(1)
    print("average portfolio position: ", portfolio.sum().mean())#, portfolio_input.sum())
    DeltaW = 1 + (wealth_gain(portfolio, Ytest.squeeze(1))).T.detach().cpu().numpy() 
    size = 2
    Wealth_trajectory = [np.ones(size-1)]
    for i in range(len(DeltaW)):
      Wealth_trajectory.append(Wealth_trajectory[-1] * DeltaW[i])
    message = "results," + symbol_list[k] + ',' + str(sharpe(Wealth_trajectory)) + '\n'
    print("results: ", symbol_list[k], sharpe(Wealth_trajectory))
    file.write(message)
    file.flush()
file.close()
