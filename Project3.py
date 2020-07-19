
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torch.optim as optim
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('D:/NOW/FTEC5580/Project/Project3/NVDA.csv')
df.shape


# In[3]:


import seaborn as sns; sns.set(style="whitegrid")
price_path = np.array(df['Adj Close'])
df.plot(x="Date", y="Adj Close", figsize=(10, 7))


# In[4]:


price_path_scale = (price_path - np.mean(price_path)) / np.std(price_path)
fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(price_path_scale)


# In[5]:


lag = 10
# stack the blocks along axis 0
data = np.concatenate([price_path_scale[i: i+lag+1].reshape(1, -1)
                       for i in range(len(price_path_scale)-lag)], 0)
print("shape of data:", data.shape)


# In[6]:



# devide data into 3 parts
trainData = torch.from_numpy(data[0:643]).float()
valiData = torch.from_numpy(data[643:743]).float()
testData = torch.from_numpy(data[743:]).float()

# create PyTorch datasets
trainset = Data.TensorDataset(
    trainData[:, 0:-1], trainData[:, -1:])
valiset = Data.TensorDataset(
    valiData[:, 0:-1], valiData[:, -1:])
testset = Data.TensorDataset(
    testData[:, 0:-1], testData[:, -1:])

# create PyTorch dataloaders
trainloader = Data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=0)
valiloader = Data.DataLoader(
    valiset, batch_size=32, shuffle=True, num_workers=0)
testloader = Data.DataLoader(
    testset, batch_size=32, shuffle=True, num_workers=0)


# In[7]:


class RNN_net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(RNN_net, self).__init__()  # inherit methods from torch.nn.Module

        self.rnn = torch.nn.RNN(input_size=n_feature, hidden_size=n_hidden,
                                num_layers=1, batch_first=True)  # rnn layer
        self.output = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x, _ = self.rnn(x.unsqueeze(2))
        x = self.output(x[:, -1, :])
        return x


class GRU_net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(GRU_net, self).__init__()  # inherit methods from torch.nn.Module

        self.rnn = torch.nn.GRU(input_size=n_feature, hidden_size=n_hidden,
                                num_layers=1, batch_first=True)  # rnn layer
        self.output = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x, _ = self.rnn(x.unsqueeze(2))
        x = self.output(x[:, -1, :])
        return x


class LSTM_net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # inherit methods from torch.nn.Module
        super(LSTM_net, self).__init__()

        self.rnn = torch.nn.LSTM(input_size=n_feature, hidden_size=n_hidden,
                                 num_layers=1, batch_first=True)  # rnn layer
        self.output = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x, _ = self.rnn(x.unsqueeze(2))
        x = self.output(x[:, -1, :])
        return x


class FNN_net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(FNN_net, self).__init__()  # inherit methods from torch.nn.Module

        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # h1 layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # h2 layer
        self.output = torch.nn.Linear(n_hidden2, n_output)  # output layer
        self.n_feature = n_feature

    def forward(self, x):
        x = torch.relu_(self.hidden1(x))
        x = torch.relu_(self.hidden2(x))
        x = self.output(x)
        return x


# In[8]:


def test(model, dataloader):
    with torch.no_grad():
        running_loss = 0.0
        batch_num = 0
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = model(inputs)  # forward propagation, get outputs
            loss = criterion(outputs, labels)  # get loss

            # record loss and other statistics
            running_loss += inputs.shape[0] * loss.item()
            batch_num += inputs.shape[0]
    return (running_loss / batch_num)


# In[9]:


criterion = torch.nn.MSELoss()


def train(model, num_epoch, dataloader):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_train_list = []  # record training loss
    loss_vali_list = []  # record validation loss
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
#             print(inputs.shape, labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)  # forward propagation, get outputs
#             print(outputs.squeeze())
#             print(labels.shape, outputs.shape)
            loss = criterion(outputs, labels)  # get loss
            loss.backward()  # back propagation, get gradients of loss
            optimizer.step()  # optimize one step

            # record loss and other statistics
            running_loss += inputs.shape[0] * loss.item()
            batch_num += inputs.shape[0]
        loss_train = running_loss / batch_num
        loss_vali = test(model, valiloader)
#         print('[%4d] train loss %.5f | validation loss %.5f' %
#               (epoch+1, loss_train, loss_vali))

        loss_train_list.append(loss_train)
        loss_vali_list.append(loss_vali)
    print('Finished Training')
    return loss_train_list, loss_vali_list


# In[10]:


rnn = RNN_net(1, 8, 1)
gru = GRU_net(1, 8, 1)
lstm = LSTM_net(1, 8, 1)
fnn = FNN_net(10, 8, 8, 1)

net_names = ["RNN", "GRU", "LSTM", "FNN"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.set_title("Training Loss")
ax2.set_title("Validation Loss")
for i, net in enumerate([rnn, gru, lstm, fnn]):
    print("Training", net_names[i])
    loss_train, loss_vali = train(net, 500, trainloader)
    ax1.plot(range(1, 1+len(loss_train)), loss_train, label=net_names[i])
    ax2.plot(range(1, 1+len(loss_vali)), loss_vali, label=net_names[i])
ax1.set_ylim([0, 0.1])
ax2.set_ylim([0, 0.1])
ax1.legend()
ax2.legend()


# In[11]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
ax1.set_title("Prediction on Training set")
ax2.set_title("Prediction on Validation set")
ax3.set_title("Prediction on Test set")

ax1.plot(trainData[500:550, -1], label="label")
ax2.plot(valiData[:, -1], label="label")
ax3.plot(testData[:, -1], label="label")

print("  Net | Train loss | Vali Loss | Test Loss")
print("------------------------------------------")
for i, net in enumerate([rnn, gru, lstm, fnn]):
    ax1.plot(net(trainData[500:550, :-1])
             [:, ].squeeze().data.numpy(), label=net_names[i])
    ax2.plot(net(valiData[:, :-1])
             [:, 0].squeeze().data.numpy(), label=net_names[i])
    ax3.plot(net(testData[:, :-1])
             [:, 0].squeeze().data.numpy(), label=net_names[i])

    print("%5s |  %8.5f  |  %8.5f | %8.5f" % (net_names[i], test(
        net, trainloader), test(net, valiloader), test(net, testloader)))
ax1.legend()
ax2.legend()
ax3.legend()


# In[12]:


criterion1 = torch.nn.MSELoss()


def train_1(model, num_epoch, dataloader,lr):
    optimizer = optim.Adam(model.parameters(), lr)
    loss_train_list = []  # record training loss
    loss_vali_list = []  # record validation loss
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
#             print(inputs.shape, labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)  # forward propagation, get outputs
#             print(outputs.squeeze())
#             print(labels.shape, outputs.shape)
            loss = criterion1(outputs, labels)  # get loss
            loss.backward()  # back propagation, get gradients of loss
            optimizer.step()  # optimize one step

            # record loss and other statistics
            running_loss += inputs.shape[0] * loss.item()
            batch_num += inputs.shape[0]
        loss_train = running_loss / batch_num
        loss_vali = test(model, valiloader)
#         print('[%4d] train loss %.5f | validation loss %.5f' %
#               (epoch+1, loss_train, loss_vali))

        loss_train_list.append(loss_train)
        loss_vali_list.append(loss_vali)
    print('Finished Training')
    return loss_train_list, loss_vali_list


# In[15]:


def demo(m,lr):
    rnn = RNN_net(1, 8, 1)
    gru = GRU_net(1, 8, 1)
    lstm = LSTM_net(1, 8, 1)
    fnn = FNN_net(10, 8, 8, 1)

    net_names = ["RNN", "GRU", "LSTM", "FNN"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.set_title("Training Loss")
    ax2.set_title("Validation Loss")
    for i, net in enumerate([rnn, gru, lstm, fnn]):
        print("Training", net_names[i])
        loss_train, loss_vali = train_1(net, 500, trainloader,lr)
        ax1.plot(range(1, 1+len(loss_train)), loss_train, label=net_names[i])
        ax2.plot(range(1, 1+len(loss_vali)), loss_vali, label=net_names[i])
    ax1.set_ylim([0, 0.1])
    ax2.set_ylim([0, 0.1])
    ax1.legend()
    ax2.legend()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    ax1.set_title("Prediction on Training set")
    ax2.set_title("Prediction on Validation set")
    ax3.set_title("Prediction on Test set")

    ax1.plot(trainData[500:550, -1], label="label")
    ax2.plot(valiData[:, -1], label="label")
    ax3.plot(testData[:, -1], label="label")

    print("  Net | Train loss | Vali Loss | Test Loss")
    print("------------------------------------------")
    for i, net in enumerate([rnn, gru, lstm, fnn]):
        ax1.plot(net(trainData[500:550, :-1])
                 [:, ].squeeze().data.numpy(), label=net_names[i])
        ax2.plot(net(valiData[:, :-1])
                 [:, 0].squeeze().data.numpy(), label=net_names[i])
        ax3.plot(net(testData[:, :-1])
                 [:, 0].squeeze().data.numpy(), label=net_names[i])

        print("%5s |  %8.5f  |  %8.5f | %8.5f" % (net_names[i], test(
            net, trainloader), test(net, valiloader), test(net, testloader)))
    ax1.legend()
    ax2.legend()
    ax3.legend()


# In[16]:


demo(100,0.10)


# In[17]:


demo(500,0.001)

