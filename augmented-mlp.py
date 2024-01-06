import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

All = np.loadtxt("monk2.txt",delimiter=",")
Null_val =-1000001
#print(All.shape[0])
#print(All.shape[1])
#index = np.random.choice(All.shape[0], (len(All)*0.5), replace=False)
from sklearn.model_selection import train_test_split

#train1 = np.loadtxt("Train1.txt",delimiter=",")
#test1 = np.loadtxt("Test1.txt",delimiter=",")
#test_miss1 = np.loadtxt("Test1_100.txt",delimiter=",")
num_clusters = 2
learning_rate = 0.0001
epochs = 1000
lamda = 10

#print(class_number)
Repeat = 0
MLP_w_m_v1=0
AMLP_w_m_v1=0

#Zero Imputation
MLP_w_m_v2=0
AMLP_w_m_v2=0

#Rand Imputation
MLP_w_m_v3=0
AMLP_w_m_v3=0

#Mean Imputation
MLP_w_m_v4=0
AMLP_w_m_v4=0

#1-NN Imputation
MLP_w_m_v5=0
AMLP_w_m_v5=0

#Weight
W1_all=0
W2_all=0
W3_all=0

#---------Repeat all process--------
while(Repeat<10):
    np.random.seed(Repeat)
    lamda1 = 0
    lamda2 = 0
    W1 = 0.33
    W2 = W1
    W3 = W1
    train1, test1, y_train, y_test = train_test_split(All, All[:,All.shape[1]-1], test_size=0.5)
    test_miss1 = test1.copy()
    #print(train1)
    import random
    for i1 in range(test_miss1.shape[0]):
      counter = 0
      for i2 in range(test_miss1.shape[1]-1):
        rand1 = random.uniform(0, 1)
        #print((All.shape[1]/2))
        #print(counter)
        #print(rand1)
        if (counter < (test_miss1.shape[1]/2)  and rand1 >=0.5):
        #if (counter <= 1  and rand1 >=0.5):
          test_miss1[i1,i2] = Null_val
          counter =counter+1
        if (counter >= (test_miss1.shape[1]/2)):
           break;

    Repeat = Repeat+1
    print(Repeat)

    columns1 = [(i)
              for i in range(0, len(train1[0])-1)]
    columns1.append('clas')
    print(columns1)

    Train = pd.DataFrame(train1, columns= columns1)#= [0,1,2,3,4,5,6,'clas'])
    Classes = Train.clas.unique()
    train =Train.copy()
    Test = pd.DataFrame(test1, columns= columns1)# = [0,1,2,3,4,5,6,'clas'])
    test = Test
    Test_miss = pd.DataFrame(test_miss1, columns= columns1)# = [0,1,2,3,4,5,6,'clas'])
    input_feature = train1.shape[1]-1
    hidden_node = input_feature*10
    class_number = len(np.unique(y_train))

    """K-Means Clustering and finding Centroids"""
    from kmeans_pytorch import kmeans, kmeans_predict
    import torch.utils.data as data_utils
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    cluster_centres_all = []
    for i in range(class_number):
      class0 = train.loc[train["clas"] == Classes[i]]
      #centroid = class0.mean(axis=0)
      class0 = class0.drop('clas', axis=1)
      class0_tensor = torch.tensor(class0.to_numpy())
      cluster_ids_x0, cluster_centers0 = kmeans(X=class0_tensor, num_clusters=num_clusters, distance='euclidean')
      if i>0:
       cluster_centres_all = np.concatenate((cluster_centres_all,cluster_centers0),axis=0)
      else:
        cluster_centres_all = cluster_centers0
    centroid_list = cluster_centres_all.tolist()
    #print((centroid_list))
    center_list = []
    for i in range(class_number):
      class0 = train.loc[train["clas"] == Classes[i]]
      class0 = class0.drop('clas', axis=1)
      centroid = class0.mean(axis=0)
      centroid = np.expand_dims(centroid, axis=0)
      #print(np.transpose(centroid))
      #print(len(centroid_list))
      if i>0:
        center_list = np.concatenate(((centroid),center_list),axis=0)
      else:
        center_list = (centroid)
    centroid_list.extend(center_list)
    len(centroid_list)

    """Training MLP on given dataset points

    """

    encoder = LabelEncoder()
    train['clas'] = encoder.fit_transform(train['clas'])

    #train

    train=train.iloc[np.random.permutation(len(train))]
    train=train.reset_index(drop=True)
    #train.head()

    X = train.drop("clas",axis=1).values
    y = train["clas"].values

    X_train = X
    #X_test = X
    y_train = y
    #y_test = y

    X_train = torch.FloatTensor(X_train)
    #X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    #y_test = torch.LongTensor(y_test)

    class Model(nn.Module):
        def __init__(self, input_features=input_feature, hidden_layer1=hidden_node, output_features=class_number):
            super().__init__()
            self.fc1 = nn.Linear(input_features,hidden_layer1)
            self.out = nn.Linear(hidden_layer1, output_features)
            self.relu = nn.ReLU()

        def forward(self, x):
            x1 = F.relu(self.fc1(x))
            #x = F.relu(self.out(x))
            x=self.out(x1)
            return x,x1

    model1     = Model()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    #model1

    #train MLP on given data points
    #import tensorflow as tf
    losses1 = []

    for i in range(epochs):
        y_pred,w_y_pred = model1.forward(X_train)

        loss = criterion1(y_pred, y_train) #+ lamda1*(loss1/len(y_pred)) - lamda2*(loss2/len(y_pred))
        losses1.append(loss)
        #if(i%2 == 0):
         # print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

    """Training MLP on Augmented Data points

    ### *Augmenting dataset with 1-NN*
    """

    train = Train.copy()#pd.read_csv('Train4.txt')

    def distance_calculate_and_return_neighbour(total_feature,feature_vec,missing_index):
      initial_list = list(range(0,total_feature))
      initial_list.remove(missing_index)
      #print(initial_list)
      min_index = 0
      index=0
      min = 1000000
      for centroid in centroid_list:
        sum=0
        for k in initial_list:
          #print(feature_vec[k])
          #print(centroid[k])
          sum = sum+(feature_vec[k]-centroid[k])**2
        if(sum <=min):
          min=sum
          min_index = index
        index = index+1
      #print(min_index)
      missing = round(centroid_list[min_index][missing_index],2)
      return missing

    class AE(nn.Module):
        def __init__(self, input_features=input_feature, hidden_layer1=hidden_node, output_features=input_feature):
            super().__init__()
            self.fc1 = nn.Linear(input_features,hidden_layer1)
            self.out = nn.Linear(hidden_layer1, output_features)
            self.relu = nn.ReLU()

        def forward(self, x):
            x1 = F.relu(self.fc1(x))
            #x = F.relu(self.out(x))
            x=self.out(x1)
            return x,x1

    model0     = AE()
    criterion0 = nn.CrossEntropyLoss()
    optimizer0 = torch.optim.Adam(model0.parameters(), lr=learning_rate)
    for i in range(epochs):
          y_pred,w_y_pred = model0.forward(X_train)
          #y_pred,w_y_pred = model2.forward(X_train)
          loss = criterion0(y_pred, X_train)
          optimizer0.zero_grad()
          loss.backward()
          optimizer0.step()
    #preds1 = []
    preds1 = y_pred



    """### *Second Method*"""
    class AE(nn.Module):
        def __init__(self, input_features=input_feature,latent_dims = hidden_node, hidden_layer1=hidden_node, output_features=input_feature):
            super().__init__()

            self.fc1 = nn.Linear(input_features,hidden_layer1)
            self.out = nn.Linear(hidden_layer1, output_features)

            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_layer1, latent_dims)
            self.linear3 = nn.Linear(hidden_layer1, latent_dims)

            self.N = torch.distributions.Normal(0, 1)
            #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            #self.N.scale = self.N.scale.cuda()
            self.kl = 0
            self.out1 = nn.Linear(hidden_layer1, input_features)

        def forward(self, x):
           x1 = F.relu(self.fc1(x))
           #x = F.relu(self.out(x))
           x0=  x
           x00=self.out(x1)
           #x2 = self.out1(x1)
           #x2=self.out1(x1)
           #x = torch.flatten(x0, start_dim=1)
           x = F.relu(self.fc1(x))
           mu =  self.linear2(x)
           sigma = torch.exp(self.linear3(x))
           x2 = mu + sigma*self.N.sample(mu.shape)
           x2=self.out1(x2)
           self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
           return x00,x2




    model1     = AE()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    for i in range(epochs):
        y_pred,w_y_pred = model1.forward(X_train)
        loss = criterion1(y_pred, y_train)
        loss = loss +  criterion2(w_y_pred, X_train)
        #if(i%2 == 0):
         # print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    #preds1 = []
    preds10 = w_y_pred

    """### *Third Method*"""
    mu, sigma = 0, 0.1 ;
    noise = np.random.normal(mu, sigma, [train1.shape[0],train1.shape[1]-1]);
    w_y_pred = X
    #preds1 = []
    preds100 = w_y_pred + noise
    #preds2 = preds1.detach().numpy()

    #####################################################################################################################################
    #import tensorflow.compat.v1 as tf
    #import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    h_matrix = np.random.uniform(low=1.0, high=1.0, size=(1,1))
    #tf.disable_eager_execution()
    #tf.compat.v2.disable_eager_execution()
    X0= tf.placeholder(tf.float32)
    X1 = tf.placeholder(tf.float32)
    X2 = tf.placeholder(tf.float32)
    X3 = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    W01 = tf.Variable(W1,dtype=tf.float32)
    W02 = tf.Variable(W2,dtype=tf.float32)
    W03 = tf.Variable(W3,dtype=tf.float32)
    preds02 = W01*X1 + W02*X2 + W03*X3
    X = tf.concat([X0, preds02], 0)
    H = tf.placeholder(tf.float32)
    n_input = input_feature
    n_hidden = hidden_node
    n_output = class_number
    #y_train
    y_train0=np.zeros([train1.shape[0],n_output])
    y_train00 = y_train.detach().numpy()
    for sjc1 in range(train1.shape[0]):
      y_train0[sjc1,y_train00[sjc1]]=1

    #weights
    W1_h = tf.Variable(tf.random.uniform([
        n_input, n_hidden], -0.3, 0.3))
    W2_h = tf.Variable(tf.random.uniform(
        [n_hidden, n_output], -0.3, 0.3))

    #bias
    b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
    b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")

    L2 = tf.sigmoid(tf.matmul(X, W1_h) + b1)
    hy = tf.sigmoid(tf.matmul(L2, W2_h) + b2)

    #cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy)) #cross entropy
    cost = tf.reduce_mean(tf.pow(tf.subtract(Y, hy), 2)) + lamda*(W01 * tf.math.log(W01) + W02 * tf.math.log(W02) + W03 * tf.math.log(W03))
    Min1 = tf.minimum(W01,W02)
    Min1 = tf.minimum(W03,Min1)
    W01 = W01 + (Min1 * (-1)) + tf.constant(1e-3,dtype=tf.float32)
    W02 = W02 + (Min1 * (-1)) + tf.constant(1e-3,dtype=tf.float32)
    W03 = W03 + (Min1 * (-1)) + tf.constant(1e-3,dtype=tf.float32)
    W_sum= W01 + W02 + W03
    W01 = W01 / W_sum
    W02 = W02 / W_sum
    W03 = W03 / W_sum
    #!pip uninstall tensorflow==2.2.0
    #!pip install tensorflow==1.15.0
    #!pip install bert-tensorflow
    #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(cost)

    init = tf.initialize_all_variables()
    with tf.Session() as session:
      session.run(init)
      for step in range(epochs*2):
           session.run(optimizer, feed_dict={X0:X_train,X1:preds1.detach().numpy(),X2:preds10.detach().numpy(), X3:preds100, Y:np.concatenate((y_train0, y_train0), axis=0), H:h_matrix})
           W1 =  session.run(W01, feed_dict={X0:X_train,X1:preds1.detach().numpy(),X2:preds10.detach().numpy(), X3:preds100, Y:np.concatenate((y_train0, y_train0), axis=0), H:h_matrix})
           W2 =  session.run(W02, feed_dict={X0:X_train,X1:preds1.detach().numpy(),X2:preds10.detach().numpy(), X3:preds100, Y:np.concatenate((y_train0, y_train0), axis=0), H:h_matrix})
           W3 =  session.run(W03, feed_dict={X0:X_train,X1:preds1.detach().numpy(),X2:preds10.detach().numpy(), X3:preds100, Y:np.concatenate((y_train0, y_train0), axis=0), H:h_matrix})

    #####################################################################################################################################
    print(W1)
    print(W2)
    print(W3)
    preds2 = W1*preds1.detach().numpy() + W2*preds10.detach().numpy() + W3*preds100
    preds2=torch.from_numpy(preds2)


    train = torch.cat((X_train, preds2), 0)


    """### *Training the MLP*"""

    X_aug = train.clone()
    y_aug = torch.cat((y_train, y_train), 0)
    #y_aug = pd.concat([y_train, y_train], ignore_index = True).values

    X_train_aug = X_aug.clone()
    #X_test = X
    y_train_aug = y_aug.clone()
    #y_test = y

    X_train_aug = torch.FloatTensor(X_train_aug.numpy())
    #X_test = torch.FloatTensor(X_test)
    y_train_aug = torch.LongTensor(y_train_aug)
    #y_test = torch.LongTensor(y_test)

    class Model(nn.Module):
        def __init__(self, input_features=input_feature, hidden_layer1=hidden_node, output_features=class_number):
            super().__init__()
            self.fc1 = nn.Linear(input_features,hidden_layer1)
            self.out = nn.Linear(hidden_layer1, output_features)
            self.relu = nn.ReLU()

        def forward(self, x):
            x1 = F.relu(self.fc1(x))
            #x = F.relu(self.out(x))
            x=self.out(x1)
            return x,x1

    model2     = Model()
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
    model2

    #train MLP on given data points + augmented data points
    #epochs = 1000
    losses2 = []

    for i in range(epochs*2):
        if (i<(epochs)):
          y_pred,w_y_pred = model2.forward(X_train)
          loss = criterion2(y_pred, y_train)
          losses2.append(loss)
          optimizer2.zero_grad()
          loss.backward()
          optimizer2.step()
        else:
          y_pred,w_y_pred = model2.forward(X_train_aug)
          y_train_aug2 = y_train_aug#[index]
          loss = criterion2(y_pred, y_train_aug2) #+ lamda1*(loss1/len(y_pred)) - lamda2*(loss2/len(y_pred))
          losses2.append(loss)
          optimizer2.zero_grad()
          loss.backward()
          optimizer2.step()

    """Testing both the MLP models

    ### *Without any missing value*
    """

    test = Test#pd.read_csv('Test4.txt')

    encoder = LabelEncoder()
    test['clas'] = encoder.fit_transform(test['clas'])

    test=test.iloc[np.random.permutation(len(test))]
    test=test.reset_index(drop=True)
    #test

    X1 = test.drop("clas",axis=1).values
    y1 = test["clas"].values

    X_test = X1
    y_test = y1

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    #Test_miss = pd.DataFrame(test_miss1, columns= columns1)# = [0,1,2,3,4,5,6,'clas'])
    #Test_miss

    """***Method 1 - MLP on normal dataset***"""

    preds1 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model1.forward(val)
            preds1.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds1})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    df

    MLP_w_m_v=df['Correct'].sum() / len(df)
    ##print(MLP_w_m_v)
    """***Method 2 - MLP on augmented dataset***"""

    preds2 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model2.forward(val)
            preds2.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds2})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    #df

    AMLP_w_m_v=df['Correct'].sum() / len(df)
    #print(AMLP_w_m_v)



    MLP_w_m_v1=MLP_w_m_v1+MLP_w_m_v
    AMLP_w_m_v1=AMLP_w_m_v1+AMLP_w_m_v


    """### *Missing Values - replace with zero*"""

    test = Test_miss.copy()

    for i in range(len(test)):
      for j in range(input_feature):
        if(test[j][i] == Null_val):
          test[j][i] = 0

    encoder = LabelEncoder()
    test['clas'] = encoder.fit_transform(test['clas'])
    test

    test=test.iloc[np.random.permutation(len(test))]
    test=test.reset_index(drop=True)

    X1 = test.drop("clas",axis=1).values
    y1 = test["clas"].values

    X_test = X1
    y_test = y1

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    """***Method 1 - MLP on normal dataset***"""

    preds1 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model1.forward(val)
            preds1.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds1})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    df

    MLP_w_m_v=df['Correct'].sum() / len(df)
    ##print(MLP_w_m_v)
    """***Method 2 - MLP on augmented dataset***"""

    preds2 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model2.forward(val)
            preds2.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds2})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    #df

    AMLP_w_m_v=df['Correct'].sum() / len(df)
    ##print(AMLP_w_m_v)


    MLP_w_m_v2=MLP_w_m_v2+MLP_w_m_v
    AMLP_w_m_v2=AMLP_w_m_v2+AMLP_w_m_v


    """### *Missing Values - replace with random value*"""

    import random

    test = Test_miss.copy()#pd.read_csv('Test4_100.txt')

    for i in range(len(test)):
      for j in range(input_feature):
        if(test[j][i] == Null_val):
          test[j][i] = round(random.uniform(0, 5),2)

    encoder = LabelEncoder()
    test['clas'] = encoder.fit_transform(test['clas'])
    test

    test=test.iloc[np.random.permutation(len(test))]
    test=test.reset_index(drop=True)

    X1 = test.drop("clas",axis=1).values
    y1 = test["clas"].values

    X_test = X1
    y_test = y1

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)


    """***Method 1 - MLP on normal dataset***"""

    preds1 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model1.forward(val)
            preds1.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds1})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    df

    MLP_w_m_v=df['Correct'].sum() / len(df)
    ##print(MLP_w_m_v)
    """***Method 2 - MLP on augmented dataset***"""

    preds2 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model2.forward(val)
            preds2.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds2})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    #df

    AMLP_w_m_v=df['Correct'].sum() / len(df)
    #print(AMLP_w_m_v)


    MLP_w_m_v3=MLP_w_m_v3+MLP_w_m_v
    AMLP_w_m_v3=AMLP_w_m_v3+AMLP_w_m_v

    """### *Missing Values - replace with mean feature value*"""

    test = Test_miss.copy()#pd.read_csv('Test4_100.txt')
    train = Train.copy()#pd.read_csv('Train4.txt')

    del train['clas']
    #T= train.drop('class', axis=0)
    mean_values = train.mean(axis=0)
    #print(mean_values)
    #mean_values = mean_values.drop('class', axis=1)
    #mean_values =mean_values.drop('class', axis=1)

    for i in range(len(test)):
      for j in range(input_feature):
        if(test[j][i] == Null_val):
          test[j][i] = mean_values[j]

    encoder = LabelEncoder()
    test['clas'] = encoder.fit_transform(test['clas'])
    #test

    test=test.iloc[np.random.permutation(len(test))]
    test=test.reset_index(drop=True)

    X1 = test.drop("clas",axis=1).values
    y1 = test["clas"].values

    X_test = X1
    y_test = y1

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)


    """***Method 1 - MLP on normal dataset***"""

    preds1 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model1.forward(val)
            preds1.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds1})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    df

    MLP_w_m_v=df['Correct'].sum() / len(df)
    ##print(MLP_w_m_v)
    """***Method 2 - MLP on augmented dataset***"""

    preds2 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model2.forward(val)
            preds2.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds2})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    #df

    AMLP_w_m_v=df['Correct'].sum() / len(df)
    #print(AMLP_w_m_v)


    MLP_w_m_v4=MLP_w_m_v4+MLP_w_m_v
    AMLP_w_m_v4=AMLP_w_m_v4+AMLP_w_m_v


    """### *Missing Values - replace with 1-NN value*"""

    test = Test_miss.copy()#pd.read_csv('Test4_100.txt')

    def distance_calculate_and_return_neighbour(total_feature,feature_vec,missing_index):
      initial_list = list(range(0,total_feature))
      initial_list.remove(missing_index)
      #print(initial_list)
      min_index = 0
      index=0
      min = 100000
      for centroid in centroid_list:
        sum=0
        for k in initial_list:
          #print(feature_vec[k])
          #print(centroid[k])
          sum = sum+(feature_vec[k]-centroid[k])**2
        if(sum <=min):
          min=sum
          min_index = index
        index = index+1
      #print(min_index)
      missing = round(centroid_list[min_index][missing_index],2)
      return missing

    for i in range(len(test)):
      for j in range(input_feature):
        if(test[j][i] == Null_val):
          replace_num = distance_calculate_and_return_neighbour(input_feature,test.iloc[i],j)
          test[j][i] = replace_num

    encoder = LabelEncoder()
    test['clas'] = encoder.fit_transform(test['clas'])
    test

    test=test.iloc[np.random.permutation(len(test))]
    test=test.reset_index(drop=True)

    X1 = test.drop("clas",axis=1).values
    y1 = test["clas"].values

    X_test = X1
    y_test = y1

    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)


    """***Method 1 - MLP on normal dataset***"""

    preds1 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model1.forward(val)
            preds1.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds1})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    df

    MLP_w_m_v=df['Correct'].sum() / len(df)
    ##print(MLP_w_m_v)
    """***Method 2 - MLP on augmented dataset***"""

    preds2 = []
    with torch.no_grad():
        for val in X_test:
            y_hat,w_y_pred = model2.forward(val)
            preds2.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds2})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
    #df

    AMLP_w_m_v=df['Correct'].sum() / len(df)
    #print(AMLP_w_m_v)


    MLP_w_m_v5=MLP_w_m_v5+MLP_w_m_v
    AMLP_w_m_v5=AMLP_w_m_v5+AMLP_w_m_v

    W1_all = W1_all + W1
    W2_all = W2_all + W2
    W3_all = W3_all + W3




"""## **Results**"""


print(Repeat)

print("Without Missing values with MLP and AMLP")
print(MLP_w_m_v1/Repeat)
print(AMLP_w_m_v1/Repeat)


print("Zero imputation with MLP and AMLP")
print(MLP_w_m_v2/Repeat)
print(AMLP_w_m_v2/Repeat)

print("Rand imputation with MLP and AMLP")
print(MLP_w_m_v3/Repeat)
print(AMLP_w_m_v3/Repeat)

print("Mean imputation with MLP and AMLP")
print(MLP_w_m_v4/Repeat)
print(AMLP_w_m_v4/Repeat)

print("1-NN imputation with MLP and AMLP")
print(MLP_w_m_v5/Repeat)
print(AMLP_w_m_v5/Repeat)

print("Average Weight")
print(W1_all/Repeat)
print(W2_all/Repeat)
print(W3_all/Repeat)
