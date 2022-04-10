#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


def ReLu(x):
    return np.maximum(x, 0)


# In[4]:


def SoftMax(x):
    #防止溢出
    M = np.max(x, axis=0, keepdims = True)
    x = np.exp(x-M)
    epsilon = 1e-6
    s = x / (epsilon+np.sum(x, axis=0, keepdims=True))
    return s


# In[5]:


def CrossEntropy(y_hat, y):
    #交叉熵函数
    loss = -np.mean(np.sum(y * np.log(y_hat), axis = 0))
    return loss


# In[6]:


import pickle

class Nerwork:
    def __init__(self, input_dim, num_classes = 10, hidden_dim = 1024, seed =287):
        # 初始化权重和梯度
        #weight_0:[hidden_dim, input_dim]
        #bias_0:[hidden_dim, 1]
        #weight_1:[num_classes, hidden_dim]
        #bias_1:[num_classes, 1]
        np.random.seed(seed)
        self.weight_0 = np.random.normal(0.0, np.sqrt(2 / input_dim),(hidden_dim, input_dim))
        self.bias_0 = np.zeros((hidden_dim, 1))
        self.weight_1 = np.random.normal(0.0, np.sqrt(2 / hidden_dim),(num_classes, hidden_dim))
        self.bias_1 = np.zeros((num_classes, 1))

        self.weight_0_grad = np.zeros((hidden_dim, input_dim))
        self.bias_0_grad = np.zeros((hidden_dim, 1))
        self.weight_1_grad = np.zeros((num_classes, hidden_dim))
        self.bias_1_grad = np.zeros((num_classes, 1))

    def loss(self, input, label, train = True):
        #input:(input_dim, samples)
        #weight_0:[hidden_dim, input_dim]
        #bias_0:[hidden_dim, 1]
        #weight_1:[num_classes, hidden_dim]
        #bias_1:[num_classes, 1]
        hidden_before = self.weight_0 @ input + self.bias_0
        hidden_after = ReLu(hidden_before)
        #hidden_before,hidden_after:[hidden_dim, samples]
        logits = self.weight_1 @ hidden_after + self.bias_1
        results = SoftMax(logits)
        #print(results)
        #logits,results:[num_classes, samples]
        #if train:
        self.cache = (input, hidden_before, hidden_after, logits, results, label)
        return CrossEntropy(results, label)
    
    def backward(self):
        #input:(input_dim, samples)
        input, hidden_before, hidden_after, logits, results, label = self.cache
        m = results.shape[1]
        #logits,results:[num_classes, samples]
        d_logit = 1/m * (results - label)
        #d_logit:[num_classes, samples]
        d_bias_1 = np.sum(d_logit, axis=1, keepdims = True)
        d_weight_1 = d_logit @ hidden_after.T
        #d_bias_1:[num_classes, 1]
        #hidden_after:[hidden_dim, samples]
        #d_weight_1:[num_classes, hidden_dim]

        #input:(input_dim, samples)
        #weight_0:[hidden_dim, input_dim]
        #bias_0:[hidden_dim, 1]
        #weight_1:[num_classes, hidden_dim]
        #bias_1:[num_classes, 1]
        d_hidden_after = self.weight_1.T @ d_logit
        #d_hidden_after,d_hidden_before,hidden_before:[hidden_dim, samples]
        d_hidden_before = (hidden_before > 0) * d_hidden_after
        d_weight_0 = d_hidden_before @ input.T
        #d_weight_0:[hidden_dim, input_dim]
        d_bias_0 = np.sum(d_hidden_before, axis=1, keepdims = True)


        self.weight_0_grad = d_weight_0 
        self.bias_0_grad = d_bias_0
        self.weight_1_grad = d_weight_1
        self.bias_1_grad = d_bias_1
    
    def l2regularization(self, reg):
        #l2正则化
        self.weight_0_grad = self.weight_0_grad  + 2 * reg * self.weight_0 
        self.bias_0_grad  = self.bias_0_grad  + 2 * reg * self.bias_0 
        self.weight_1_grad  = self.weight_1_grad  + 2 * reg * self.weight_1 
        self.bias_1_grad = self.bias_1_grad  + 2 * reg * self.bias_1 
        
    def SGD(self, lr):
        self.weight_0 = self.weight_0 - lr * self.weight_0_grad
        self.bias_0 = self.bias_0 - lr * self.bias_0_grad
        self.weight_1 = self.weight_1 - lr * self.weight_1_grad
        self.bias_1 = self.bias_1 - lr * self.bias_1_grad

    def save(self, path):        
        with open(path, 'wb') as file:
            pickle.dump(self, file)  


# In[7]:


def default_lr_decay(interval = 10, rate = 0.1):
    def lr_decay(epoch):
        return rate ** (epoch // interval)
    return lr_decay


# In[8]:


def random_mini_batches(X, Y, batch_size=64):
    m = X.shape[1]
    permutation = np.random.permutation(m)
    n = math.ceil(m / batch_size)
    for k in range(n):
        index = permutation[k * batch_size: (k + 1) * batch_size]
        yield (X[:,index], Y[index])


# In[9]:


MNIST_path = "./data"
import struct

def load_mnist(kind='train', path = MNIST_path):
    if kind == "test":
        kind = "t10k"

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    #labels = convert_to_one_hot(labels, 10)
    return images.T/ 255.0, labels


# In[10]:


def convert_to_one_hot(Y, num_classes = 10):
    Y = np.eye(num_classes)[Y.reshape(-1)].T
    return Y


# In[11]:


import math
import os


# In[12]:


data, label = load_mnist("train")


# In[13]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    #滑动平均
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[14]:


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #topk准确率
    #预测结果前k个中出现的正确结果的次数
    batch_size = target.shape[0]
    pred =  np.argmax(output, axis = 0)
    return np.mean(pred == target)


# In[15]:


def mkdir_if_missing(directory):
    #创建文件夹，如果这个文件夹不存在的话
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# In[16]:


def train(epoch = 20, hidden = 1024,reg = 1e-4, batch_size = 128, seed = 287, lr_init = 0.1, lr_scheduler = default_lr_decay(), save_path = None):
    
    mkdir_if_missing(save_path)
    np.random.seed(seed)
    train_image, train_label = load_mnist("train")
    test_image, test_label = load_mnist("test")
    
    #每个epoch的准确率
    training_acc_list = []
    test_acc_list = []
    
    training_loss_list = []
    test_loss_list = []
    lr_list = []
    
    #定义网络
    network = Nerwork(input_dim = 28 * 28, num_classes = 10, hidden_dim = hidden)
    
    for k in range(epoch):
        #定义单个epoch的acc和loss
        training_acc = AverageMeter()
        training_loss = AverageMeter()
        test_acc = AverageMeter()
        test_loss = AverageMeter()
        
        
        lr = lr_init * lr_scheduler(k)
        for image, label in random_mini_batches(train_image, train_label, batch_size):
            loss = network.loss(image, convert_to_one_hot(label, 10)) 
            logit = network.cache[3]
            training_loss.update(loss, label.shape[0])
            
            network.backward()
            network.l2regularization(reg)
            network.SGD(lr)
            
            acc = accuracy(logit, label)
            training_acc.update(acc, label.shape[0])
        
        training_acc_list.append(training_acc.avg)
        training_loss_list.append(training_loss.avg)
        lr_list.append(lr)

        
        for image, label in random_mini_batches(test_image, test_label, 200):
            loss = network.loss(image, convert_to_one_hot(label, 10)) 
            logit = network.cache[3]
            test_loss.update(loss, label.shape[0])

           
            acc = accuracy(logit, label)
            test_acc.update(acc, label.shape[0])            
        
        test_acc_list.append(test_acc.avg)
        test_loss_list.append(test_loss.avg)
        
        print("epoch:{:2d}".format(k), end = ",\t")
        print("train acc:{:5.3f}%".format(100 * training_acc.avg), end = ",\t")
        print("train loss:{:4.4f}".format(training_loss.avg), end = ",\t")
        print("test acc:{:5.3f}%".format(100 * test_acc.avg), end = ",\t")
        print("test loss:{:4.4f}".format(test_loss.avg))
        #[print(len(i)) for i in [lr_list, training_loss_list, training_acc_list, test_loss_list, test_acc_list]]

        save_file = os.path.join(save_path, "epoch_{}".format(k))
        network.save(save_file)
    
    pd.DataFrame({"epoch": list(range(epoch)),
                 "lr":lr_list,
                 "train loss": training_loss_list,
                 "train acc": training_acc_list,
                 "test loss": test_loss_list,
                 "test acc": test_acc_list
                 }).to_csv(os.path.join(save_path, "results.csv"))



# In[18]:


def test(load_file):
    with open(load_file, 'rb') as file:
        network = pickle.load(file)
    test_image, test_label = load_mnist("test")
    test_acc = AverageMeter()
    test_loss = AverageMeter()
    for image, label in random_mini_batches(test_image, test_label, 1000):
        loss = network.loss(image, convert_to_one_hot(label, 10))
        #labels = 
        test_loss.update(loss, label.shape[0])
        logit = network.cache[3]
        acc = accuracy(logit, label)
        test_acc.update(acc, label.shape[0])

    print("accuracy:{:.3f}%".format(100 * test_acc.avg), end = "\t")
    print("loss:{:.4}".format(test_loss.avg))
    
    return test_acc, test_loss
    


# In[19]:

