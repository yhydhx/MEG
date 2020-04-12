

#-------- Initialization --------

import os
import time
import h5py
import yaml
import argparse
import random
import scipy.io as sio
import numpy as np

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.autograd import Variable
from Model_5block import CNN


parser = argparse.ArgumentParser( description = 'Random Wired Neural Network for fMRI Time Series Classification', epilog = 'Created by Lin Zhao')
parser.add_argument('--config', default='configs/config_regular_c109_n32.yaml')
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--graph_model', default='WS', type=str, help='ER,BA,WS')
parser.add_argument('--nodes', default=30,type=int)
parser.add_argument('--model_dir', type=str)
parser.add_argument('--resume', default=False, help='resume')
parser.add_argument('--gpu_ids', default=1, help='gpu_ids')
parser.add_argument('--num_classes', default=2, help='number of classes')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
args = parser.parse_args()



class Random_Wired_CNN():

    
    def __init__(self):
        #default parameters.
        self.Task_Dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405,'1':115}
        self.Task = 'MOTOR'
        self.Num_TR =32   #self.Task_Dict[self.Task]
        self.Num_Class = 6
        #self.Data_Path = '../../data/sulci_2hinge_3hinge/ABIDEII-BNI_1/'
        self.Data_Path = './'
        self.Results_Path = '../../results/'
        self.subj_i = 1
        self.data_location = ""
	#use this function to load the data.
    def load_data(self):
        data_location = self.data_location
        raw_data = np.load(data_location+"y_norm.npy")
        
        data = []
        for line in raw_data:
            data.append(line[0])
        
        data = np.array(data )
        label = []

        for line in open(data_location+"../condid_label.csv"):
            label.append(line.strip())
        label = np.array(label)
    	#events = np.load(data_location+"events.npy")
        x_G0, y_G0, x_G1, y_G1 = [],[],[],[]

        for i in range(len(data)):

            if random.random() <0.75:
            #if count > 1847:
                x_G0.append(data[i])
                y_G0.append(label[i])
            else:
                x_G1.append(data[i])
                y_G1.append(label[i])

        x_G0 = np.array(x_G0)
        x_G1 = np.array(x_G1)
        
        y_G0 = np.array(y_G0)
        y_G1 = np.array(y_G1)
        
        x_G0 = x_G0.reshape(x_G0.shape[0],x_G0.shape[1],1)
        y_G0 = y_G0.astype(int)
        
        x_G1 = x_G1.reshape(x_G1.shape[0],x_G1.shape[1],1)
        y_G1 = y_G1.astype(int)
        print(np.shape(x_G0))
        print(np.shape(x_G1))
        return x_G0, y_G0, x_G1, y_G1

    def load_data_mat(self):
        data_location ="/home/share/nas2/AUSTISM/ABIDEII-EMC_1/Thr0.3/"
        subj_i = self.subj_i

        print(subj_i)
        x_G0  = sio.loadmat(data_location+'%d.X.train.mat'%(subj_i))['X_train']
        x_G0 = x_G0.reshape(x_G0.shape[0],x_G0.shape[1],1)
        
        y_G0  = sio.loadmat(data_location+'%d.Y.train.mat'%(subj_i))['Y_train']
        y_G0 = y_G0.astype(int)
        print(np.shape(y_G0))
        #print(y_G0)
        x_G1  = sio.loadmat(data_location+'%d.X.test.mat'%(subj_i))['X_test']
        x_G1 = x_G1.reshape(x_G1.shape[0],x_G1.shape[1],1)
        y_G1  = sio.loadmat(data_location+'%d.Y.test.mat'%(subj_i))['Y_test']
        y_G1 = y_G1.astype(int)
        return x_G0, y_G0, x_G1, y_G1


    def main(self):
	
        global args
        args = parser.parse_args()
        with open(args.config) as f:
            config = yaml.load(f)
        for key in config:
            for k, v in config[key].items():
                setattr(args, k, v)
        print(args.nodes)
        print(args.graph_model)
	
	    #-------- Loading and Preparing Data--------
        x_G0,y_G0,x_G1,y_G1 = self.load_data()
		
        train_x = torch.from_numpy(x_G0)
        train_x = train_x.permute(0,2,1) # Input should be Instances_Num*Channels_Num*Time_Series_Length
        train_y = torch.from_numpy(y_G0)
        train_y = torch.squeeze(train_y) # Target should be one-dimensional
        train_set = Data.TensorDataset(train_x,train_y)
        train_loader = Data.DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True,num_workers=3) 
		
        test_x = torch.from_numpy(x_G1)
        test_x = test_x.permute(0,2,1) # Input should be Instances_Num*Channels_Num*Time_Series_Length
        test_y = torch.from_numpy(y_G1)
        test_y = torch.squeeze(test_y) # Target should be one-dimensional
        test_set = Data.TensorDataset(test_x,test_y)
        test_loader = Data.DataLoader(dataset=test_set,batch_size=args.batch_size,shuffle=True,num_workers=3)
		

        model = CNN(args,num_classes=6)
        model.cuda()
        cudnn.benchmark = True
        model = model.double()


		#-------- Optimizer and Loss Function --------		
        optimizer = torch.optim.Adam(model.parameters())
        loss_func = torch.nn.CrossEntropyLoss()

		#-------- Training and Tesing --------
        for epoch in range(args.epochs):
            #print('epoch {}'.format(epoch + 1))
            train_loss,train_acc = self.train(train_loader, model, loss_func, optimizer, epoch)
            test_loss,test_acc,val_run_time = self.test(test_loader, model, loss_func, optimizer, epoch)
            if epoch > 20:
                print('Epoch: {0} Loss: {1:.6f}, Acc: {2:.6f}, Val_Loss: {3:.6f}, Val_Acc: {4:.6f}'.format(epoch+1, train_loss,
                       train_acc, test_loss, test_acc))


    def train(self,train_loader, model, loss_func, optimizer, epoch):

        Batch_Time = AverageMeter()
        Loss = AverageMeter()
        Acc = AverageMeter()

		
        model.train() # Switch to train mode
        
        start = time.time()
		
        for i,(batch_x, batch_y) in enumerate(train_loader):
            end = time.time()  
	
		    #-------- Data Preparing --------
            batch_x = Variable(batch_x.cuda())
            batch_x = batch_x.double()
            batch_y = batch_y.cuda(async=True)
            batch_y = Variable(batch_y)
			
		    #-------- Computing Output, Loss and Accuracy --------
            out = model(batch_x)
            #print(torch.max(out,1))
            loss = loss_func(out, batch_y)
            pred = torch.max(out, 1)[1]
            #print(pred)
            #print((pred==batch_y).sum())
            #print(batch_y)
            correct = (pred == batch_y).sum()
            #print(pred)
            acc = float(correct.item())/batch_x.size(0)
			
			#-------- Recording Loss and Accuracy --------
            Loss.update(loss.item())
            Acc.update(acc)
			
			#-------- Backpropagation and Updating Weights --------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Batch_Time.update(time.time()-end)

            if i % args.print_freq == 10000:
                current = time.time()-start
                print('Epoch: [{0}][{1}/{2}]\t'
					  'Time {3:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
					  epoch, i, len(train_loader), current, batch_time=Batch_Time,loss=Loss, acc=Acc))

        return Loss.avg, Acc.avg
			
	   
    def test(self,test_loader, model, loss_func, optimizer, epoch):

        Loss = AverageMeter()
        Acc = AverageMeter()
        n_classes = self.Num_Class
        conf_matrix = torch.zeros(n_classes, n_classes)
        model.eval() # Switch to evaluate mode
        start = time.time()

        for i,(batch_x, batch_y) in enumerate(test_loader):

		    #-------- Data Preparing --------
            batch_x = Variable(batch_x.cuda())
            batch_x = batch_x.double()
            batch_y = batch_y.cuda(async=True)
            batch_y = Variable(batch_y)

			
		    #-------- Computing Output, Loss and Accuracy --------
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            pred = torch.max(out, 1)[1]
            correct = (pred == batch_y).sum()
            #print(correct.item(),batch_x.size(0))
            acc = float(correct.item())/batch_x.size(0)
            self.confusion_matrix( pred, batch_y, conf_matrix)
            #-------- Recording Loss and Accuracy --------
            Loss.update(loss.item())
            Acc.update(acc)

        end = time.time()   
		
        val_time = start-end

			
        return Loss.avg, Acc.avg, val_time

    def confusion_matrix(self, preds, labels, conf_matrix):
        n_classes = self.Num_Class
        
        
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1

        #print(conf_matrix)
        TP = conf_matrix.diag()
        for c in range(n_classes):
            idx = torch.ones(n_classes).byte()
            idx[c] = 0
            TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
            FP = conf_matrix[c, idx].sum()
            FN = conf_matrix[idx, c].sum()

            sensitivity = (TP[c] / (TP[c]+FN))
            specificity = (TN / (TN+FP))

            #print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            #    c, TP[c], TN, FP, FN))
            #print('Sensitivity = {}'.format(sensitivity))
            #print('Specificity = {}'.format(specificity))





class AverageMeter(object):

    #-------- Computing and Storing the Avearge and Current Value --------
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
		
			

#main part running the model.


if __name__ == '__main__':
    for i in [1,2,3,4,5,6,7,8,9,11,12,13,15,18,19,20]:
        index = str(i)
        print("---------------------")
        print("subject "+index)
        print("60-120  for index   "+index)
        RW_CNN = Random_Wired_CNN()
        RW_CNN.data_location = "./ASD/subj"+index+"/60-120/"
        RW_CNN.main()


        print("120-200  for index    "+index)
        RW_CNN = ""
        RW_CNN = Random_Wired_CNN()
        RW_CNN.data_location = "./ASD/subj"+index+"/120-200/"
        RW_CNN.main()


        print("60-200  for index    "+index)
        RW_CNN = ""
        RW_CNN = Random_Wired_CNN()
        RW_CNN.data_location = "./ASD/subj"+index+"/60-200/"
        RW_CNN.main()
