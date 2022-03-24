import multiprocessing as mp
import os
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets,transforms
import random
from torch import optim,ceil
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
import pickle
import numpy as np
import dataGenerator as dataGenerator
import torchvision.models as models

class Client():
    def __init__(self,id=9999,dataClass=0,optimizer=None,initModel=models.mobilenet_v2(pretrained=True)):
        self.id=id
        self.dataClass=dataClass
        self.optimizer=optimizer
        self.curModel=initModel
        #Get dataset:
        self.dataset=dataGenerator.plantdisease(isTest=False,rank=self.dataClass)
    def clientID(self):
        print("ID of client: {}".format(self.id))
    def processID(self):
        print("ID of process: {}".format(os.getpid()))




    "Update model"
    def update_Model(self,model,data=None):
        self.curModel=model

        #Change data class
        if data:
            self.dataClass=data

    def client_train(self,num_batch=1,lr=0.00001,epoch=10,optimizer=None):
        
        model = self.curModel
        self.optimizer =optim.SGD(model.parameters(),lr=0.0001)
        ####

        dataLoader=torch.utils.data.DataLoader(self.dataset,
                                         batch_size=int(num_batch),
                                         shuffle=True)

        ######
        for epoc in range(epoch):
            epoc_loss = 0.0
            for data, target in dataLoader:
            #
            #data = data.repeat(1, 3, 1, 1)
            #
                self.optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                epoc_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            
        print('Client ID ', self.clientID, ', epoch ',
              epoch, ': ', epoc_loss / num_batch)
    def client_evaluate(self):
        self.model.eval()
        testDataset=dataGenerator.plantdisease(isTest=True,rank=self.dataClass)

        testDataLoader=torch.utils.data.DataLoader(testDataset,
                                         batch_size=int(num_batch),
                                         shuffle=True)
        test_loss,correct=0,0
        with torch.no_grad():
            for data,labels in testDataLoader:
                outputs=self.model(data)
                test_loss+=F.nll_loss(outputs, labels).item()
                predicted=outputs.argmax(dim=1,keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
        test_loss = test_loss / len(testDataLoader)
        test_accuracy=correct/len(testDataset)
        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True)
a=Client()
a.client_train()