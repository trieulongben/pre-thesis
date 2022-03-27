class history():
    def __init__(self):
        import numpy as np
        self.clientList=[]
        self.train_lossList=np.array([['epoch','id','loss']])
        self.test_lossList={}
        self.train_accuracy=np.array([['epoch','id','loss']])
        self.test_accuracy={}
        form=['epoch','id','loss']
    def add_trainLoss(self,epoch,id,loss):
        self.train_lossList.append([epoch,id,loss])
    def add_testLoss(self,epoch,id,loss):
        self.test_lossList.append([epoch,id,loss])
    def add_trainAccurate(self,epoch,id,acc):
        self.test_lossList.append([epoch,id,acc])
    def add_testAccurate(self,epoch,id,acc):
        self.test_lossList.append([epoch,id,acc])
    def sliceArrayByClientID(self,arr,id):
        return arr[arr[:,0]==id]

    #Array after using sliceArrayByClientID
    def draw(self,arr):
        import matplotlib.pyplot as plt
        arr=arr[:,2]
        plt.plot(arr)

# id:
#server:9999
#client 0-n

