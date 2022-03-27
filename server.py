from dataclasses import dataclass
import multiprocessing as mp
import client
import torchvision.models as models
import dataGenerator
import torch
class server():
    def __init__(self):
        self.list_Client=[]
        self.centralModel=models.mobilenet_v2(pretrained=True)
        self.centralDataset=dataGenerator.plantdisease(isTest=True,rank=None)
    def __len__(self):
        return len(self.list_Client)


    def generateClient(self,n_client):
        for i in range(0,n_client):
            #
            tempModel=client.Client(id=i,dataClass=i,initModel=self.centralModel)
            self.list_Client.append(tempModel)
    def updateClientModel(self):
        params=self.collectModel()
        paraAverage=self.averageModel(params)
        self.centralModel.load_state_dict(paraAverage)
        for client in self.list_Client:
            client.curModel.load_state_dict(paraAverage)
        print('All clients updated')
    def collectModel(self):
        model_params=[]
        for client in self.list_Client:
            model_params.append(client.curModel.state_dict())
        return model_params
    def averageModel(self,params):
        paraOut={}
        for key in params[0]:
            paraOut[key] = sum([params[i][key] for i in range(0,len(params)-1)]) / len(params)-1
        return paraOut
    def evaluate_CentralModel(self):
        model=self.centralModel
        model.eval()

        testDataLoader=torch.utils.data.DataLoader(self.centralDataset,
                                         batch_size=int(num_batch),
                                         shuffle=True)
        test_loss,correct=0,0
        with torch.no_grad():
            for data,labels in testDataLoader:
                outputs=self.centralModel(data)
                test_loss+=F.nll_loss(outputs, labels).item()
                predicted=outputs.argmax(dim=1,keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
        test_loss = test_loss / len(testDataLoader)
        test_accuracy=correct/len(testDataset)
        message = f"\t[Central Model {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True)
    def trainPerClient(self,clientID):
        self.list_Client[clientID].client_train(epoch=5)
    def evaluatePerClient(self,clientID):
        self.list_Client[clientID].client_evaluate()
    def trainClients(self):
        with mp.Pool(processes=len(self.list_Client)) as pool:
            
            pool.map(self.trainPerClient,[i for i in range(0,len(self.list_Client))])
            pool.map(self.evaluatePerClient,[i for i in range(0,len(self.list_Client))])
            self.updateClientModel()
            self.evaluate_CentralModel()
def saveServer(self,server):
    import pickle
    with open('server.pickle', 'wb') as f:
        pickle.dump(server, f)
if __name__ == '__main__':
    a=server()
    a.generateClient(4)
    a.trainClients()
    saveServer(a)
