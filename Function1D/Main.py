import torch
from Function1D.CustomClass import CustomClass
from Function1D.CustomDataSet import CustomDataSet
from sklearn.model_selection import train_test_split
from Function1D.Train import train_model
import matplotlib.pyplot as plt


def target_function(x):
    return 2 ** x * torch.sin(2 ** -x)

if __name__=="__main__":
    HiddenNeurons=100
    Model = CustomClass(HiddenNeurons,1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=3e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    lossMSE = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss(reduction='mean')
    batch_size = 200
    X = torch.linspace(-10,5,200)
    num_epochs = 100
    #Y = np.zeros(X.shape) #target_function(X)
    Y = target_function(X)

    X.unsqueeze_(1)
    Y.unsqueeze_(1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=60)
    test_DataLoader=torch.utils.data.DataLoader(CustomDataSet(x_test, y_test),batch_size=batch_size,shuffle=False)
    train_DataLoader = torch.utils.data.DataLoader(CustomDataSet(x_train, y_train), batch_size=batch_size, shuffle=False)

    train_model(Model,lossMSE,optimizer,scheduler,num_epochs,train_DataLoader,test_DataLoader,device)
    print("Loss L1: {}".format(lossL1(Model(X),Y)))

    X = torch.linspace(-20, 20, 1000)
    Y = target_function(X)
    x_flat=X.numpy()


    plt.plot(x_flat, torch.flatten(Model(X.unsqueeze(1))).data.numpy(),'x', c='r', label="Prediction")
    plt.plot(x_flat,torch.flatten(Y).numpy(), '-', c='g', label='Actual')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(" 2 ** x * torch.sin(2 ** -x) ")
    plt.show()


