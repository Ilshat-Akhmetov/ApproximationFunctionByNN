from Function2D.NN_Training_Parameters import *
from Function2D.CustomDataSet import CustomDataSet
from sklearn.model_selection import train_test_split
from Function2D.Train import train_model
import matplotlib.pyplot as plt
import numpy as np
import subprocess


def ClearGraphFolderAndPlotNewOne():

    bashCommandSecond = ' tensorboard --logdir Graph'
    process = subprocess.Popen(bashCommandSecond.split(), stdout=subprocess.PIPE)
    process.communicate()  # run bash script
    bashCommandFirst = 'rm -r Graph/'
    process = subprocess.Popen(bashCommandFirst.split(), stdout=subprocess.PIPE)
    process.communicate()  # run bash script


def target_function(x,y):
    return x**2+y*x+x*torch.sin(y)

def function_plot(X,Y,Z,Z_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel(Z_label)
    plt.show()

def GetData():
    # setting meshgrid for train and test
    X = torch.linspace(-5, 5, NumberOfX)
    Y = torch.linspace(-10, 10, NumberOfY)
    Xm, Ym = torch.meshgrid([X, Y])
    # get value of the approximated function
    Function_value = target_function(Xm, Ym)
    return Function_value, Xm, Ym

def GetFunctionByNN(Function, Xmesh,Ymesh):
    # prepare the data in order to be processable by neural model
    Xm_flatted = torch.flatten(Xmesh)
    Ym_flatted = torch.flatten(Ymesh)
    DomainArray = torch.stack((Xm_flatted, Ym_flatted), dim=1)

    #split data into train and test subsets
    Domain_train, Domain_test, Function_train, Function_test = train_test_split(DomainArray, torch.flatten(Function),
                                                                                test_size=0.2)
    test_DataLoader = torch.utils.data.DataLoader(CustomDataSet(Domain_test, Function_test), batch_size=batch_size,
                                                  shuffle=False)
    train_DataLoader = torch.utils.data.DataLoader(CustomDataSet(Domain_train, Function_train), batch_size=batch_size,
                                                   shuffle=False)

    # run training
    train_model(Model, lossMSE, optimizer, scheduler, num_epochs, train_DataLoader, test_DataLoader, device)

    # get value of function in each point of the domain area
    Function_By_NN = Model(DomainArray)

    return Function_By_NN

def GetMSEofApproximation(Function_By_NN,Original_Data):
    return lossMSE(torch.flatten(Function_By_NN),torch.flatten(Original_Data))

def GetSTDofApproximation(Function_By_NN, Original_Data):
    return np.std(Function_By_NN - Original_Data)

def GetDomainAndFunctions():
    Original_Function, Xmesh, Ymesh = GetData()
    # split into test and train datasets

    # get function approximated by NN
    Function_By_NN = GetFunctionByNN(Original_Function, Xmesh, Ymesh)
    # MSE loss between Function and its approximation

    return Xmesh, Ymesh, Original_Function, Function_By_NN

def main_script():

    Xmesh, Ymesh, Original_Function, Function_By_NN = GetDomainAndFunctions()
    print("Loss L2: {}".format(GetMSEofApproximation(Function_By_NN, Original_Function)))

    # Convert everything to numpy so its data can be plotted
    Function_By_NN = Function_By_NN.reshape(NumberOfY, NumberOfX).detach()
    Function_By_NN = Function_By_NN.numpy()
    Original_Function = Original_Function.numpy()
    Xmesh = Xmesh.numpy()
    Ymesh = Ymesh.numpy()
    # standart error of difference of the function and its neural approximation
    print("Function - NN std: {}".format(GetSTDofApproximation(Function_By_NN,Original_Function)))

    # plotting final results

    # plot graph of the true function
    function_plot(Xmesh, Ymesh, Original_Function, "True Function")

    # plot graph calculated by NN
    function_plot(Xmesh, Ymesh, Function_By_NN, "Approximated by NeuralNet")


if __name__=="__main__":
    main_script()
    #ClearGraphFolderAndPlotNewOne()

