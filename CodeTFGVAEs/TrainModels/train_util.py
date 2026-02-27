
import time, datetime
from matplotlib import pyplot as plt
import numpy as np

def cuda2numpy(loss_val):
    
    loss_val=list(loss_val)
    for k in np.arange(len(loss_val)):
         loss_val[k]=loss_val[k].cpu().detach().numpy() 
    
    return np.array(loss_val)

def display_elapsed_time(time_elapsed, msg =None):
    if msg is None:
        msg = 'Training time'
    secs = time_elapsed % 60
    h_t = (time_elapsed - secs) / 3600
    hours = (time_elapsed - secs) // 3600
    mins = (h_t - hours) * 60
    current_time = datetime.datetime.now().time().strftime("%H:%M:%S")
    print(msg + f' : {hours:.0f}h {mins:.0f}m {secs:.0f}s finished at ' + current_time)


def plot_metrics(avg_cost, msg= None):

    if msg is None:
        msg = ''

    train_loss = avg_cost[:,0]
    valid_loss = avg_cost[:,3]

    train_acc = avg_cost[:,1]
    valid_acc = avg_cost[:,4]

    plt.plot(train_loss, label='training loss')
    plt.plot(valid_loss, label='validation loss')
    plt.legend()
    plt.xlabel(r'epochs')
    plt.title(msg + ' Training Loss ')
    plt.show()

    plt.plot(train_acc, label='training')
    plt.plot(valid_acc, label='validation')
    plt.legend()
    plt.xlabel(r'epochs')
    plt.title(msg + ' Accuracy ' )
    plt.show()
