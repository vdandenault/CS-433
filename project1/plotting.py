from matplotlib.pylab import plt
import numpy as np

def plot_bar_from_counter(values, title):
    """"
    This function creates a bar plot from a counter.

    :param counter: This is a counter object, a dictionary with the item as the key
     and the frequency as the value
    :param ax: an axis of matplotlib
    :return: the axis wit the object in it
    """
    labels, counts = np.unique(values, return_counts=True)
    ticks = range(len(counts))
    plt.bar(ticks,counts, align='center')
    plt.title(title)
    plt.xticks(ticks, labels)
    plt.show()

def plot_loss_function(train_loss, nb_epochs): 
    epochs = range(1, nb_epochs+1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()