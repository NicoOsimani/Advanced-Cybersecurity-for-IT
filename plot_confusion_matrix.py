import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# todo set params
out_path = "/home/nico/Desktop/Sicurezza_2/Progetto"
test_name = "dga_domains"
class_names = ["legit", "dga"]
cm = np.array([[0.0, 0.0], [0.0, 0.0]])

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig = plt.figure(figsize=(4, 4), dpi=80)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)
    print("")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.savefig(out_path + "/LSTM-MI_" + test_name + "_cm_averaged.png")
    plt.show()
    plt.close()

def main():
    plot_confusion_matrix(cm, class_names)
    print("Confusion matrix plotted")

if __name__ == "__main__":
    main()
