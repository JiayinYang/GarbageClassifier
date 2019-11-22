# A Deep Learning Approach to Sustainable Waste Management
# Author: Jiayin Yang

from PIL import Image
import numpy as np
import glob
import keras
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from random import shuffle
from classifier import classifier

# Plotting the Precision-Recall Curve and the ROC Curve
def plott(precision, recall, falsepos):
    index = np.argsort(precision)
    precision = np.sort(precision)
    recall = np.asarray(recall)
    recallsort = recall[index]

    plt.figure()
    print('auc')
    print(auc(precision, recallsort))
    print(recallsort)
    print(precision)
    plt.plot(recallsort, precision)
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    index = np.argsort(falsepos)
    falsepos = np.sort(falsepos)
    recall = np.asarray(recall)
    recall = recall[index]
    plt.figure()
    print('roc_auc')
    print(auc(falsepos, recall))

    plt.plot(falsepos, recall)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

# Reading Data from the file
def readData():
    image_list = []
    myclass = []
    for filename in glob.glob('dataset-resized/glass/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        myclass.append(0)
        image_list.append(im)
    for filename in glob.glob('dataset-resized/cardboard/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        image_list.append(im)
        myclass.append(1)
    for filename in glob.glob('dataset-resized/metal/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        image_list.append(im)
        myclass.append(2)
    for filename in glob.glob('dataset-resized/paper/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        image_list.append(im)
        myclass.append(3)
    
    myclass = keras.utils.to_categorical(np.asarray(myclass), num_classes=4)
    c = list(zip(image_list, myclass))
    shuffle(c)
    image_list, myclass = zip(*c)
    return image_list, myclass

[x,y] = readData()
x = np.asarray(x)
y = np.asarray(y)
precision = np.zeros([4,10])
recall = np.zeros([4,10])
rocfp = np.zeros([4,10])
falsepos = np.zeros([4,10])
# Ten folds Cross Validation
cv = KFold(n_splits = 10, shuffle = True, random_state = 100)
z = 0
for index_train, index_test in cv.split(x):
    trainsize = index_train.size
    testsize = index_test.size
    train_dataset = x[index_train]
    test_dataset = x[index_test]
    train_labels = y[index_train] 
    test_labels = y[index_test]
    precision, recall, falsepos = classifier(z, precision, recall, falsepos, train_dataset, test_dataset, train_labels, test_labels, trainsize, testsize)
    z = z + 1

# Plotting the Precision-Recall Curve and the ROC Curve for each class
print('class 1')
plott(precision[0], recall[0], falsepos[0])
print('class 2')
plott(precision[1], recall[1], falsepos[1])
print('class 3')
plott(precision[2], recall[2], falsepos[2])
print('class 4')
plott(precision[3], recall[3], falsepos[3])



    