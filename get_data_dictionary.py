import numpy as np
from pylab import *
from PIL import Image


def get_data_dictionary():
    scrubs = ['baldwin', 'hader', 'carell']
    scrubesses = ['drescher', 'ferrera', 'chenoweth']
    data = {}
    for name in scrubs:
        training = []
        for i in range(90):
            im = Image.open("cropped_actors/" + name + str(i) + ".jpg")
            im = np.array(im)
            im = im.flatten()
            training.append(im)
        label = "train" + name
        training = np.array(training)
        data[label] = training

        validation = []
        for i in range(90, 100):
            im = Image.open("cropped_actors/" + name + str(i) + ".jpg")
            im = np.array(im)
            im = im.flatten()
            validation.append(im)
        validation = np.array(validation)
        label = "valid" + name
        data[label] = validation
        
        test = []
        for i in range(100, 130):
            im = Image.open("cropped_actors/" + name + str(i) + ".jpg")
            im = np.array(im)
            im = im.flatten()
            test.append(im)
        test = np.array(test)
        label = "test" + name
        data[label] = test

    for name in scrubesses:
        training = []
        for i in range(90):
            im = Image.open("cropped_actresses/" + name + str(i) + ".jpg")
            im = np.array(im)
            im = im.flatten()
            training.append(im)
        training = np.array(training)
        label = "train" + name
        data[label] = training

        validation = []
        for i in range(90, 100):
            im = Image.open("cropped_actresses/" + name + str(i) + ".jpg")
            im = np.array(im)
            im = im.flatten()
            validation.append(im)
        validation = np.array(validation)
        label = "valid" + name
        data[label] = validation
        
        test = []
        for i in range(100, 130):
            im = Image.open("cropped_actresses/" + name + str(i) + ".jpg")
            im = np.array(im)
            im = im.flatten()
            test.append(im)
        test = np.array(test)
        label = "test" + name
        data[label] = test

    return data


def check():
    '''
    Checks if the output data is in the correct shape to be parsed
    Runs get_data and asserts that the shapes of all the training sets
    and test sets are the same
    '''
    M = get_data_dictionary()
    
    x = set([M["testbaldwin"].shape, M["testhader"].shape, M["testcarell"].shape, M["testdrescher"].shape, M["testferrera"].shape, M["testchenoweth"].shape])
    
    y = set([M["trainbaldwin"].shape, M["trainhader"].shape, M["traincarell"].shape, M["traindrescher"].shape, M["trainferrera"].shape, M["trainchenoweth"].shape])
    
    assert (len(x) == 1)
    assert (len(y) == 1)
    print("All Good!")
    
    
# check()
