from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import os
import hashlib
import numpy as np
from scipy.misc import imresize
import scipy
from PIL import Image
import urllib.request

def timeout(func, args=(), kwargs={}, timeout_duration=2, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[: ,: ,0], rgb[: ,: ,1], rgb[: ,: ,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/ 255.

testfile = urllib.request.URLopener()


################################################################################

def download_images():
    act = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
    for a in act:
        name = a
        i = 0
        for line in open("facescrub_actors.txt"):
            if a in line:
                filename = name + str(i) + '.jpg'
                timeout(testfile.retrieve, (line.split()[4], "actors/" + filename), {}, 30)
                if not os.path.isfile("actors/" + filename):
                    continue

                parameters = (line.split('\t', 5))[4].split(',')
                try:
                    m = hashlib.sha256()
                    m.update(open("actors/" + filename, 'rb').read())
                    sha = m.hexdigest()
                    if sha != line.split()[6]:
                        os.remove("actors/" + filename)
                        continue
                    im = Image.open("actors/" + filename)

                    im = im.crop(
                        (float(parameters[0]), float(parameters[1]), float(parameters[2]), float(parameters[3])))
                    im = np.asarray(im)
                    im227 = imresize(im, (227, 227))
                    if im227.shape != (227, 227, 3):
                        os.remove("actors/" + filename)
                        continue
                    im32 = rgb2gray(im)
                    im32 = imresize(im32, (32, 32))
                except:
                    os.remove("actors/" + filename)
                    continue

                try:
                    scipy.misc.toimage(im227, cmin=0.0, cmax=...).save("actors/" + filename)
                    scipy.misc.toimage(im32, cmin=0.0, cmax=...).save("cropped_actors/" + filename)
                except:
                    continue

                print(filename)
                i += 1

        for line in open("facescrub_actresses.txt"):
            if a in line:
                filename = name + str(i) + '.jpg'
                timeout(testfile.retrieve, (line.split()[4], "actresses/" + filename), {}, 30)
                if not os.path.isfile("actresses/" + filename):
                    continue

                parameters = (line.split('\t', 5))[4].split(',')
                try:
                    m = hashlib.sha256()
                    m.update(open("actresses/" + filename, 'rb').read())
                    sha = m.hexdigest()
                    if sha != line.split()[6]:
                        os.remove("actresses/" + filename)
                        continue
                    im = Image.open("actresses/" + filename)

                    im = im.crop(
                        (float(parameters[0]), float(parameters[1]), float(parameters[2]), float(parameters[3])))
                    im = np.asarray(im)
                    im227 = imresize(im, (227, 227))
                    if im227.shape != (227, 227, 3):
                        os.remove("actresses/" + filename)
                        continue
                    im32 = rgb2gray(im)
                    im32 = imresize(im32, (32, 32))
                except:
                    os.remove("actresses/" + filename)
                    continue

                try:
                    scipy.misc.toimage(im227, cmin=0.0, cmax=...).save("actresses/" + filename)
                    scipy.misc.toimage(im32, cmin=0.0, cmax=...).save("cropped_actresses/" + filename)
                except:
                    continue

                print(filename)
                i += 1
                
    # manual edit, drescher83 messes up, remove and rename the last image to 83
    os.chdir("actresses")
    os.remove("drescher83.jpg")
    for filename in os.listdir(os.getcwd()):
        if "drescher15" in filename:
            last = filename
    os.rename(last, "drescher83.jpg")
    os.chdir("..")
    os.chdir("cropped_actresses")
    os.remove("drescher83.jpg")
    for filename in os.listdir(os.getcwd()):
        if "drescher15" in filename:
            last = filename
    os.rename(last, "drescher83.jpg")
    os.chdir("..")




