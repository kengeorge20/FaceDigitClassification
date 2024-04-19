# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import zipfile
import os

## Constants
DATUM_WIDTH = 0 # in pixels
DATUM_HEIGHT = 0 # in pixels

## Module Classes

class Datum:
    """
    A datum is a pixel-level encoding of digits or face/non-face edge maps.

    Digits are from the MNIST dataset and face images are from the 
    easy-faces and background categories of the Caltech 101 dataset.
    
    Each digit is 28x28 pixels, and each face/non-face image is 60x74 
    pixels, each pixel can take the following values:
      0: no edge (blank)
      1: gray pixel (+) [used for digits only]
      2: edge [for face] or black pixel [for digit] (#)
      
    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:
    
      For example, the + in the above diagram is stored in pixels[2][3], or
      more generally pixels[column][row].
      
    The contents of the representation can be accessed directly
    via the getPixel and getPixels methods.
    """
    def __init__(self, data, width, height):
        DATUM_HEIGHT = height
        DATUM_WIDTH = width
        self.height = DATUM_HEIGHT
        self.width = DATUM_WIDTH
        if data == None:
            data = [[' ' for i in range(DATUM_WIDTH)] for j in range(DATUM_HEIGHT)]
        self.pixels = data

    def getPixel(self, column, row):
        return self.pixels[column][row]

    def getPixels(self):
        return self.pixels

def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    (Return less then n items if the end of file is encountered).
    """
    fin = readlines(filename)
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(map(convertToInteger, fin.pop(0))))
        if len(data[0]) < width-1:
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(Datum(data, width, height))
    return items

def readlines(filename):
    "Opens a file or reads it from the zip archive data.zip"
    if(os.path.exists(filename)):
        with open(filename, 'r') as f:
            return [l.strip() for l in f.readlines()]
    else:
        with zipfile.ZipFile('data.zip') as z:
            return z.read(filename).decode('utf-8').split('\n')

def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels

def asciiGrayscaleConversionFunction(value):
    """
    Helper function for display purposes.
    """
    if(value == 0):
        return ' '
    elif(value == 1):
        return '+'
    elif(value == 2):
        return '#'

def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if(character == ' '):
        return 0
    elif(character == '+'):
        return 1
    elif(character == '#'):
        return 2

def convertToInteger(data):
    """
    Helper function for file reading.
    """
    if isinstance(data, list):
        return list(map(convertToInteger, data))
    else:
        return IntegerConversionFunction(data)

# Testing

def _test():
    import doctest
    doctest.testmod() # Test the interactive sessions in function comments
    n = 1
    items = loadDataFile("digitdata/trainingimages", n,28,28)
    labels = loadLabelsFile("digitdata/traininglabels", n)
    for i in range(1):
        print(items[i])
        print(items[i].height)
        print(items[i].width)
        print(dir(items[i]))
        print(items[i].getPixels())

if __name__ == "__main__":
    _test()
