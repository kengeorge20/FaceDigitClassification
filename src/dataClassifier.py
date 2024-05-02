# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import perceptron
import samples
import sys
import util
import neuralNet

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70

def basicFeatureExtractorDigit(datum):
    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            features[(x, y)] = datum.getPixel(x, y) > 0
    return features

def calculate_symmetry(datum, width, height):
    horizontal_symmetry = sum(1 for x in range(width) for y in range(height) if datum.getPixel(x, y) == datum.getPixel(width - x - 1, y))
    vertical_symmetry = sum(1 for x in range(width) for y in range(height) if datum.getPixel(x, y) == datum.getPixel(x, height - y - 1))
    return horizontal_symmetry / (width * height), vertical_symmetry / (width * height)

def count_white_regions(datum, width, height):
    visited = set()
    regions = 0

    def dfs(x, y):
        if (x, y) in visited or not datum.getPixel(x, y):
            return
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and datum.getPixel(nx, ny):
                    stack.append((nx, ny))

    for x in range(width):
        for y in range(height):
            if datum.getPixel(x, y) and (x, y) not in visited:
                dfs(x, y)
                regions += 1

    return regions

def enhancedFeatureExtractorDigit(datum):
    """
    Enhanced feature extraction for digits.
    """
    features = basicFeatureExtractorDigit(datum)
    width, height = DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT

    # Symmetry features
    h_symmetry, v_symmetry = calculate_symmetry(datum, width, height)
    features[('horizontal_symmetry')] = h_symmetry
    features[('vertical_symmetry')] = v_symmetry

    # White region count
    num_white_regions = count_white_regions(datum, width, height)
    features[('num_white_regions')] = num_white_regions

    return features

def basicFeatureExtractorFace(datum):
    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            features[(x, y)] = datum.getPixel(x, y) > 0
    return features

def enhancedFeatureExtractorFace(datum):
    """
    Enhanced feature extraction for faces.
    """
    features = basicFeatureExtractorFace(datum)
    width, height = FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT

    # Define regions for eyes and mouth
    # Assuming faces are aligned and centered
    eye_region = [(x, y) for x in range(width // 4, 3 * width // 4) for y in range(height // 4, height // 2)]
    mouth_region = [(x, y) for x in range(width // 4, 3 * width // 4) for y in range(3 * height // 4, height)]

    # Calculate the density of 'on' pixels in these regions
    eye_density = sum(datum.getPixel(x, y) for x, y in eye_region) / len(eye_region)
    mouth_density = sum(datum.getPixel(x, y) for x, y in mouth_region) / len(mouth_region)

    features[('eye_density')] = eye_density
    features[('mouth_density')] = mouth_density
    return features

def readCommand(argv):
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Run classifiers on digit and face data using various extraction methods.')

    parser.add_argument('-c', '--classifier', help='The type of classifier [Default: %(default)s]', choices=['perceptron', 'neuralNet'], default='perceptron')
    parser.add_argument('-d', '--data', help='Dataset to use [Default: %(default)s]', choices=['digits', 'faces'], default='digits')
    parser.add_argument('-t', '--training', help='The size of the training set [Default: %(default)s]', default=100, type=int)
    parser.add_argument('-e', '--enhanced', help='Use enhanced features [Default: %(default)s]', action='store_true', default=False)

    options = parser.parse_args(argv)
    args = {}

    print("Doing classification")
    print("--------------------")
    print("Data:\t\t" + options.data)
    print("Classifier:\t\t" + options.classifier)
    print("Training set size:\t" + str(options.training))
    print("Using enhanced features:\t" + str(options.enhanced))

    if options.data == "digits":
        featureFunction = enhancedFeatureExtractorDigit if options.enhanced else basicFeatureExtractorDigit
    elif options.data == "faces":
        featureFunction = enhancedFeatureExtractorFace if options.enhanced else basicFeatureExtractorFace
    else:
        print("Unknown dataset", options.data)
        sys.exit(2)

    if options.data == "digits":
        legalLabels = list(range(10))
    else:
        legalLabels = list(range(2))

    if options.training <= 0:
        print("Training set size should be a positive integer (you provided: %d)" % options.training)
        sys.exit(2)

    if options.classifier == "perceptron":
        classifier = perceptron.PerceptronClassifier(legalLabels, 3)  # Assume default 3 iterations
    elif options.classifier == "neuralNet":
        classifier = neuralNet.NeuralNetworkClassifier(legalLabels)
    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = lambda image: image  # Replace this with actual function to print image if needed

    return args, options

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    print("Analysis of the results")
    
    for i in range(len(guesses)):
        predicted = guesses[i]
        truth = testLabels[i]

        if predicted == truth:
            print("===================================")
            print(f"Correctly classified example #{i}")
            print(f"Predicted: {predicted}; Truth: {truth}")
            printImage(rawTestData[i].getPixels())
        else:
            print("===================================")
            print(f"Misclassified example #{i}")
            print(f"Predicted: {predicted}; Truth: {truth}")
            printImage(rawTestData[i].getPixels())

    # Example: Calculate and print overall accuracy
    accuracy = float(sum(1 for i in range(len(guesses)) if guesses[i] == testLabels[i])) / len(guesses)
    print("Overall accuracy: %.2f%%" % (accuracy * 100))

def runClassifier(args, options):
    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data  
    numTraining = options.training
    numTest = TEST_SET_SIZE

    if(options.data == "faces"):
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)

    # Extract features
    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    testData = list(map(featureFunction, rawTestData))

    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, None, None)
    print("Testing...")
    guesses = classifier.classify(testData)
    correct = sum(guesses[i] == testLabels[i] for i in range(len(testLabels)))
    print("%d correct out of %d (%.1f%%)." % (correct, len(testLabels), 100.0 * correct / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

if __name__ == '__main__':
    args, options = readCommand(sys.argv[1:])  # Get game components based on input
    runClassifier(args, options)
