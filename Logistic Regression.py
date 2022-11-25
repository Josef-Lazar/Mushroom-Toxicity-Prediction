import pandas as pd
import numpy as np
import sklearn.metrics


def read_data(filename):
    #reads in a csv and sements the data
    #randomizes the order of the data, then splits it into different sets
    #returns separate inputs (x) and outputs (y) for each of training, test, and validation
    #also returns a list of column names, which may be useful for determining heavily weighted features
    df = pd.read_csv(filename)
    df.insert(1, "x_0", 1)
    #print(df)
    data = df.to_numpy()
    np.random.shuffle(data)
    test_size = int(data.shape[0]/10)
    data_test = data[:test_size]
    data_val = data[test_size:2*test_size]
    data_train = data[2*test_size:]
    x_train = data_train[:,1:] #mushroom features
    y_train = data_train[:,0] #poisonous or not
    x_val = data_val[:,1:]
    y_val = data_val[:,0]
    x_test = data_test[:,1:]
    y_test = data_test[:,0]
    return x_train, y_train, x_val, y_val, x_test, y_test, df.columns.values

data = read_data("mushrooms_logistic.csv")
weights = [0] * len(data[0][0])

def add_ones(x):
    #takes an array of feature vectors and adds a column of 1s to the start
    #useful for logistic regression, since x_0 is always 1
    return np.insert(x, 0, np.ones(x.shape[0]), axis = 1)

def compute_perceptron_error(x,y, weights, bias):
    #takes in a matrix of feature vectors x and a vector of class labels y
    #also takes a vector weights and a scalar bias for the classifier
    #returns the error on the data (x, y) of the perceptron classifier
    y_pred = np.sign(x.dot(weights) + bias)
    accuracy = sklearn.metrics.accuracy_score(y_pred, y)
    return 1 - accuracy

def compute_hypothesis(x, weights):
    #computes the hypothesis function for logistic function given data x and weights
    #if x is a single feature vector, will return a scalar
    #if x is a matrix of feature vectors, will return a vector containing the hypothesis for each row
    prod = x.dot(weights)
    return 1/(1 + np.exp(-prod))

def compute_logistic_error(x,y, weights):
    #takes in a matrix of feature vectors x, a vector of class labels y
    #also takes a vector weights for the classifier
    #returns the error on the data (x, y) of the logistic regression classifier
    y_pred = (compute_hypothesis(x, weights) > 0.5).astype(float)
    print(x[0].dot(weights))
    print(y_pred[0])
    print(y[0])
    accuracy = sklearn.metrics.accuracy_score(y_pred, y)
    return 1 - accuracy

def rank_features(weights, feats):
    #takes in a weight vector and an array of feature names
    #returns a sorted array of features, sorted from most negatively weighted to most positively weighted
    #note that feats MUST be a numpy array of the same length as weights
    #if feats[i] does not correspond to weights[i], this will not return accurate results
    imp = np.argsort(weights)
    return feats[imp]

def validation_error(x, y):
    edible_correct = 0 #edible mushrooms that were correctly labled as edible
    edible_incorrect = 0 #edible mushrooms that were incorrectly labled as poisonous
    poisonous_correct = 0 #poisonous mushrooms that were correctly labled as poisonous
    poisonous_incorrect = 0 #poisonous mushrooms that were incorrectly labled as edible
    for i in range(len(y)):
        if y[i] == 1:
            if hypothesis(x[i]) > 0.5:
                poisonous_correct += 1
            else:
                poisonous_incorrect += 1
        if y[i] == 0:
            if hypothesis(x[i]) < 0.5:
                edible_correct += 1
            else:
                edible_incorrect += 1
    correct = edible_correct + poisonous_correct
    incorrect = edible_incorrect + poisonous_incorrect
    return (incorrect) / (incorrect + correct)

def model_accuracy():
    data_sets = ["training data set:", "validation data set:", "test data set:"]
    for i in range(len(data_sets)):
        edible_correct = 0 #edible mushrooms that were correctly labled as edible
        edible_incorrect = 0 #edible mushrooms that were incorrectly labled as poisonous
        poisonous_correct = 0 #poisonous mushrooms that were correctly labled as poisonous
        poisonous_incorrect = 0 #poisonous mushrooms that were incorrectly labled as edible
        x = data[2 * i]
        y = data[1 + 2 * i]
        for j in range(len(y)):
            if y[j] == 1:
                if hypothesis(x[j]) > 0.5:
                    poisonous_correct += 1
                else:
                    poisonous_incorrect += 1
            if y[j] == 0:
                if hypothesis(x[j]) < 0.5:
                    edible_correct += 1
                else:
                    edible_incorrect += 1
        print(data_sets[i])
        print("edible correct: " + str(edible_correct))
        print("edible incorrect: " + str(edible_incorrect))
        print("poisonous correct: " + str(poisonous_correct))
        print("poisonous incorrect: " + str(poisonous_incorrect))
        correct = edible_correct + poisonous_correct
        incorrect = edible_incorrect + poisonous_incorrect
        accuracy = correct / (correct + incorrect)
        print("accuracy: " + str(accuracy))
        print()

def precision(poisonous_correct, edible_incorrect):
    return poisonous_correct / (poisonous_correct + edible_incorrect)

def recall(poisonous_correct, poisonous_incorrect):
    return poisonous_correct / (poisonous_correct + poisonous_incorrect)

#logistic regression

def hypothesis(x):
    hypothesis = 1 / (1 + np.exp(- x.dot(weights)))
    return hypothesis

def stoch_grad_asc(x, y, learning_rate):
    global weights
    old_weights = weights
    old_validation_error = 2
    new_validation_error = validation_error(data[2], data[3])
    iterations = 0
    while old_validation_error > new_validation_error:
        old_weights = weights
        for i in range(len(y)):
            for j in range(len(x[0])):
                #print("x[i][j]: " + str(x[i][j]))
                derivative = (y[i] - hypothesis(x[i])) * x[i][j]
                #print("derivative: " + str(derivative))
                weights[j] = weights[j] + learning_rate * derivative
        old_validation_error = new_validation_error
        new_validation_error = validation_error(data[2], data[3])
        iterations += 1
        print("iterations: " + str(iterations) + ", validation error: " + str(new_validation_error))
    weights = old_weights
    return weights



