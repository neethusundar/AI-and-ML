import csv
import os
import math
import operator
import matplotlib.pyplot as plt
from sys import argv
from matplotlib.backends.backend_pdf import PdfPages

class Environment:
    def __init__(self,directory,k,folder_name,accuracy):
        self.directory = directory
        self.k = k
        self.folder_name = folder_name
        self.accuracy = accuracy

    #method to send data and percepts to agent
    def sense(self,folder_name, k):
        predictions = []
        average_accuracy = 0.00
        agent = Agent()
        file = []
        print ('K = {}'.format(k))
        #iterate to read all 10 training and test datasets
        #path = os.getcwd()+"\\"+folder_name
       #for file in os.listdir(path):
        for j in range(1, 11):
            #open training set
            tra_file = self.directory + os.sep + "{}-10-{}tra.dat".format(self.folder_name,j)
            #read training set as csv file
            with open(tra_file, 'rb') as csvfile:
                lines = csv.reader(csvfile)
                train_set = list(lines)
                #truncate metadata in training set
                train_set = [item for item in train_set if '@' not in item[0]]
            #open test set
            tst_file = self.directory + os.sep + "{}-10-{}tst.dat".format(self.folder_name,j)
            #read test set as csv file
            with open(tst_file, 'rb') as csvfile:
                lines = csv.reader(csvfile)
                test_set = list(lines)
                #truncate metadata in test set
                test_set = [item for item in test_set if '@' not in item[0]]
            #send all test set percepts to agent
            for i in range(len(test_set)):
                predictions.append(agent.predict(train_set, test_set[i],k))
            #compute induvidual test set accuracy and average accuracy by k value
            accuracy = agent.measureAccuracy(test_set, predictions)
            average_accuracy += accuracy/10
            print('     Accuracy for {} is {}%'.format(tst_file,repr(round(accuracy,2))))
        print('Average accuracy {}%'.format(repr(round(average_accuracy,2))))
        print("\n")
        return average_accuracy

class Agent:
    def __init__(self):
        pass

    #method to find neighbours and predictons
    def predict(self, train_set,  test_percept,k):
        neighbors = self.findNeighbors(train_set, test_percept, k)
        result = self.findPredictions(neighbors)
        return result

    #routine to find the neighbours
    def findNeighbors(self, train_set, test_percept, k):
        distance = 0
        distances = []
        length = len(test_percept)-1
        for x in range(len(train_set)):
            distance = self.calculateEuclideanDistance(test_percept, train_set[x], length)
            distances.append((train_set[x], distance))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    #routine to calculate the euclidean distance
    def calculateEuclideanDistance(self, test_percept, train_percept, length):
        distance = 0
        for x in range(length):
            distance += pow(abs(float(test_percept[x]) - float(train_percept[x])), 2)
        return math.sqrt(distance)

    #routine to find class based on votes
    def findPredictions(self, neighbors):
        classVotes = {}
        for i in range(len(neighbors)):
            result = neighbors[i][-1]
            if result in classVotes:
                classVotes[result] += 1
            else:
                classVotes[result] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    #routine to compute accuracy metric
    def measureAccuracy(self, test_set, predictions):
        accurate = 0
        for x in range(len(test_set)):
            if test_set[x][-1] == predictions[x]:
                accurate += 1
        return (accurate / float(len(test_set))) * 100.0


def main():
    #instantiating the environment
    directory = argv[1]
    k = int(argv[2])
    accuracy = 0
    folder_name = ['iris','heart','appendicitis','bupa','ionosphere','sonar','spectfheart','monk-2','titanic','banana']
    for f in range(0,len(folder_name)):
        pdf = PdfPages('neethu_sundarprasad_{}_plot.pdf'.format(folder_name[f]))
        environment = Environment(argv[1], argv[2], folder_name[f], accuracy)
        #variables for creating plots
        k_values = []
        accuracy_values = []
        #iterating through odd values of k from k-5 to k+5
        for n in range(k-5,k+5+1):
            if n<=0:
                continue
            k_values.append(n)
            accuracy = environment.sense(directory, n)
            accuracy_values.append(accuracy)
        #plotting accuracy against value of k
        figure = plt.figure('{} dataset'.format(directory))
        figure.suptitle('K versus Accuracy plot for {} dataset'.format(folder_name[f]))
        plt.plot(k_values, accuracy_values)
        plt.xlabel("K")
        plt.ylabel("Accuracy%")
        plt.show()
        pdf.savefig(figure)
        pdf.close()

#calling the main method
if __name__=='__main__':
    main()