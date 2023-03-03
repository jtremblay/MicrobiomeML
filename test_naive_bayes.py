#!/usr/bin/env python

"""Python script that implements Naive Bayes model training and testing for microbiome data.
Developed and tested with python 3.9.0

Julien Tremblay - julien.tremblay@nrc-cnrc.gc.ca
"""

import argparse
import os
import sys
import re
import signal
import random
cwd = os.getcwd()
sys.path.append(cwd + '/lib')
sys.path.append(cwd + '/naive_bayes')
from lib.utils import Utils
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from naive_bayes.naive_bayes_v1 import NaiveBayes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from statistics import mean, stdev
import re

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Test Naive Bayes for microbiome data.')
    parser.add_argument('-i', '--infile_taxa_abundance', required=True, help='Input file (i.e. taxonomy summary abundance file).', type=argparse.FileType('r'))
    parser.add_argument('-m', '--infile_mapping', required=True, help='Metadata input file (i.e. mapping file)', type=argparse.FileType('r'))
    parser.add_argument('-v', '--variable', type=str, default=1, help='Variable from the metadata file on which should be train and tested the Naive Bayes model.')
    parser.add_argument('-d', '--distribution_type', type=str, default="gaussian", help='What type of distribution best fits the data. TODO:at the moment only Gaussian implemented.', choices=['gaussian'])
    parser.add_argument('-r', '--number_of_rounds', type=int, default=1, help='Number of training rounds to randomly train the model. for instance, if -r 4, four distinct training and testing sets will be generated from the complete data provided in input.')
    parser.add_argument('-f', '--find_optimal_features', action=argparse.BooleanOptionalAction, default=False, help='Attempt to find the optimal features from the entire dataset.')
    parser.add_argument('-o', '--outfile_failed', type=argparse.FileType('w'), help='Output file for failed alignments')
    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction, help='Verbose output')
    args = parser.parse_args()
    return args

def main(arguments):

    def dict_val(x):
        return x[1]

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    args = parse_command_line_arguments()
    verbose = args.verbose
    number_of_rounds = args.number_of_rounds
 
    infile_taxa_abundance = os.path.abspath(args.infile_taxa_abundance.name)
    infile_mapping = os.path.abspath(args.infile_mapping.name)
    variable = args.variable

    ut = Utils(verbose)
    ut.load_mapping_file(infile_mapping, variable)
    ut.load_abundance_taxonomy(infile_taxa_abundance)
    ut.validate_data()
    ut.generate_y_keys()
    if verbose: print(ut.mapping, file=sys.stderr)
    if verbose: print(ut.data.shape, file=sys.stderr)
    if verbose: print(ut.X.shape, file=sys.stderr)
    if verbose: print(len(ut.sample_ids), file=sys.stderr)
    if verbose: print("ut.y:" + str(ut.y.shape), file=sys.stderr)
    if verbose: print(ut.y_key, file=sys.stderr)

    print("#########################################################################")
    print("Naive Bayes model was trained with variable '" + str(ut.variable) + "'", file=sys.stderr)
    res = {}
    
    # Here basically do combination of all possible combination of features
    if args.find_optimal_features is True:
        #for i in range(1, 
        #taxa = ut.taxa
        curr_taxa = np.array([], dtype=object)
        j = 0
        for i in range(0, ut.taxa.shape[0] - 1):
            for j in range(i + 1, ut.taxa.shape[0]):
                for k in range(0,2):
                    #print("i: {} j: {} k: {}".format(i, j, k))
                    if(k == 0):
                        curr_taxa = ut.taxa[[i,j]]
                    else:
                        curr_taxa = ut.taxa[i:(j+1):1]

                    ut.generate_data_selection_by_taxa(curr_taxa)
                    #print(ut.X)
                    #print("current taxa: " + str(list(curr_taxa)))
                    
                    my_accuracies = []
                    for m,v in enumerate(range(1,(number_of_rounds + 1))):
                        nb = NaiveBayes(verbose)
                        my_random_int = random.randint(123,12345)
                        X_train, X_test, y_train, y_test = train_test_split(ut.X_selected_by_taxa, ut.y, test_size=0.2, random_state=my_random_int)
                        nb.fit(X_train, y_train)
                        predictions = nb.predict(X_test)
                        my_accuracy = accuracy(y_test, np.array(predictions))
                        #print("    Accuracy: {0:0.4f} ;random int {1}:".format(my_accuracy, my_random_int), file=sys.stderr)
                        my_accuracies.append(my_accuracy)
                    res[str(curr_taxa)] =  mean(my_accuracies)
        
        res_sorted = sorted(res.items(), key=dict_val)

        for key in  res:
            print(key)
            key2 = re.sub("\n", "\\n", key) 
            #print(key2)
            print(res[key])

    
    else:
        my_accuracies = []
        for i,v in enumerate(range(1,(number_of_rounds + 1))):
            nb = NaiveBayes(verbose)
            my_random_int = random.randint(123,12345)
            X_train, X_test, y_train, y_test = train_test_split(ut.X, ut.y, test_size=0.2, random_state=my_random_int)
            nb.fit(X_train, y_train)
            predictions = nb.predict(X_test)
            my_accuracy = accuracy(y_test, np.array(predictions))
            print("Accuracy: {0:0.4f} ;random int {1}:".format(my_accuracy, my_random_int), file=sys.stderr)

            res[i] = {'random_int': my_random_int,'accuracy' : my_accuracy}
            my_accuracies.append(my_accuracy)
        print("mean of accuracies: {}, stdev: {}".format(mean(my_accuracies), stdev(my_accuracies)))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

