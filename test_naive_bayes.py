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

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Test Naive Bayes for microbiome data.')
    parser.add_argument('-i', '--infile_taxa_abundance', required=True, help='Input file (i.e. taxonomy summary abundance file).', type=argparse.FileType('r'))
    parser.add_argument('-m', '--infile_mapping', required=True, help='Metadata input file (i.e. mapping file)', type=argparse.FileType('r'))
    parser.add_argument('-v', '--variable', type=str, default=1, help='Variable from the metadata file on which should be train and tested the Naive Bayes model.')
    parser.add_argument('-o', '--outfile_failed', type=argparse.FileType('w'), help='Output file for failed alignments')
    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction, help='Verbose output')
    args = parser.parse_args()
    return args

def main(arguments):

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    args = parse_command_line_arguments()
    verbose = args.verbose
 
    #if alignment_length < seed_end:
    #    raise Exception("--alignment_length <int> has to be >= than --seed_end <int>")
    #print("#query_id\tsubject_id\tmatch_aln\tquery_aln\tsubject_aln\tq_start\tq_end\ts_start\ts_end\texpect_value\tstrand", file=sys.stdout)
 
    infile_taxa_abundance = os.path.abspath(args.infile_taxa_abundance.name)
    infile_mapping = os.path.abspath(args.infile_mapping.name)
    variable = args.variable

    ut = Utils()
    ut.load_mapping_file(infile_mapping, "VisitCategory")
    ut.load_abundance_taxonomy(infile_taxa_abundance)
    ut.validate_data()
    ut.generate_y_keys()
    if verbose: print(ut.mapping, file=sys.stderr)
    if verbose: print(ut.data.shape, file=sys.stderr)
    if verbose: print(ut.X.shape, file=sys.stderr)
    if verbose: print(len(ut.sample_ids), file=sys.stderr)
    if verbose: print(ut.y, file=sys.stderr)
    if verbose: print(ut.y_key, file=sys.stderr)

    nb = NaiveBayes()
    X_train, X_test, y_train, y_test = train_test_split(ut.X, ut.y, test_size=0.2, random_state=123)
    nb.fit(X_train, y_train)
    # And then predict outcome of test data using training model.
    predictions = nb.predict(X_test)
    my_accuracy = accuracy(y_test, np.array(predictions))
    print("#########################################################################")
    print("Naive Bayes model was trained with variable '" + str(ut.variable) + "'", file=sys.stderr)
    print("Accuracy:" + str(my_accuracy), file=sys.stderr)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

