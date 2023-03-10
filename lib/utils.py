#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:15:38 2023

@author: tremblayju
"""

import numpy as np
import sys

class Utils:
    
    def __init__(self, verbose=False):
        self._mapping = np.array([], dtype=object)
        self._variable_index = int
        self._variable = str
        self._data = np.array([], dtype=np.float64)
        self._sample_ids = []
        self._X = np.array([], dtype=np.float64)
        self._y_string = np.array([], dtype=object)
        self._y = np.array([], dtype=int)
        self._y_key = []
        self._verbose = verbose
        self._taxa = np.array([], dtype=object)
        # for selected taxa
        self._X_selected_by_taxa = np.array([], dtype=np.float64)
        self._y_selected_by_taxa = np.array([], dtype=object)

    @property
    def data(self):
        return self._data
    
    @property
    def mapping(self):
        return self._mapping
    
    @property
    def variable_index(self):
        return self._variable_index

    @property
    def sample_ids(self):
        return self._sample_ids
    
    @property
    def variable(self):
        return self._variable
    
    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y
    
    @property
    def X_selected_by_taxa(self):
        return self._X_selected_by_taxa

    @property
    def y_selected_by_taxa(self):
        return self._y_selected_by_taxa

    @property
    def y_key(self):
        return self._y_key

    @property
    def taxa(self):
        return self._taxa

    """Import metadata from a mapping tsv file)"""
    def load_mapping_file(self, infile_mapping, variable):
        fhand_m = open(infile_mapping, 'r')
        mapping = np.array([], dtype=object)
        variable_index = int
        self._variable = variable

        i = 0
        for line in fhand_m:
            line = line.rstrip()
            row = line.split("\t")
            
            if i == 0:
                header = row
                variable_index = header.index(variable)
                
            elif i == 1:
                mapping = np.array(row[0:len(row)], dtype=object)
            else:
                mapping2 = np.array(row[0:len(row)], dtype=object)
                mapping = np.vstack((mapping, mapping2))
                #print(mapping2)
                
            i = i + 1
            
        self._mapping = mapping
        self._variable_index = variable_index
     
    """"Import data from a taxonomic table tsv file"""    
    def load_abundance_taxonomy(self, infile):
        fhand = open(infile, 'r')
        data = np.array([], dtype=np.float64)
        taxa = []
        i = 0
        for line in fhand:
            line = line.rstrip()
            row = line.split("\t")
    
            if i == 0:
                sample_ids = row
                sample_ids.pop(0)
            elif i == 1:
                data = np.array(row[1:len(row)], dtype=float)
            else:
                taxa.append(row[0])
                data = np.vstack((data, np.array(row[1:len(row)])))
                
            i = i + 1
        
        self._data = np.transpose(data)
        self._sample_ids = sample_ids
        self._taxa = np.array(taxa)
    
    """"Import data from an ASV/OTU table tsv file format"""    
    def load_abundance_asv(self, infile):
        fhand = open(infile, 'r')
        data = np.array([], dtype=np.float64)
        taxa = []
        asvs = []
        i = 0
        for line in fhand:
            line = line.rstrip()
            row = line.split("\t")
    
            if i == 0:
                sample_ids = row
                sample_ids.pop(0)
            elif i == 1:
                data = np.array(row[1:len(row)], dtype=float)
            else:
                taxa.append(row[-1:][0])
                asvs.append(row[0])
                data = np.vstack((data, np.array(row[1:len(row)])))
                
            i = i + 1
        
        self._data = data
        self._sample_ids = sample_ids

    # make sure we have the same samples in data, mapping and sample_ids
    def validate_data(self): 
        idxs = []
        idxs2 = []

        for i,v in enumerate(self._sample_ids):
            # Do mapping
            if str(list(self._mapping.T[0])).find(v) != -1:
                k = list(self._mapping.T[0]).index(v)
                idxs.append(k)
                idxs2.append(i)
                #sample_ids_present.append(v)
            else:
                if self._verbose: print(v + " was not found!!!", file=sys.stderr)

        #print(idxs2)    
        sample_ids_present = np.array(self._sample_ids)[idxs2]
        if self._verbose: print(self._data.shape, file=sys.stderr)
        X = self._data[idxs2]

        y = self._mapping[:,self._variable_index]
        if self._verbose: print(sample_ids_present.shape, file=sys.stderr)
        #print(X.shape)
        #print(y.shape)
        self._sample_ids = sample_ids_present
        self._y_string = y 
        self._X = np.array(X, dtype=np.float64)
    
    def validate_data(self): 
        idxs = []
        idxs2 = []

        for i,v in enumerate(self._sample_ids):
            # Do mapping
            if str(list(self._mapping.T[0])).find(v) != -1:
                k = list(self._mapping.T[0]).index(v)
                idxs.append(k)
                idxs2.append(i)
                #sample_ids_present.append(v)
            else:
                if self._verbose: print(v + " was not found!!!", file=sys.stderr)

        #print(idxs2)    
        sample_ids_present = np.array(self._sample_ids)[idxs2]
        if self._verbose: print(self._data.shape, file=sys.stderr)
        X = self._data[idxs2]

        y = self._mapping[:,self._variable_index]
        if self._verbose: print(sample_ids_present.shape, file=sys.stderr)
        #print(X.shape)
        #print(y.shape)
        self._sample_ids = sample_ids_present
        self._y_string = y 
        self._X = np.array(X, dtype=np.float64)

    def generate_y_keys(self):
        # Before going further, we'll create a simple key equivalence scheme for classes/anwers
        # here normal = 0 and altered = 1. Then we can split the data and train the model.
        ys = np.unique(self._y_string)
        y2 = []
        for i,v in enumerate(self._y_string):
            k = list(ys).index(v)
            y2.append(k)

        # make sure to force conversion to appropriate data type before fitting the model.
        y2 = np.array(y2, dtype=int)
        self._y = y2
        self._y_key = ys
        #X = np.array(X, dtype=np.float64)
        # sicne data contains many 0, lets add 1 to each cell.
        #X = X + 1
        #print(X)

    def generate_data_selection_by_taxa(self, taxa_to_select):
        idxs = []
        #to_select = np.array(["taxon-3", "taxon-5"])
        for i,v in enumerate(taxa_to_select):
            if str(list(self._taxa)).find(v) != -1:
                k = list(self._taxa).index(v)
                idxs.append(k)
        print(idxs)
        
        self._X_selected_by_taxa = self._X[:,idxs]
        #self._y_selected_by_taxa = np.array(self._y[idxs])

