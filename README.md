# MicrobiomeML
This repository contains python code to train and test machine learning algorithms for microbiome data. Still in early development.

To run:
```
./test_naive_bayes.py -i ./data/feature_table_final_normalized_L6_diffabun.tsv -m ./data/mapping_file.tsv  -v "VisitCategory" -r 10
```

Will give something similar as output.

```
#########################################################################
Naive Bayes model was trained with variable 'VisitCategory'
Accuracy: 0.7857 ;random int 7259:
Accuracy: 0.6964 ;random int 7322:
Accuracy: 0.7321 ;random int 9258:
Accuracy: 0.7679 ;random int 10686:
Accuracy: 0.7143 ;random int 11636:
Accuracy: 0.6607 ;random int 11278:
Accuracy: 0.7679 ;random int 7987:
Accuracy: 0.6607 ;random int 4283:
Accuracy: 0.7143 ;random int 6236:
Accuracy: 0.7857 ;random int 215:
mean of accuracies: 0.7285714285714285, stdev: 0.047470004850897356
```
