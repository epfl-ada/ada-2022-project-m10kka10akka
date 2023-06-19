# Blind Mitigation
## Machine Learning
This folder implements the entire machine learning flow. 
```
├── crossvalidators
│   └── implements the type of cross validation
├── models
│   └── implements the type of model to use
├── samplers
│   └── implements the way the data is sampled (oversample/undersample) before training the data
├── scorers
│   └── determines how to compute the ml and fairness metrics at each fold
├── splitters
│   └── implements the way the data is split across the folds
└── xval_maker
    └── Creates the pipeline with each of the chosen in the previous folders


```