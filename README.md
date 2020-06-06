# Scalable Gaussian Process Regression Networks

This is the python implementation of the paper [_Scalable Variational Gaussian Process Regression Networks_](https://arxiv.org/abs/2003.11489). we propose a scalable variational inference algorithm for GPRN, which not only captures the abundant posterior dependencies but also is much more efficient for massive outputs. Please refer our paper for more details.

If you have any questions, please email me at shibo 'at' cs.utah.edu, or create an issue on github. The implementation only contains the [_Jura_](https://rdrr.io/cran/gstat/man/jura.html) example. If you are interested about other datasets presented in our paper, please contact our data collaboratos.

## System Requirement
We tested our code with python 3.6 on Ubuntu 18.04. Our implementation relies on TensorFlow 1.15. Other packages include scikit-learn for data standarlization and hdf5stroage for saving the results to mat file. Please use pip or conda to install those dependencies. 

```
pip install hdf5storage
```
```
pip install scikit-learn
```

