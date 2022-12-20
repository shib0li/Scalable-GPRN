# Scalable Gaussian Process Regression Networks

## Updates on Dec 20th 2022

* Since the previous implemnetation was based on TF1.x which is far from outdated. I provide a light implementation with PyTorch. I have tested it with the latest PyTorch 1.13.1
* In this version, I only provide one Heat equation example, the inputs are  3 dimensional and the outputs are the solutions on a 64 by 64 mesh which is 4096 dimensional. I will provide more examples and test on the previous dataset later
* Calculation of the predictive probability is not inlcuded, but sample from the posteriors are provided. You need to install the Tensorly to support the factorized sampling with structured Kronecker posteriors
* Fix the known issue when compute the trace term of the expectations over the log likelihood.
* More stable and intuitive than previous version.

---

This is the python implementation of the paper [_Scalable Variational Gaussian Process Regression Networks_](https://arxiv.org/abs/2003.11489). we propose a scalable variational inference algorithm for GPRN, which not only captures the abundant posterior dependencies but also is much more efficient for massive outputs. Please refer our paper for more details.

If you have any questions, please email me at shibo 'at' cs.utah.edu, or create an issue on github. <s>The implementation only contains the [_Jura_](https://rdrr.io/cran/gstat/man/jura.html) example. If you are interested about other datasets presented in our paper, please contact our data collaboratos.

## System Requirement
We tested our code with python 3.6 on Ubuntu 18.04. Our implementation relies on TensorFlow 1.15. Other packages include scikit-learn for data standarlization and hdf5stroage for saving the results to mat file. Please use pip or conda to install those dependencies. 

```
pip install hdf5storage
```
```
pip install scikit-learn
```
We highly recommend to use [_Docker_](https://www.docker.com/) to freeze the running experiments. We attach our docker build file.

## Run
Please find the details of running configuration from *run.sh* 
</s>

## Citation
Please cite our work if you would like to use the code

```
@inproceedings{ijcai2020-340,
  title     = {Scalable Gaussian Process Regression Networks},
  author    = {Li, Shibo and Xing, Wei and Kirby, Robert M. and Zhe, Shandian},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {2456--2462},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/340},
  url       = {https://doi.org/10.24963/ijcai.2020/340},
}
```

## License
SGPRN is released under the MIT License (refer to the LICENSE file for details)
