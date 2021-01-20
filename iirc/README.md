# iirc package
![img.png](https://raw.githubusercontent.com/chandar-lab/IIRC/master/docs/images/task_summary.png)

This package provides a way for adapting the different datasets (currently supports *CIFAR-100* and *ImageNet*) to 
the *iirc* setup and the *class incremental learning* setup, and loading them in a standardized manner.

The documentation and usage guide are available [here](https://iirc.readthedocs.io/en/latest/)

[Homepage](https://chandar-lab.github.io/IIRC/) | [Paper](https://arxiv.org/abs/2012.12477) | 
[Documentation](https://iirc.readthedocs.io/en/latest/)

## Installation
you can install this package using the following command

```pip install iirc```

## Dataset Downloading Instructions
### CIFAR-100
To be able to run the code with *CIFAR-100* derived datasets, just download the dataset from the 
[official website](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it, or use the 
*./utils/download_cifar.py* file. 

### ImageNet
In the case of ImageNet, it has to be downloaded manually, and be arranged in the following manner: 
* dataset folder
  * train 
    * n01440764
    * n01443537
    * …
  * val 
    * n01440764 
    * n01443537 
    * …
    
## Contributing
If you think you can help us make the **iirc** package more useful for the lifelong learning 
community, please don't hesistate to submit an issue or send a pull request.

## Citation
If you find this work useful for your research, this is the way to cite it:

```
@misc{abdelsalam2021iirc,
title = {IIRC: Incremental Implicitly-Refined Classification},
author={Mohamed Abdelsalam and Mojtaba Faramarzi and Shagun Sodhani and Sarath Chandar},
year={2021}, eprint={2012.12477}, archivePrefix={arXiv},
primaryClass={cs.CV} }
```
