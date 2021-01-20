.. IIRC documentation master file, created by
   sphinx-quickstart on Fri Jan  8 15:55:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IIRC's documentation!
================================
.. image:: ../images/task_summary.png
   :width: 600
   :alt: IIRC Setup

**iirc** is a package for adapting the different datasets (currently supports *CIFAR-100* and *ImageNet*) to
the *iirc* setup and the *class incremental learning* setup, and loading them in a standardized manner.

**lifelong_methods** is a package that standardizes the different stages any lifelong learning method passes by,
hence it provides a faster way for implementing new ideas and embedding them in the same training code as other
baselines, it provides as well the implementation of some of these baselines.

`Project Homepage <https://chandar-lab.github.io/IIRC/>`__ | `Project Paper <https://arxiv.org/abs/2012.12477>`__ |
`Source Code <https://github.com/chandar-lab/IIRC>`__ | `PyPI Package <https://pypi.org/project/iirc/>`__

.. toctree::
   :maxdepth: 1
   :caption: Contents:

Installation:
=============
you can install the **iirc** package using the following command

.. code-block:: none

   pip install iirc

To use it with PyTorch, you will need as well to install *PyTorch* (1.5.0) and *torchvision* (0.6.0)

Guide:
======
.. toctree::
   :maxdepth: 1

   iirc_tutorial
   loading_datasets_tutorial
   lifelong_methods_guide

Documentation:
==============
.. toctree::
   :maxdepth: 1

   modules

.. Indices and tables
.. ==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Paper
---------

IIRC is introduced in the paper `“IIRC: Incremental Implicitly-Refined Classification ” <https://arxiv.org/abs/2012.12477>`__.
If you find this work useful for your research, this is the way to cite it:

.. code:: bibtex

    @misc{abdelsalam2021iirc,
        title = {IIRC: Incremental Implicitly-Refined Classification},
        author={Mohamed Abdelsalam and Mojtaba Faramarzi and Shagun Sodhani and Sarath Chandar},
        year={2021}, eprint={2012.12477}, archivePrefix={arXiv},
        primaryClass={cs.CV} 
    }

Community
-----------

If you think you can help us make the **iirc** and **lifelong_methods** packages more useful for the lifelong learning
community, please don't hesitate to `submit an issue or send a pull request. <https://github.com/chandar-lab/IIRC>`__
