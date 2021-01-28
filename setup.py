from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'iirc', 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='iirc',
    version='1.0.1',
    packages=['iirc', 'iirc.utils', 'iirc.lifelong_dataset'],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://chandar-lab.github.io/IIRC/',
    project_urls={
        'Documentation': 'https://iirc.readthedocs.io/en/latest/',
        'Source': 'https://github.com/chandar-lab/IIRC',
    },
    license='MIT',
    install_requires=['numpy', 'Pillow', 'lmdb'],
    python_requires='>=3',
    author='Mohamed Abdelsalam',
    author_email='mabdelsalam944@gmail.com',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description='This package provides a way for adapting the different datasets (currently supports *CIFAR-100* and '
                '*ImageNet*) to  the *iirc* setup and the *class incremental learning* setup, and loading them in a '
                'standardized manner. '
)
