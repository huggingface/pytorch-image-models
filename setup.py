""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('timm/version.py').read())
setup(
    name='timm',
    version=__version__,
    description='PyTorch Image Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/huggingface/pytorch-image-models',
    author='Ross Wightman',
    author_email='ross@huggingface.co',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained models efficientnet mobilenetv3 mnasnet resnet vision transformer vit',
    packages=find_packages(exclude=['convert', 'tests', 'results']),
    include_package_data=True,
    install_requires=['torch >= 1.7', 'torchvision', 'pyyaml', 'huggingface_hub', 'safetensors'],
    python_requires='>=3.7',
)

