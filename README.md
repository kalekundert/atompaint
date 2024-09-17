AtomPaint
=========

[![Last release](https://img.shields.io/pypi/v/atompaint.svg)](https://pypi.python.org/pypi/atompaint)
[![Python version](https://img.shields.io/pypi/pyversions/atompaint.svg)](https://pypi.python.org/pypi/atompaint)
[![Documentation](https://img.shields.io/readthedocs/atompaint.svg)](https://torch-fuel.readthedocs.io/en/latest/)
[![Test status](https://img.shields.io/github/actions/workflow/status/kalekundert/atompaint/test.yml?branch=master)](https://github.com/kalekundert/atompaint/actions)
[![Test coverage](https://img.shields.io/codecov/c/github/kalekundert/atompaint)](https://app.codecov.io/github/kalekundert/atompaint)
[![Last commit](https://img.shields.io/github/last-commit/kalekundert/atompaint?logo=github)](https://github.com/kalekundert/atompaint)

*AtomPaint* is a collection of convolutional neural networks (CNNs) meant for 
learning from 3D images of macromolecular structures.  A unique, unifying 
feature of many of these networks is that maintain rotational equivariance; 
that is, they will recognize features no matter what orientation they appear 
in.  The models are implemented using the [PyTorch](https://pytorch.org) and 
[ESCNN](https://quva-lab.github.io/escnn/) frameworks.

Below are links to some of the datasets that these models can be trained on:

- [Macromolecular Gym](https://github.com/kalekundert/macromol_gym)
- [Atom3D](https://www.atom3d.ai/)
