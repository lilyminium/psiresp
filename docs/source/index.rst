.. psiresp documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PsiRESP's documentation!
=========================================================

PsiRESP is a package for calculating atomic partial charges from
restrained and unrestrained electrostatic potential fits using Psi4.
It is highly flexible, configurable, easy to use, and totally written in Python.
It supports fitting to multiple orientations and conformers,
as well as both intra-molecular and inter-molecular charge constraints for
multi-molecule fits. As fitting to multiple conformers and orientations
is recommended for best results, PsiRESP furthermore has methods for
automated conformer generation using RDKit, and automated orientation generation
around heavy atoms.

For ease of use, PsiRESP offers pre-configured classes that correspond with
existing popular tools such as the RESP ESP Charge Derive (R.E.D.) tools
and the Automated Topology Builder.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   parallelization
   hpc
   api
   io
   options/index
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
