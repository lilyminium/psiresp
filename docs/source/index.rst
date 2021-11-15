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
and the Automated Topology Builder. Please see :ref:`RESP` for more details
on RESP fitting and the pre-configured classes (:ref:`Pre-configured classes`).


----------------
Getting involved
----------------

If you have a problem or feature request, please let us know
at the `Issue Tracker`_ on GitHub. If you have any questions or comments,
visit us on our `Discussions`_ board.

Pull requests and other contributions are also very welcomed. Please see
:ref:`Contributing` for more tips.


-------
License
-------

PsiRESP is licensed under the GNU Lesser General Public License (LGPL).


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   examples/README
   resp
   molecule
   charge_constraints
   hpc
   contributing
   api
   references
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _`Issue Tracker`: https://github.com/lilyminium/psiresp/issues
.. _Discussions: https://github.com/lilyminium/psiresp/discussions