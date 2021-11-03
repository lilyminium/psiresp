.. -*- coding: utf-8 -*-

.. _references:

==========
References
==========

References good


.. _citations-with-duecredit:

Citations with Duecredit
========================

Some citations can be genrated with duecredit_. This is installed with 
the conda environment files in ``devtools/``, or can be otherwise installed 
with:

.. code-block:: bash

   pip install duecredit

Generate a bibliography with:

.. code-block:: bash

   cd /path/to/yourmodule
   python -m duecredit yourscript.py

Do it in the BibTeX format, using:

.. code-block:: bash
 
   duecredit summary --format=bibtex 


.. _duecredit: https://github.com/duecredit/duecredit


Bibliography
============

.. bibliography::