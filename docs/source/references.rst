.. -*- coding: utf-8 -*-

.. _references:

==========
References
==========

PsiRESP is built on published scientific work.
**Please cite the relevant references when using PsiRESP in published work**,
specifically the citation in :ref:`Pre-configured classes`.

PsiRESP uses the Connolly algorithm to generate molecular surfaces :cite:p:`connolly1983`.


.. _citations-with-duecredit:

Citations with Duecredit
========================

Some citations can be automatically generated with duecredit_. This is installed with 
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

.. bibliography:: bibliography.bib
   :all: