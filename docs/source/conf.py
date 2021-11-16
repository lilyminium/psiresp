# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Incase the project was not installed


import psiresp
import os
import sys
from ipywidgets.embed import DEFAULT_EMBED_REQUIREJS_URL

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'psiresp'
copyright = ("2020, Lily Wang. Project structure based on the "
             "Computational Molecular Science Python Cookiecutter version 1.2")
author = 'Lily Wang'

# The short X.Y version
version = '0.1'
# The full version, including alpha/beta/rc tags
release = '0.1-unstable'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinxcontrib.bibtex',
    'sphinx.ext.todo',
    'sphinx_sitemap',
    'sphinx_rtd_theme',
    # 'myst_parser',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinxcontrib.autodoc_pydantic',
    'nbsphinx',
    'sphinx.ext.autosectionlabel'
]

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

autosummary_generate = True
napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_ivar = True
autodoc_typehints = "description"

autodoc_default_options = {
    'inherited-members': "BaseModel",
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css']


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'psirespdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'psiresp.tex', 'psiresp Documentation', 'psiresp', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'psiresp', 'psiresp Documentation', [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'psiresp', 'psiresp Documentation', author, 'psiresp', 'A RESP plugin for Psi4', 'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {'https://docs.python.org/3': None,
                       'https://numpy.org/doc/stable/': None,
                       'https://docs.scipy.org/doc/scipy/reference/': None,
                       'https://matplotlib.org': None,
                       'https://www.rdkit.org/docs/': None,
                       'https://docs.mdanalysis.org/stable/': None,
                       'https://psicode.org/psi4manual/master/': None,
                       'https://networkx.org/documentation/stable/': None,
                       'https://www.rdkit.org/docs/': None,
                       'https://docs.qcarchive.molssi.org/projects/QCFractal/en/stable/': None,
                       'https://docs.qcarchive.molssi.org/projects/QCElemental/en/stable/': None,
                       }

ipython_warning_is_error = False
ipython_execlines = [
    'import numpy as np',
]


autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_config_summary = False
# Hide parameter list within class signature
autodoc_pydantic_model_hide_paramlist = True

nbsphinx_prolog = r"""
.. raw:: html
    <script src='http://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js'></script>
    <script>require=requirejs;</script>
"""

html_js_files = []

bibtex_bibfiles = ["bibliography.bib"]
