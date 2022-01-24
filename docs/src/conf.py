# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'SKA SDP Processing Function Library'
copyright = '2022, The SKA SDP Processing Function Library Developers'
author = 'The SKA SDP Processing Function Library Developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'breathe'
]

# Set Breathe configuration (uses Doxygen XML output).
breathe_projects = { "ska-sdp-func": os.path.join("..", "doxygen", "xml")}
breathe_default_project = "ska-sdp-func"

cpp_index_common_prefix = ['sdp_']

# Run Doxygen to generate the XML.
subprocess.call('cd ../; doxygen', shell=True)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_context = {
    'favicon': 'img/favicon.ico',
    'logo': 'img/logo.jpg',
    'theme_logo_only' : True
}

html_static_path = ["_static"]

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "preamble": r"""
""",
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

pygments_style = 'sphinx'
autodoc_member_order = 'bysource'
# html4_writer = True
html_show_sourcelink = False
master_doc = 'index'
