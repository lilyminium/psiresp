.. psiresp documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to psiresp's documentation!
=========================================================

See tests for examples of calculating RESP or ESP fits. For 
example, calculating a 2-stage restrained electrostatic potential fit 
as used in AMBER, analogous to the RESP-A1 model in R.E.D. ::

   import psi4
   import psiresp

   mol = psi4.Molecule.from_string("""10
      ! from R.E.D. examples/2-Dimethylsulfoxide
      C  3.87500   0.67800  -8.41700
      H  3.80000   1.69000  -8.07600
      H  3.40600   0.02600  -7.71100
      H  3.38900   0.58300  -9.36600
      S  5.35900   0.29300  -8.55900
      O  5.46000  -1.05900  -9.01400
      C  6.05500   0.43000  -7.19300
      H  7.08700   0.16300  -7.29000
      H  5.98000   1.44100  -6.85300
      H  5.58200  -0.21900  -6.48500""", fix_com=True,
      fix_orientation=True)

   r = psiresp.Resp.from_molecules([mol], charge=0, name='dmso')
   charges = r.run(stage_2=True,  # run stage 2
                   opt=True,  # geometry optimize first
                   hyp_a1=0.0005, # hyperbola restraints
                   hyp_a2=0.001,
                   restraint=True,  # restrain
                   method='hf',
                   basis='6=31g*',
                   equal_methyls=True,  # restrain methyl carbons to have the same charge
                   n_orient=2)  # automatically generate 2 molecules


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
