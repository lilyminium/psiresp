API
===

Important classes
-----------------

.. autosummary::
   :toctree: _autosummary
   :caption: Important classes

   psiresp.job.Job
   psiresp.molecule.Molecule
   psiresp.conformer.Conformer
   psiresp.orientation.Orientation
   psiresp.resp.RespCharges

The below classes are less used, but part of
constructing and interacting with charge constraints.

.. autosummary::
   :toctree: _autosummary
   
   psiresp.molecule.Atom
   psiresp.charge.ChargeSumConstraint
   psiresp.charge.ChargeEquivalenceConstraint
   


Options for customizing RESP
----------------------------

Use these classes to customise the calculation of RESP charges.

.. autosummary::
   :toctree: _autosummary
   :caption: Options

   psiresp.charge.ChargeConstraintOptions
   psiresp.conformer.ConformerGenerationOptions
   psiresp.grid.GridOptions
   psiresp.qm.PCMOptions
   psiresp.qm.QMGeometryOptimizationOptions
   psiresp.qm.QMEnergyOptions
   psiresp.resp.RespOptions


Base and utility classes
------------------------

Users are not expected to interact with these classes directly.

.. autosummary::
   :toctree: _autosummary
   :caption: Utility classes

   psiresp.base.Model
   psiresp.moleculebase.BaseMolecule
   psiresp.charge.MoleculeChargeConstraints
   psiresp.constraint.ESPSurfaceConstraintMatrix
   psiresp.constraint.SparseGlobalConstraintMatrix
   psiresp.qcutils.QCWaveFunction
