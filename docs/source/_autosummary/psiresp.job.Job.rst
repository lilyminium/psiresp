psiresp.job.Job
===============

.. currentmodule:: psiresp.job

.. autoclass:: Job

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Job.__init__
      ~Job.compute_charges
      ~Job.compute_esps
      ~Job.compute_orientation_energies
      ~Job.construct
      ~Job.construct_surface_constraint_matrix
      ~Job.copy
      ~Job.dict
      ~Job.from_orm
      ~Job.generate_conformers
      ~Job.generate_molecule_charge_constraints
      ~Job.generate_orientations
      ~Job.json
      ~Job.optimize_geometries
      ~Job.parse_file
      ~Job.parse_obj
      ~Job.parse_raw
      ~Job.run
      ~Job.schema
      ~Job.schema_json
      ~Job.update_forward_refs
      ~Job.update_molecule_charges
      ~Job.validate
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Job.charges
      ~Job.molecules
      ~Job.qm_optimization_options
      ~Job.qm_esp_options
      ~Job.grid_options
      ~Job.resp_options
      ~Job.charge_constraints
      ~Job.working_directory
      ~Job.defer_errors
      ~Job.temperature
      ~Job.stage_1_charges
      ~Job.stage_2_charges
      ~Job.n_processes
   
   