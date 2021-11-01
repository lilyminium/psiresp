from .job import Job


def configure(**configuration):
    class ConfiguredJob(Job):
        def __init__(self, *args, **kwargs):
            obj = Job(*args, **kwargs)
            objdct = obj.dict()
            for option_name, option_config in configuration.items():
                prefix = option_name.split("_")[0] + "_"
                for field in objdct.keys():
                    if field.startswith(prefix):
                        for name, value in option_config.items():
                            objdct[field][name] = value
            super().__init__(**objdct)
    return ConfiguredJob


RespA1 = configure(qm_options=dict(method="hf",
                                   basis="6-31g*"
                                   ),
                   grid_options=dict(use_radii="msk"),
                   resp_options=dict(resp_a1=0.0005,
                                     resp_a2=0.001,
                                     resp_b=0.1,
                                     stage_2=True,
                                     exclude_hydrogens=True),
                   )
