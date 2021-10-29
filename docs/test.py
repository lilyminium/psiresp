import time
from psiresp.testing import FractalSnowflake  # , FractalSnowflakeHandler
import qcfractal.interface as ptl

# server = FractalSnowflake()
server = FractalSnowflake()
client = ptl.FractalClient(server, verify=False)
print(client)
mol = ptl.Molecule.from_data("""
    O 0 0 0
    H 0 0 2
    H 0 2 0
    units bohr
    """)
spec = {
    "keywords": None,
    "qc_spec": {
        "driver": "gradient",
        "method": "b3lyp",
        "basis": "6-31g",
        "program": "psi4"
    },
}

# Ask the server to compute a new computation
r = client.add_procedure("optimization", "geometric", spec, [mol])
print(r)
print(r.ids)
proc = client.query_procedures(id=r.ids)[0]
print(proc)
for i in range(10):
    time.sleep(30)
    proc = client.query_procedures(id=r.ids)[0]
    print(proc)
