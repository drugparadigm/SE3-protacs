import pandas as pd
import os
from openbabel import pybel,openbabel
openbabel.obErrorLog.SetOutputLevel(0)


df = pd.read_csv('data/1979_samples.csv') 

output_base = "data/mol2_files/"
os.makedirs(output_base, exist_ok=True)

def conversion(smiles, prefix, compound_id):
    output_relative_path=f"{prefix}_{compound_id}.mol2"
    output_path = os.path.join(output_base, output_relative_path)

    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        mol.write("mol2", output_path, overwrite=True)

    except Exception as e:
        print(f"Failed for SMILES {smiles}: {e}")

for _, row in df.iterrows():
    linker_smiles = row["linker_smiles"]
    warhead_smiles = row["warhead_smiles"]
    e3_ligase_smiles = row["e3_ligase_smiles"]
    compound_id=row["compound id"]

    if linker_smiles:
        conversion(linker_smiles,"linker",compound_id)
    if warhead_smiles:
        conversion(warhead_smiles,"warhead",compound_id)
    if e3_ligase_smiles:
        conversion(e3_ligase_smiles,"e3_ligand",compound_id)

print("Conversion complete.")
