import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import argparse
from model import *
from utils import *
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = "model/SE3-PROTACs.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MAX_LENGTH = 1000

LIGAND_ATOM_TYPE = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
EDGE_ATTR = {'1': 1, '2': 2, '3': 3, 'ar': 4, 'am': 5}


def mol2graph(smiles, ATOM_TYPE):
    mol2_str = smiles2mol2(smiles)
    lines = mol2_str.splitlines(keepends=True)

    try:
        atom_end_line = lines.index('@<TRIPOS>UNITY_ATOM_ATTR\n')
    except ValueError:
        atom_end_line = lines.index('@<TRIPOS>BOND\n')

    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n') + 1:atom_end_line]
    bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]

    atoms = []
    positions = []
    for atom in atom_lines:
        parts = atom.split()
        ele = parts[5].split('.')[0]
        atoms.append(ATOM_TYPE.index(ele) if ele in ATOM_TYPE else len(ATOM_TYPE))
        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])

    edge_1 = [int(i.split()[1]) - 1 for i in bond_lines]
    edge_2 = [int(i.split()[2]) - 1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]

    x = torch.tensor(atoms, dtype=torch.long)
    edge_idx = torch.tensor([edge_1 + edge_2, edge_2 + edge_1], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr + edge_attr, dtype=torch.long)

    positions = torch.tensor(positions, dtype=torch.float)
    tdEdge = to_scipy_sparse_matrix(edge_idx, edge_attr).todense()
    tdEdge = torch.from_numpy(np.array(tdEdge, dtype=np.float32).flatten())

    graph = Data(x=x, pos=positions, edge=tdEdge)
    return graph


def read_fasta(file_path):
    """Read first sequence from .fa/.fasta file."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    return seq


def read_smi(file_path):
    """Read first SMILES string from .smi file."""
    with open(file_path, "r") as f:
        smiles = f.readline().strip()
    return smiles


def predict_for_molecule(model, ligase_ligand_smiles, ligase_seq,
                         target_ligand_smiles, target_seq,
                         linker_smiles):
    try:
        esm = ESMEmbedder(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Convert SMILES → graph
        warhead = mol2graph(target_ligand_smiles, LIGAND_ATOM_TYPE)      # target_ligand
        ligase_ligand = mol2graph(ligase_ligand_smiles, LIGAND_ATOM_TYPE)  # ligase_ligand
        linker = mol2graph(linker_smiles, LIGAND_ATOM_TYPE)              # linker

        # Embed protein sequences
        e3_ligase_sequence = esm.embed_sequence(ligase_seq)
        target_sequence = esm.embed_sequence(target_seq)

        sample = {
            "ligase_ligand": ligase_ligand,
            "ligase": e3_ligase_sequence,
            "target_ligand": warhead,
            "target": target_sequence,
            "linker": linker,
            "label": 0  # Dummy label for inference
        }

    except Exception as e:
        print(f"[!] Error processing inputs: {e}")
        return None

    with torch.no_grad():
        ligase = sample['ligase'].unsqueeze(0)
        target = sample['target'].unsqueeze(0)

        predict, _, _ = model(
            sample['ligase_ligand'].to(device),
            ligase.to(device),
            sample['target_ligand'].to(device),
            target.to(device),
            sample['linker'].to(device)
        )
        pred_y = torch.max(predict, 1)[1].item()
    return pred_y


def load_model():
    target_ligand_model = GraphTransformer(num_embeddings=10)
    ligase_ligand_model = GraphTransformer(num_embeddings=10)
    linker_model = GraphTransformer(num_embeddings=10)

    ligase_model = ESMWrapper()
    target_model = ESMWrapper()

    model = Model(
        ligase_ligand_model=ligase_ligand_model,
        ligase_model=ligase_model,
        target_ligand_model=target_ligand_model,
        target_model=target_model,
        linker_model=linker_model
    )

    try:
        loaded_content = torch.load(MODEL_PATH,
                                    map_location=lambda storage, loc: storage)
        model.load_state_dict(loaded_content['model_state_dict'])
    except Exception as e:
        print(f"[!] Error loading model from {MODEL_PATH}: {e}")
        return None

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligase_smi", type=str, required=True, help="Path to E3 ligase ligand .smi file")
    parser.add_argument("--ligase_fa", type=str, required=True, help="Path to E3 ligase protein .fa file")
    parser.add_argument("--target_smi", type=str, required=True, help="Path to target ligand (warhead) .smi file")
    parser.add_argument("--target_fa", type=str, required=True, help="Path to target protein .fa file")
    parser.add_argument("--linker_smi", type=str, required=True, help="Path to linker .smi file")

    args = parser.parse_args()

    # Read inputs
    ligase_smiles = read_smi(args.ligase_smi)
    ligase_seq = read_fasta(args.ligase_fa)
    target_smiles = read_smi(args.target_smi)
    target_seq = read_fasta(args.target_fa)
    linker_smiles = read_smi(args.linker_smi)

    model = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return

    model.to(device)
    model.eval()

    print("Model loaded. Starting prediction...")
    result = predict_for_molecule(
        model,
        ligase_smiles,
        ligase_seq,
        target_smiles,
        target_seq,
        linker_smiles
    )
    print(f"Prediction result: {result}")


if __name__ == "__main__":
    main()
