import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import argparse
from model import *
from utils import *
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "model/SE3-PROTACs.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LIGAND_ATOM_TYPE = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
EDGE_ATTR = {'1': 1, '2': 2, '3': 3, 'ar': 4, 'am': 5}

# ------------------------------------------------------------------
# SMILES → Graph
# ------------------------------------------------------------------
def mol2graph(smiles, ATOM_TYPE):
    mol2_str = smiles2mol2(smiles)
    lines = mol2_str.splitlines(keepends=True)

    try:
        atom_end_line = lines.index('@<TRIPOS>UNITY_ATOM_ATTR\n')
    except ValueError:
        atom_end_line = lines.index('@<TRIPOS>BOND\n')

    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n') + 1:atom_end_line]
    bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]

    atoms, positions = [], []
    for atom in atom_lines:
        parts = atom.split()
        ele = parts[5].split('.')[0]
        atoms.append(ATOM_TYPE.index(ele) if ele in ATOM_TYPE else len(ATOM_TYPE))
        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])

    edge_1 = [int(i.split()[1]) - 1 for i in bond_lines]
    edge_2 = [int(i.split()[2]) - 1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]

    x = torch.tensor(atoms, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    edge_idx = torch.tensor([edge_1 + edge_2, edge_2 + edge_1], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr + edge_attr, dtype=torch.long)

    tdEdge = to_scipy_sparse_matrix(edge_idx, edge_attr).todense()
    tdEdge = torch.from_numpy(np.array(tdEdge, dtype=np.float32).flatten())

    return Data(x=x, pos=pos, edge=tdEdge)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
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
        linker_model=linker_model,
    )

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

# ------------------------------------------------------------------
# Read input files
# ------------------------------------------------------------------
def read_fasta(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return "".join([l.strip() for l in lines if not l.startswith(">")])

def read_smi(file_path):
    with open(file_path, "r") as f:
        return f.readline().strip()

# ------------------------------------------------------------------
# Single sample prediction
# ------------------------------------------------------------------
def predict_single(model, ligase_smi, ligase_seq, target_smi, target_seq, linker_smi, esm):
    ligase_ligand = mol2graph(ligase_smi, LIGAND_ATOM_TYPE)
    warhead = mol2graph(target_smi, LIGAND_ATOM_TYPE)
    linker = mol2graph(linker_smi, LIGAND_ATOM_TYPE)

    e3_ligase_emb = esm.embed_sequence(ligase_seq)
    target_emb = esm.embed_sequence(target_seq)

    with torch.no_grad():
        logits, _, _ = model(
            ligase_ligand.to(DEVICE),
            e3_ligase_emb.unsqueeze(0).to(DEVICE),
            warhead.to(DEVICE),
            target_emb.unsqueeze(0).to(DEVICE),
            linker.to(DEVICE),
        )

        probs = torch.softmax(logits, dim=1)
        score = probs[:, 1].item()
        pred = torch.argmax(probs, dim=1).item()

    return pred, score

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligase_smi", type=str, required=True)
    parser.add_argument("--ligase_fa",  type=str, required=True)
    parser.add_argument("--target_smi", type=str, required=True)
    parser.add_argument("--target_fa",  type=str, required=True)
    parser.add_argument("--linker_smi", type=str, required=True)
    args = parser.parse_args()

    ligase_smi = read_smi(args.ligase_smi)
    ligase_seq = read_fasta(args.ligase_fa)
    target_smi = read_smi(args.target_smi)
    target_seq = read_fasta(args.target_fa)
    linker_smi = read_smi(args.linker_smi)

    print("Loading model...")
    model = load_model()

    print("Loading ESM embedder...")
    esm = ESMEmbedder(device=str(DEVICE))

    print("Running prediction...")
    pred, score = predict_single(
        model, ligase_smi, ligase_seq,
        target_smi, target_seq, linker_smi, esm
    )

    print("\n================ PREDICTION RESULT ================")
    print(f"Degradation Score : {score:.4f}")
    print(f"Prediction        : {'Good Degrader' if pred == 1 else 'Bad Degrader'} ({pred})")
    print("====================================================")

if __name__ == "__main__":
    main()