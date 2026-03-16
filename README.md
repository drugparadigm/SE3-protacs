# SE3-PROTACs: Structure-Equivariant Transformer for PROTAC Degradation Prediction  

## 📌 Overview  
This repository implements **SE3-PROTACs**, a structure-equivariant deep learning framework for predicting **PROTAC-mediated protein degradation**.  

- PROTAC molecules are represented as **3D molecular graphs** from `.mol2` files.  
- Protein sequences (POI and E3 ligase) are encoded using **ESM embeddings**.  
- A **SE(3)-Transformer** backbone ensures rotational and translational equivariance for molecular inputs.  
- Outputs are **binary degradation predictions** (degrader vs. non-degrader).  
---

## 🧩 Features  
- **SE(3)-Transformer backbone** for equivariant molecular graph learning  
- **ESM-2 embeddings** for protein sequences  
- **Feature fusion** of molecules and proteins  
- **Input**:  
  - PROTAC components (smiles for warhead, linker, E3 ligand)  
  - Proteins (FASTA string for POI and E3 ligase)  
- **Output**: PROTAC **degradation prediction** (0/1)  

---


## ⚙️ Installation  

### 1. Clone the repository  
```bash
git clone https://github.com/drugparadigm/SE3-protacs.git
cd SE3-protacs
conda env create -f environment.yml
conda activate se3protacs
```


📥 Data Preparation

Place your PROTAC data in the data/ folder.

PROTAC components: smiles

Proteins: FASTA strings

Convert SMILES → mol2 format using:

```python prepare_data.py```

Then pre-compute ESM embeddings:  

```bash

python pre_compute_emb.py

```

🚀 Training

Run the main training script:
```python main.py```

Training logs and model checkpoints will be saved inside the model/ directory.


🔍 Inference
Run on one sample PROTAC
```
python casestudy.py\
  --ligase_smi data/casestudy/e3_ligase_ligand.smi \
  --ligase_fa data/casestudy/e3_ligase.fa \
  --target_smi data/casestudy/warhead.smi \
  --target_fa data/casestudy/target.fa \
  --linker_smi data/casestudy/linker.smi
```
