import esm
import torch
from openbabel import pybel,openbabel
openbabel.obErrorLog.SetOutputLevel(0)

def smiles2mol2(smiles):
    mol = pybel.readstring("smi", smiles)
    mol.make3D()
    return mol.write("mol2")

class ESMEmbedder:
    def __init__(self, model_name='esm2_t6_8M_UR50D', device='cuda'):

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().to(device)
        self.device = device
        self.repr_layer = 6

    def embed_sequence(self, seq):
        batch_labels, batch_strs, batch_tokens = self.batch_converter([("protein", seq)])
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        rep = results["representations"][self.repr_layer][0, 1:-1]
        return rep 
    
