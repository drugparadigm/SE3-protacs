import torch
import torch.nn as nn
import esm
from torch_geometric.utils import to_dense_batch
from se3_transformer_pytorch import SE3Transformer


class ESMWrapper(nn.Module):
    def __init__(self,dropout=0.2):
        super().__init__()
        self.down_proj = nn.Identity()
        
    def forward(self, x):
        x = self.down_proj(x)
        return x


class GraphTransformer(nn.Module):
    """SE(3) Transformer block to process 3D graph data."""
    def __init__(self, num_embeddings, dim=128
                 , depth=1, heads=9, dim_head=8, num_degrees=1):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, dim)
        self.transformer = SE3Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            num_degrees=num_degrees
        )

    def forward(self, data):
        feats, coors, batch = data.x, data.pos, data.batch
        feats = self.embed(feats.squeeze(-1))
        dense_feats, node_mask = to_dense_batch(feats, batch)
        dense_coors, _ = to_dense_batch(coors, batch)
        transformed_feats_dict = self.transformer(dense_feats, dense_coors, mask=node_mask)
        scalar_feats = transformed_feats_dict
        scalar_feats = scalar_feats.masked_fill(~node_mask[..., None], 0.)
        return scalar_feats

class Model(nn.Module):
    def __init__(self,
                 ligase_ligand_model,
                 ligase_model,
                 target_ligand_model,
                 target_model,
                 linker_model,
                 dim=320,
                 proj_dim=128,
                 drop_out=0.2):
        super().__init__()

        self.ligase_ligand_model = ligase_ligand_model
        self.ligase_model = ligase_model
        self.target_ligand_model = target_ligand_model
        self.target_model = target_model
        self.linker_model = linker_model

        self.proj = nn.Linear(dim, proj_dim)

        self.attention_layer = nn.Linear(proj_dim, proj_dim)
        self.target_attention_layer = nn.Linear(proj_dim, proj_dim)
        self.ligase_attention_layer = nn.Linear(proj_dim, proj_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

        self.fc1 = nn.Linear(proj_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.out = nn.Linear(128, 2)

   
    def forward(self,
                ligase_ligand,
                ligase,
                target_ligand,
                target,
                linker):

        target = self.proj(target)
        ligase = self.proj(ligase)

        v_0 = self.ligase_ligand_model(ligase_ligand)
        v_1 = self.target_ligand_model(target_ligand)
        v_2 = self.linker_model(linker)
        v_t = self.target_model(target)
        v_l = self.ligase_model(ligase)

        v_d = torch.cat([v_1, v_2, v_0], dim=1)
        v_t_d = torch.cat([v_t, v_d], dim=1)
        v_l_d = torch.cat([v_l, v_d], dim=1)

        target_att = self.target_attention_layer(v_t_d)
        ligase_att = self.ligase_attention_layer(v_l_d)

        t_att_layers = target_att.unsqueeze(2).repeat(1, 1, v_l_d.shape[-2], 1)
        l_att_layers = ligase_att.unsqueeze(1).repeat(1, v_t_d.shape[-2], 1, 1)

        Atten_matrix = self.attention_layer(self.relu(t_att_layers + l_att_layers))

        target_atte = self.sigmoid(Atten_matrix.mean(2))
        ligase_atte = self.sigmoid(Atten_matrix.mean(1))

        v_t_d = v_t_d * 0.5 + v_t_d * target_atte
        v_l_d = v_l_d * 0.5 + v_l_d * ligase_atte

        fully1 = self.relu(self.fc1(
            torch.cat([torch.sum(v_t_d, 1), torch.sum(v_l_d, 1)], dim=1)
        ))

        fully1 = self.dropout2(fully1)
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.relu(self.fc3(fully2))

        predict = self.out(fully3)

        return predict, target_atte, ligase_atte

