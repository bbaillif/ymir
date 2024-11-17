import torch

from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn.math import (soft_one_hot_linspace,
                       soft_unit_step)
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch_geometric.data import Batch

L_MAX = 3
MAX_RADIUS = 6.0
NUMBER_OF_BASIS = 10
IRREPS_QUERY = o3.Irreps('16x0e + 16x1o')
IRREPS_KEY = o3.Irreps('16x0e + 16x1o')
IRREPS_ATTENTION = o3.Irreps('0e')
MOL_ID_EMBEDDING_SIZE = 16
NODE_Z_EMBEDDING_SIZE = 64 - MOL_ID_EMBEDDING_SIZE
MIDDLE_LAYER_SIZE = 32

class Transformer(torch.nn.Module):
    
    def __init__(self, 
                 irreps_output: o3.Irreps,
                 num_elements: int,
                 node_z_embedding_size: int = NODE_Z_EMBEDDING_SIZE,
                 node_mol_embedding_size: int = MOL_ID_EMBEDDING_SIZE,
                 irreps_query: o3.Irreps = IRREPS_QUERY,
                 irreps_key: o3.Irreps = IRREPS_KEY,
                 irreps_attention: o3.Irreps = IRREPS_ATTENTION,
                 lmax: int = L_MAX,
                 max_radius: float = MAX_RADIUS,
                 number_of_basis: int = NUMBER_OF_BASIS,
                 middle_layer_size: int = MIDDLE_LAYER_SIZE,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.irreps_output = irreps_output
        self.num_elements = num_elements
        self.node_z_embedding_size = node_z_embedding_size
        self.node_mol_embedding_size = node_mol_embedding_size
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_attention = irreps_attention
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.middle_layer_size = middle_layer_size
        
        self.node_z_embedder = torch.nn.Embedding(num_embeddings=self.num_elements, 
                                                  embedding_dim=self.node_z_embedding_size)
        self.node_mol_embedder = torch.nn.Embedding(num_embeddings=2, # seed or pocket
                                                    embedding_dim=self.node_mol_embedding_size)
        
        self.node_embedding_size = self.node_z_embedding_size + self.node_mol_embedding_size
        self.irreps_input_node = o3.Irreps(f'{self.node_embedding_size}x0e')
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, 
                                        normalize=True, 
                                        normalization="component")
        self.query_linear = o3.Linear(irreps_in=self.irreps_input_node, 
                                      irreps_out=self.irreps_query)
        
        # tp = Tensor Product
        self.tp_key = o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_input_node, 
                                                   irreps_in2=self.irreps_sh, 
                                                   irreps_out=self.irreps_key, 
                                                   shared_weights=False)
        
        # hs has the form [input layer, middle layer, output layer]
        self.key_weights_hs = [self.number_of_basis, 
                               self.middle_layer_size, 
                               self.tp_key.weight_numel] 
        self.key_weights_fc = FullyConnectedNet(hs=self.key_weights_hs, 
                                                act=torch.nn.functional.silu)
        
        self.tp_value = o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_input_node, 
                                                        irreps_in2=self.irreps_sh, 
                                                        irreps_out=self.irreps_output, 
                                                        shared_weights=False)
        
        self.value_weights_hs = [self.number_of_basis, 
                                 self.middle_layer_size, 
                                 self.tp_value.weight_numel]
        self.value_weights_fc = FullyConnectedNet(hs=self.value_weights_hs, 
                                                  act=torch.nn.functional.silu)
        
        self.tp_query_key = o3.FullyConnectedTensorProduct(irreps_in1=self.irreps_query, 
                                                            irreps_in2=self.irreps_key, 
                                                            irreps_out=self.irreps_attention)
    
    def forward(self,
                batch: Batch):
        x = batch.x
        pos = batch.pos
        mol_id = batch.mol_id
        
        x = self.node_z_embedder(x)
        mol_id = self.node_mol_embedder(mol_id)
        
        x = torch.cat([x, mol_id], dim=-1)
        
        edge_src, edge_dst = radius_graph(x=pos, 
                                          r=self.max_radius)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))

        edge_sh = self.sh(edge_vec)

        query = self.query_linear(x)
        
        key_weights = self.key_weights_fc(edge_length_embedded)
        x_src = x[edge_src]
        key = self.tp_key(x=x_src, 
                          y=edge_sh, 
                          weight=key_weights)
        
        value_weights = self.value_weights_fc(edge_length_embedded)
        value = self.tp_value(x=x_src, 
                              y=edge_sh, 
                              weight=value_weights)

        query_dst = query[edge_dst]
        query_key = self.tp_query_key(x=query_dst, 
                                      y=key)
        
        exp = edge_weight_cutoff[:, None] * query_key.exp() # numerator
        z = scatter(src=exp, 
                    index=edge_dst, 
                    dim=0, 
                    dim_size=len(x)) # denominator, default scatter reduce is sum
        z[z == 0] = 1 # to avoid 0/0 when all the neighbors are exactly at the cutoff
        alpha = exp / z[edge_dst]
        
        attention = alpha.relu().sqrt()
        attended_value = attention * value
        output = scatter(src=attended_value,
                         index=edge_dst, 
                         dim=0,
                         dim_size=len(x))
        
        # import pdb;pdb.set_trace()
        
        batch_output = scatter(src=output,
                               index=batch.batch,
                               dim=0,
                               reduce='mean')

        # rot = o3.rand_matrix().to(pos)
        # pos_r = batch.pos @ rot.T
        
        # edge_src_r, edge_dst_r = radius_graph(x=pos_r, 
        #                                   r=self.max_radius)
        # edge_vec_r = pos_r[edge_src_r] - pos_r[edge_dst_r]
        # edge_length_r = edge_vec_r.norm(dim=1)

        # edge_length_embedded_r = soft_one_hot_linspace(
        #     x=edge_length_r,
        #     start=0.0,
        #     end=self.max_radius,
        #     number=self.number_of_basis,
        #     basis='smooth_finite',
        #     cutoff=True
        # )
        # edge_length_embedded_r = edge_length_embedded_r.mul(self.number_of_basis**0.5)
        # edge_weight_cutoff_r = soft_unit_step(10 * (1 - edge_length_r / self.max_radius))

        # edge_sh_r = self.sh(edge_vec_r)
        
        # key_weights_r = self.key_weights_fc(edge_length_embedded_r)
        # x_src_r = x[edge_src_r]
        # key_r = self.tp_key(x=x_src_r, 
        #                   y=edge_sh_r, 
        #                   weight=key_weights_r)
        
        # value_weights_r = self.value_weights_fc(edge_length_embedded_r)
        # value_r = self.tp_value(x=x_src_r, 
        #                       y=edge_sh_r, 
        #                       weight=value_weights_r)

        # query_dst_r = query[edge_dst_r]
        # query_key_r = self.tp_query_key(x=query_dst_r, 
        #                               y=key_r)
        
        # exp_r = edge_weight_cutoff_r[:, None] * query_key_r.exp() # numerator
        # z_r = scatter(src=exp_r, 
        #             index=edge_dst_r, 
        #             dim=0, 
        #             dim_size=len(x)) # denominator, default scatter reduce is sum
        # z_r[z_r == 0] = 1 # to avoid 0/0 when all the neighbors are exactly at the cutoff
        # alpha_r = exp_r / z_r[edge_dst_r]
        
        # attention_r = alpha_r.relu().sqrt()
        # attended_value_r = attention_r * value_r
        # output_r = scatter(src=attended_value_r,
        #                  index=edge_dst_r, 
        #                  dim=0,
        #                  dim_size=len(x))

        return batch_output
    