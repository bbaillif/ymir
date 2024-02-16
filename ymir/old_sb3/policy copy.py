import numpy as np
import torch
import gymnasium as gym

from typing import Any, Dict, List, Optional, Type, Union
from gymnasium.spaces import Space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.optim import Optimizer, Adam
from ymir.model import ComENetModel
from ymir.params import FEATURES_DIM, NET_ARCH
from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch

# Wrapper to be compatible with FeatureExtractor in SB3
class ComENetFeatureExtractor(BaseFeaturesExtractor):
    
    def __init__(self, 
                 observation_space: gym.Space, 
                 features_dim: int = FEATURES_DIM,
                 device='cuda') -> None:
        super().__init__(observation_space, features_dim)
        self.model = ComENetModel(features_dim=features_dim,
                                  device=device)
        
    def forward(self,
                features: torch.Tensor):
        
        batch_size, num_nodes, input_dim = features.shape
        
        data_list = []
        for env_feature in features:
            nan_features = torch.isnan(env_feature[:, 0]) # identify padding
            env_feature = env_feature[~nan_features] # remove padding
            x = env_feature[:, 0]
            x = x.long()
            pos = env_feature[:, 1:]
            data = Data(x=x,
                        pos=pos)
            data_list.append(data)
            
        # import pdb;pdb.set_trace()
        
        # Remove padding
        
        # features = features[~nan_features]
        
        # x = features[:, 0]
        # x = x.long()
        
        # pos = features[:, 1:]
        
        batch = Batch.from_data_list(data_list)
        
        atomic_contributions = self.model.forward(batch)
        
        unbatched_contributions = unbatch(atomic_contributions, batch=batch.batch)
        
        env_list = []
        for contrib in unbatched_contributions:
            env_list.append(contrib[-1]) # we assume the new frag environement is the last one per batch
            # might need to be modified to fit the Aggregation paradigm in torch geometric
        
        env_vector = torch.stack(env_list)
        
        assert env_vector.shape == (batch_size, FEATURES_DIM) 
        
        # import pdb;pdb.set_trace()
        
        return env_vector
    
    
class FragmentBuilderPolicy(MaskableActorCriticPolicy):
    
    def __init__(self, 
                 observation_space: Space, 
                 action_space: Space, 
                 lr_schedule: Schedule, 
                 net_arch: Union[List[int], Dict[str, List[int]], None] = NET_ARCH, 
                 activation_fn: type[nn.Module] = nn.Tanh, 
                 ortho_init: bool = True, 
                 features_dim: int = FEATURES_DIM,
                 features_extractor_kwargs: Union[Dict[str, Any], None] = None, 
                 share_features_extractor: bool = True, 
                 normalize_images: bool = True, 
                 optimizer_class: type[Optimizer] = Adam, 
                 optimizer_kwargs: Union[Dict[str, Any], None] = None):
        
        features_extractor_class = ComENetFeatureExtractor
        features_extractor_kwargs = {'features_dim': features_dim}
        
        super().__init__(observation_space, 
                         action_space, 
                         lr_schedule, 
                         net_arch, 
                         activation_fn, 
                         ortho_init, 
                         features_extractor_class, 
                         features_extractor_kwargs, 
                         share_features_extractor, 
                         normalize_images, 
                         optimizer_class, 
                         optimizer_kwargs)