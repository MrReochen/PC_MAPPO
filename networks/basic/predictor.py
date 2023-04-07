import torch.nn as nn
from .util import init, get_clones

class OutLayer(nn.Module):
    def __init__(self, output_dim, hidden_size, use_orthogonal, use_ReLU):
        super(OutLayer, self).__init__()
        self._layer_N = 1

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(hidden_size, output_dim)), active_func, nn.LayerNorm(output_dim))
        

    def forward(self, x):
        x = self.fc1(x)
        return x


class PredictorLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(PredictorLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class PredictorNet(nn.Module):
    def __init__(self, args, obs_shape, action_space, cent_obs_shape, use_obfuscator=False, cat_self=True, attn_internal=False):
        super(PredictorNet, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = 3
        self.hidden_size = args.hidden_size

        if type(obs_shape) == list:
            obs_dim = obs_shape[0]
        else:
            obs_dim = obs_shape
        cent_obs_dim = cent_obs_shape[0]
        action_dim = action_space.n

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim + action_dim)

        self.mlp = PredictorLayer(obs_dim + action_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)
        self.out = OutLayer(args.predict_dim, self.hidden_size, self._use_orthogonal, self._use_ReLU)
        
    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)
        x = self.out(x)

        return x