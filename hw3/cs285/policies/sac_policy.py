from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
from torch import distributions
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_alpha)
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        obs = ptu.from_numpy(obs)
        if sample:
            action = self.forward(obs).sample()
        else:
            action = self.forward(obs).mean
        action_min, action_max = self.action_range
        action = torch.clamp(action, action_min, action_max)
        action = ptu.to_numpy(action)
        return action[None]

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file
        mean = self.mean_net(observation)
        logstd = torch.tanh(self.logstd) # TODO: Should we do this?
        log_std_min, log_std_max = self.log_std_bounds
        logstd = torch.clamp(logstd, log_std_min, log_std_max)
        std = torch.exp(logstd)
        action_distribution = sac_utils.SquashedNormal(mean, std)
        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        # Actor loss
        obs = ptu.from_numpy(obs)

        action_distribution = self.forward(obs)
        action = action_distribution.rsample()
        q_s = critic.forward(obs, action)
        q_min = torch.min(q_s, dim=1).values
        log_pis = action_distribution.log_prob(action).sum(axis=1)
        actor_loss = self.alpha * log_pis - q_min
        actor_loss = actor_loss.mean()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Alpha loss
        alpha_loss = -self.alpha*(log_pis+self.target_entropy).detach()
        alpha_loss = alpha_loss.mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.alpha