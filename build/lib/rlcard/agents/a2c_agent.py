''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 Matthew Judell
Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from copy import deepcopy

from rlcard.utils.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

DEBUG = os.environ.get('RL_PRINT_SETTING', 'False') == 'True'

class A2CAgent(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 discount_factor=0.99,
                 num_actions=2,
                 state_shape=None,
                 pattern_shape=[None],
                 use_pattern=False,
                 train_every=1,
                 actor_mlp_layers=None,
                 critic_mlp_layers=None,
                 learning_rate=0.00005,
                 eval_with='stochastic',
                 device=None,
                 save_path=None,
                 save_every=float('inf'),):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            discount_factor (float): Gamma discount factor
            evaluate_every (int): Evaluate every N steps
            num_actions (int): The number of the actions
            state_space (list): The space of the state vector
            train_every (int): Train the network every X steps.
            actor_mlp_layers (list): The layer number and the dimension of each layer in MLP
            critic_mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent.
            eval_with (str): in eval, actor will be 'stochastic' or 'deterministic'
            device (torch.device): whether to use the cpu or gpu
            save_path (str): The path to save the model checkpoints
            save_every (int): Save the model every X training steps
        '''
        self.use_raw = False
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.train_every = train_every
        self.eval_with = eval_with
        self.state_shape = state_shape
        self.pattern_shape = pattern_shape
        self.use_pattern = use_pattern

        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # Create estimators
        if self.use_pattern:
            assert len(state_shape) == len(pattern_shape)
            obs_shape = [s_dim + p_dim for s_dim, p_dim in zip(state_shape, pattern_shape)]
        else:
            obs_shape = state_shape
        self.actor = Actor(
            num_actions=num_actions, learning_rate=learning_rate, state_shape=obs_shape, 
            mlp_layers=actor_mlp_layers, device=self.device
        )
        self.critic = Critic(
            num_actions=1, learning_rate=learning_rate, state_shape=obs_shape,
            mlp_layers=critic_mlp_layers, device=self.device
        )

        # Create replay memory
        self.memory = Memory()
        
        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every
    
    def preprocess_obs(self, state):
        if self.use_pattern:
            return np.concatenate((state['obs'], state['pattern']), axis=0)
        else:
            return state['obs']
    
    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(
            self.preprocess_obs(state), action, reward, 
            self.preprocess_obs(next_state), 
            list(next_state['legal_actions'].keys()), done)
        self.total_t += 1
        if done and len(self.memory) > 10:
            self.train()

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        obs = self.preprocess_obs(state)
        action_idx, _ = self.actor.predict_nograd(obs, list(state['legal_actions'].keys()))
        return action_idx

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        obs = self.preprocess_obs(state)
        
        if DEBUG:
            print(f"DEBUG:")
            print(f"- Obs shape: {obs.shape}")
            print(f"- Use pattern: {self.use_pattern}")

        # actor
        action_idx, greedy_action_idx = self.actor.predict_nograd(obs, list(state['legal_actions'].keys()))
        if self.eval_with == "stochastic":
            action = action_idx
        else:
            action = greedy_action_idx

        # critic
        state_value = self.critic.predict_nograd(obs)
                
        info = {}
        info['state_value'] = state_value
        
        if DEBUG:
            print(f"- Value: {state_value}")

        return action, info

    def train(self):
        ''' Train the network
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # calculate TD(0) return
        return_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * \
            self.critic.net(torch.from_numpy(next_state_batch).float().to(self.device)).detach().cpu().numpy()

        # update critic
        critic_loss = self.critic.update(state_batch, return_batch)

        # update actor
        state_value_batch = self.critic.net(
            torch.from_numpy(state_batch).float().to(self.device)
        ).detach().cpu().numpy()
        advantage_batch = return_batch - state_value_batch.squeeze(-1)
        actor_loss = self.actor.update(state_batch, action_batch, advantage_batch)

        print('\rINFO - Step {}, actor-loss: {}, critic-loss: {}'.format(self.total_t, actor_loss, critic_loss), end='')

        # reset memory
        self.memory.reset()

        # save checkpoint
        self.train_t += 1
        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately, 
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        self.device = device
        self.actor.device = device
        self.critic.device = device

    def checkpoint_attributes(self):
        '''
        Return the current checkpoint attributes (dict)
        Checkpoint attributes are used to save and restore the model in the middle of training
        Saves the model state dict, optimizer state dict, and all other instance variables
        '''
        
        return {
            'agent_type': 'A2CAgent',
            'actor': self.actor.checkpoint_attributes(),
            'critic': self.critic.checkpoint_attributes(),
            'state_shape': self.state_shape,
            'pattern_shape': self.pattern_shape,
            'use_pattern': self.use_pattern,
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'discount_factor': self.discount_factor,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'eval_with': self.eval_with,
            'device': self.device,
            'save_path': self.save_path,
            'save_every': self.save_every
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        '''
        Restore the model from a checkpoint
        
        Args:
            checkpoint (dict): the checkpoint attributes generated by checkpoint_attributes()
        '''
        
        print("\nINFO - Restoring model from checkpoint...")
        agent_instance = cls(
            discount_factor=checkpoint['discount_factor'],
            num_actions=checkpoint['num_actions'], 
            state_shape=checkpoint['state_shape'],
            pattern_shape=checkpoint['pattern_shape'],
            use_pattern=checkpoint['use_pattern'],
            train_every=checkpoint['train_every'],
            actor_mlp_layers=checkpoint['actor']['mlp_layers'],
            critic_mlp_layers=checkpoint['critic']['mlp_layers'],
            learning_rate=checkpoint['actor']['learning_rate'],
            eval_with=checkpoint['eval_with'],
            device=checkpoint['device'],
            save_path=checkpoint['save_path'],
            save_every=checkpoint['save_every'],
        )
        
        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']
        
        agent_instance.actor = Actor.from_checkpoint(checkpoint['actor'])
        agent_instance.critic = Critic.from_checkpoint(checkpoint['critic'])
        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])

        return agent_instance
                     
    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
        ''' Save the model checkpoint (all attributes)

        Args:
            path (str): the path to save the model
            filename(str): the file name of checkpoint
        '''
        torch.save(self.checkpoint_attributes(), os.path.join(path, filename))


class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        net = MLPNetwork(num_actions, state_shape, mlp_layers)
        net = net.to(self.device)
        self.net = net
        self.net.eval()

        # initialize the weights using Xavier init
        for p in self.net.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint
        '''
        return {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'device': self.device
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint
        '''
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
            device=checkpoint['device']
        )
        
        estimator.net.load_state_dict(checkpoint['net'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer'])
        return estimator

class Actor(Estimator):
    def predict_nograd(self, s, legal_actions):
        '''
        Returns:
          state_value (torch.tensor)
        '''
        with torch.no_grad():
            s = np.expand_dims(s, 0)
            s = torch.from_numpy(s).float().to(self.device)
            logits = self.net(s)[0]
            masked_logits = -torch.inf * torch.ones(self.num_actions).float().to(self.device)
            masked_logits[legal_actions] = logits[legal_actions]
            log_action_probs = F.log_softmax(masked_logits, dim=-1).cpu().numpy()
        action_probs = np.exp(log_action_probs)
        action_idx = np.random.choice(np.arange(self.num_actions), p = action_probs)
        greedy_action_idx = np.argmax(action_probs)
        if DEBUG:
            print(f"- Action probs: {action_probs}")
            print(f"- Sampled/best action: {action_idx} / {greedy_action_idx}")
        return action_idx, greedy_action_idx

    def update(self, state_batch, action_batch, advantage_batch):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.net.train()

        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        advantage_batch = torch.from_numpy(advantage_batch).float().to(self.device)

        logits = self.net(state_batch)
        log_action_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_action_probs, dim=-1, index=action_batch.unsqueeze(-1)).squeeze(-1)
        batch_loss = (-log_probs * advantage_batch.detach()).mean()

        batch_loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=10.0)
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.net.eval()

        return batch_loss

class Critic(Estimator):
    def predict_nograd(self, s):
        '''
        Returns:
          state_value (torch.tensor)
        '''
        with torch.no_grad():
            s = np.expand_dims(s, 0)
            s = torch.from_numpy(s).float().to(self.device)
            output = self.net(s).cpu().numpy()[0]
        return output

    def update(self, state_batch, return_batch):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.net.train()

        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        return_batch = torch.from_numpy(return_batch).float().to(self.device)

        state_value_batch = self.net(state_batch)
        batch_loss = (return_batch - state_value_batch).pow(2).mean()

        batch_loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=10.0)
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.net.eval()

        return batch_loss

class MLPNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(MLPNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)

class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self):
        ''' Initialize
        '''
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []
    
    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = tuple(zip(*self.memory))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''
        
        return {
            'memory': self.memory
        }
            
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' 
        Restores the attributes from the checkpoint
        
        Args:
            checkpoint (dict): the checkpoint dictionary
            
        Returns:
            instance (Memory): the restored instance
        '''
        
        instance = cls()
        instance.memory = checkpoint['memory']
        return instance
