"""Environments + agents for 2-armed bandit task."""
# pylint: disable=line-too-long
from typing import Callable, NamedTuple, Tuple, Union, Optional, List

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import norm


from . import rnn_utils
DatasetRNN = rnn_utils.DatasetRNN


# Setup so that plots will look nice
small = 15
medium = 18
large = 20
plt.rc('axes', titlesize=large)
plt.rc('axes', labelsize=medium)
plt.rc('xtick', labelsize=small)
plt.rc('ytick', labelsize=small)
plt.rc('legend', fontsize=small)
plt.rc('figure', titlesize=large)
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'

###################################
# CONVENIENCE FUNCTIONS.          #
###################################


def _check_in_0_1_range(x, name):
  if not (0 <= x <= 1):
    raise ValueError(
        f'Value of {name} must be in [0, 1] range. Found value of {x}.')


###################################
# GENERATIVE FUNCTIONS FOR AGENTS #
###################################


class AgentQ:
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

  def __init__(
      self,
      alpha: float = 0.2,
      beta: float = 3.,
      n_actions: int = 2,
      forgetting_rate: float = 0.,
      perseveration_bias: float = 0.):
    """Update the agent after one step of the task.

    Args:
      alpha: scalar learning rate
      beta: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      forgetting_rate: rate at which q values decay toward the initial values (default=0)
      perseveration_bias: rate at which q values move toward previous action (default=0)
    """
    self._prev_choice = None
    self._alpha = alpha
    self._beta = beta
    self._n_actions = n_actions
    self._forgetting_rate = forgetting_rate
    self._perseveration_bias = perseveration_bias
    self._q_init = 0.5
    self.new_sess()

    _check_in_0_1_range(alpha, 'alpha')
    _check_in_0_1_range(forgetting_rate, 'forgetting_rate')

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init * np.ones(self._n_actions)

  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    decision_variable = self._beta * self._q
    if self._prev_choice is not None:
      decision_variable[self._prev_choice] += self._perseveration_bias
    choice_probs = np.exp(decision_variable) / np.sum(np.exp(decision_variable))
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(self._n_actions, p=choice_probs)
    return choice

  def update(self,
             choice: int,
             reward: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    # Decay q-values toward the initial value.
    self._q = ((1-self._forgetting_rate) * self._q +
               self._forgetting_rate * self._q_init)

    self._prev_choice = choice
    # # Apply perseveration and anti-perseveration of chosen action.
    # onehot_choice = np.eye(self._n_actions)[choice]
    # self._q = 0.5 * (
    #     (2-self._perseveration_rate-self._anti_perseveration_rate) * self._q +
    #     self._perseveration_rate * onehot_choice + 
    #     self._anti_perseveration_rate * (1-onehot_choice))

    # Update chosen q for chosen action with observed reward.
    self._q[choice] = (1 - self._alpha) * self._q[choice] + self._alpha * reward

  @property
  def q(self):
    # This establishes q as an externally visible attribute of the agent.
    # For agent = AgentQ(...), you can view the q values with agent.q; however,
    # you will not be able to modify them directly because you will be viewing
    # a copy.
    return self._q.copy()

class ThompsonAgent:
  """Thompson Sampling agent for the two-armed bandit task.
  """

  def __init__(
      self,
      innov_variance = 100,
      noise_variance = 10,
      n_states = 10,
      n_actions: int = 2.):
    """Update the agent after one step of the task.


    """
    ### my setup ###
    self.innov_variance = innov_variance
    self.noise_variance = noise_variance
    ################
    self._n_actions = n_actions
    self.n_states = n_states
    self.new_sess()




  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    #self.kalman_gain = np.zeros(self._n_actions)
    #self.post_variance = np.ones(self._n_actions) * 10
    #self.post_mean = np.zeros(self._n_actions)
    #self.V_t = 0
    #self.P_thompson = np.zeros(self._n_actions‚)
    self.V_t = np.zeros(self.n_states)

    self.P_a0_thompson = np.zeros(self.n_states)
    self.post_mean = np.zeros((self._n_actions, self.n_states))
    self.post_variance = np.ones((self._n_actions, self.n_states)) * 5
    self.kalman_gain = np.zeros((self._n_actions, self.n_states))
    print("new session in the agent started")

  def get_choice_probs(self, state) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""

    ### my code ###
    self.V_t[state] = self.post_mean[0][state] - self.post_mean[1][state]
    sigma2_1 = self.post_variance[0][state]  # Variance of arm 1
    sigma2_2 = self.post_variance[1][state]  # Variance of arm 2

    # Compute the standard deviation for the combined variance
    self.std_dev = np.sqrt(sigma2_1 + sigma2_2)

    # Calculate the probability P(a_t = 1)
    self.P_a0_thompson[state] = norm.cdf((self.V_t[state] / self.std_dev))
    #self.P_thompson[1] = 1 - self.P_thompson[0]
    ################
    return self.P_a0_thompson[state]

  def get_choice(self, state) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs(state)
    #choice = np.random.choice(self._n_actions, p=choice_probs)
    choice = 0 if np.random.rand() < choice_probs else 1

    return choice

  def update(self,
             choice: int,
             reward: int,
             state: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent.
    """

    if state < self.n_states - 1:

      for i in range(self._n_actions):
        if choice == i:

          self.kalman_gain[i][state] = self.post_variance[i][state] / (
                    self.post_variance[i][state] + self.noise_variance)
          # self.kalman_gain[i][state] = (self.post_variance[i] + self.innov_variance) / (self.post_variance[i] +
          # self.innov_variance + self.noise_variance)
          # else:
          # self.kalman_gain[i][state] = 0
          self.post_variance[i][state + 1] = (1 - self.kalman_gain[i][state]) * self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state] + self.kalman_gain[i][
            state] * (reward - self.post_mean[i][state])

        else:
          self.post_variance[i][state + 1] = self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state]


  @property
  def q(self):
    # This establishes q as an externally visible attribute of the agent.
    # For agent = AgentQ(...), you can view the q values with agent.q; however,
    # you will not be able to modify them directly because you will be viewing
    # a copy.
    # leave this for now, so that their code still works
    return self.post_mean.copy()

class UCBAgent:
  """Thompson Sampling agent for the two-armed bandit task.
  """

  def __init__(
      self,
      innov_variance = 100,
      noise_variance = 10,
      n_states = 10,
      n_actions: int = 2.):
    """Update the agent after one step of the task.


    """
    ### my setup ###
    self.innov_variance = innov_variance
    self.noise_variance = noise_variance
    self._n_actions = n_actions
    self.n_states = n_states
    self.new_sess()


  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    #self.kalman_gain = np.zeros(self._n_actions)
    #self.post_variance = np.ones(self._n_actions) * 10
    #self.post_mean = np.zeros(self._n_actions)
    #self.V_t = 0
    #self.P_thompson = np.zeros(self._n_actions‚)
    self.V_t = np.zeros(self.n_states)
    self.P_a0_ucb = np.zeros(self.n_states)
    self.post_mean = np.zeros((self._n_actions, self.n_states))
    self.post_variance = np.ones((self._n_actions, self.n_states)) * 5
    self.kalman_gain = np.zeros((self._n_actions, self.n_states))
    self.gamma = 4
    self.theta = 2

    print("new session in the agent started")

  def get_choice_probs(self, state) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""

    ### my code ###
    self.V_t[state] = self.post_mean[0][state] - self.post_mean[1][state]
    sigma1 = self.post_variance[0][state]  # Variance of arm 1
    sigma2 = self.post_variance[1][state]  # Variance of arm 2

    # Compute ucb probs
    self.P_a0_ucb[state] = norm.cdf((self.V_t[state] + self.gamma * (sigma1 - sigma2)) / self.theta)

    return self.P_a0_ucb[state]

  def get_choice(self, state) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs(state)
    #choice = np.random.choice(self._n_actions, p=choice_probs)
    choice = 0 if np.random.rand() < choice_probs else 1

    return choice

  def update(self,
             choice: int,
             reward: int,
             state: int):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent.
    """

    if state < self.n_states - 1:

      for i in range(self._n_actions):
        if choice == i:

          self.kalman_gain[i][state] = self.post_variance[i][state] / (
                    self.post_variance[i][state] + self.noise_variance)
          # self.kalman_gain[i][state] = (self.post_variance[i] + self.innov_variance) / (self.post_variance[i] +
          # self.innov_variance + self.noise_variance)
          # else:
          # self.kalman_gain[i][state] = 0
          self.post_variance[i][state + 1] = (1 - self.kalman_gain[i][state]) * self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state] + self.kalman_gain[i][
            state] * (reward - self.post_mean[i][state])

        else:
          self.post_variance[i][state + 1] = self.post_variance[i][state]
          self.post_mean[i][state + 1] = self.post_mean[i][state]


  @property
  def q(self):
    # This establishes q as an externally visible attribute of the agent.
    # For agent = AgentQ(...), you can view the q values with agent.q; however,
    # you will not be able to modify them directly because you will be viewing
    # a copy.
    # leave this for now, so that their code still works
    return self.post_mean.copy()


class AgentNetwork:
  """A class that allows running a pretrained RNN as an agent.

  Attributes:
    make_network: A Haiku function that returns an RNN architecture
    params: A set of Haiku parameters suitable for that architecture
  """

  def __init__(self,
               make_network: Callable[[], hk.RNNCore],
               params: hk.Params,
               n_actions: int = 2,
               state_to_numpy: bool = False):
    """Initialize the agent network.
    
    Args: 
      make_network: function that instantiates a callable haiku network object
      params: parameters for the network
      n_actions: number of permitted actions (default = 2)
    """
    self.n_actions = n_actions
    self._state_to_numpy = state_to_numpy
    self.noise_variance = 10

    self.V_t = np.zeros(n_actions)
    self.std_dev = np.zeros(n_actions)

    def _step_network(xs: np.ndarray,
                      state: hk.State) -> Tuple[np.ndarray, hk.State]:
      """Apply one step of the network.
      
      Args:
        xs: array containing network inputs
        state: previous state of the hidden units of the RNN model.
        
      Returns:
        y_hat: output of RNN
        new_state: state of the hidden units of the RNN model
      """
      core = make_network() #initialize the network
      y_hat, new_state = core(xs, state) #apply the network
      return y_hat, new_state

    def _get_initial_state() -> hk.State:
      """Get the initial state of the hidden units of RNN model."""
      core = make_network()
      state = core.initial_state(1)
      return state

    key = jax.random.PRNGKey(0) # Random key for initialization
    model = hk.transform(_step_network) # transformation required because jax is a functional programming library
    state = hk.transform(_get_initial_state)

    self._initial_state = state.apply(params, key) # forward inference
    self._model_fun = jax.jit(
        lambda xs, state: model.apply(params, key, xs, state)) #translates the function into a computation graph for improved efficiency
    self._xs = np.zeros((1, 2))
    self._n_actions = n_actions
    self.new_sess()

  def new_sess(self):
    """Reset the network for the beginning of a new session."""
    self._state = self._initial_state
    self.post_mean = np.zeros(self.n_actions)
    self.post_variance = np.ones(self.n_actions) * 10
    self.kalman_gain = np.zeros(self.n_actions)

  def get_choice_probs(self) -> np.ndarray:
    """Predict the choice probabilities as a softmax over output logits."""
    output_logits, _ = self._model_fun(self._xs, self._state) #forward inference (on optimized apply function)
    output_logits = np.array(output_logits)
    output_logits = output_logits[0][:self._n_actions]
    choice_probs = np.exp(output_logits) / np.sum(
        np.exp(output_logits)) #inverse the softmax to get probabilities
    return choice_probs

  def get_choice(self) -> Tuple[int, np.ndarray]:
    """Sample choice."""
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(self._n_actions, p=choice_probs)
    return choice

  def update(self, choice: int, reward: int):
    try:
      self._xs = np.array([[choice, reward]])
      _, new_state = self._model_fun(self._xs, self._state)
      if self._state_to_numpy:
        self._state = np.array(new_state)
      else:
        self._state = new_state
    except:
      import pdb; pdb.set_trace()

  def update_kalman(self, choice: int, reward):
    # Update for the selected action
    kalman_gain = self.post_variance[choice] / (self.post_variance[choice] + self.noise_variance)

    # Update posterior mean and variance for the selected action
    self.post_mean[choice] = self.post_mean[choice] + kalman_gain * (reward - self.post_mean[choice])
    self.post_variance[choice] = (1 - kalman_gain) * self.post_variance[choice]


class VanillaAgentQ(AgentQ):
  """This agent is a wrapper of AgentQ with only alpha and beta parameters."""

  def __init__(self, alpha: float, beta: float, n_actions: int = 2):
    super(VanillaAgentQ, self).__init__(
        alpha, beta, n_actions=n_actions, forgetting_rate=0.)


class MysteryAgentQ(AgentQ):
  """Don't look at this agent if you want to do the exercises!"""

  def __init__(self, alpha: float, beta: float, mystery_param: float = -0.1, n_actions: int = 2):
    super(MysteryAgentQ, self).__init__(
        alpha, beta, n_actions=n_actions, perseveration_bias=mystery_param)


class ExtraMysteryAgentQ(AgentQ):
  """Don't look at this agent if you want to do the exercises!"""

  def __init__(self, alpha: float, beta: float, mystery_param=0.5, n_actions: int = 2):
    _check_in_0_1_range(mystery_param, 'mystery_param')
    super(ExtraMysteryAgentQ, self).__init__(
        alpha, beta, n_actions=n_actions, forgetting_rate=mystery_param)


################
# ENVIRONMENTS #
################


class EnvironmentBanditsFlips:
  """Env for 2-armed bandit task with with reward probs that flip in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.8,
      reward_prob_low: float = 0.2,
  ):
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    # Choose a random block to start in
    self._block = np.random.binomial(1, 0.5)
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self._block = 1 - self._block
    # Set the reward probabilites
    if self._block == 1:
      self.reward_probs = [self._reward_prob_high, self._reward_prob_low]
    else:
      self.reward_probs = [self._reward_prob_low, self._reward_prob_high]

  def step(self, choice):
    """Step the model forward given chosen action."""
    # Choose the reward probability associated with agent's choice
    reward_prob_trial = self.reward_probs[choice]

    # Sample a reward with this probability
    reward = float(np.random.binomial(1, reward_prob_trial))

    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    return float(reward)

  @property
  def n_actions(self) -> int:
    return 2


class EnvironmentBanditsDrift:
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and 1. On each
  trial, gaussian random noise is added to each.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_actions: number of actions available
  """

  def __init__(
      self,
      sigma: float,
      n_actions: int = 2,
      ):
    """Initialize the environment."""
    # Check inputs
    if sigma < 0:
      msg = f'Argument sigma but must be greater than 0. Found: {sigma}.'
      raise ValueError(msg)

    # Initialize persistent properties
    self._sigma = sigma
    self._n_actions = n_actions

    # Sample new reward probabilities
    self._new_sess()

  def _new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = np.random.rand(self._n_actions)

  def step(self, choice: int) -> int:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """
    # Check inputs
    if choice not in range(self._n_actions):
      msg = (
          f'Found value for choice of {choice}, but must be in '
          f'{list(range(self._n_actions))}')
      raise ValueError(msg)

    # Sample reward with the probability of the chosen side
    reward = np.random.rand() < self._reward_probs[choice]

    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self._sigma, size=self._n_actions)
    self._reward_probs += drift

    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.clip(self._reward_probs, 0, 1)

    return reward

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self._n_actions

class GershmanBandit:
  """Environment for a drifting two-armed bandit task.

  Bandit design after Gershman (2018) - each block has a mean reward value

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_actions: number of actions available
  """

  def __init__(
          self,
          innov_variance = 100,
          noise_variance = 10,
          n_actions: int = 2,
  ):
    """Initialize the environment."""


    # Initialize persistent properties
    self._n_actions = n_actions
    self.innov_variance = innov_variance,
    self.noise_variance = noise_variance

    # Sample new reward probabilities
    self._new_sess()

  def _new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._block_mean = np.random.normal(0, np.sqrt(self.innov_variance), self._n_actions)
    print("new session started")

  def step(self, choice: int) -> int:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent.

    """
    # Check inputs
    if choice not in range(self._n_actions):
      msg = (
        f'Found value for choice of {choice}, but must be in '
        f'{list(range(self._n_actions))}')
      raise ValueError(msg)

    # define reward for each step
    reward = np.random.normal(self._block_mean, np.sqrt(self.noise_variance), self._n_actions)

    # Sample reward with the probability of the chosen side
    received_reward = int(reward[choice])

    return received_reward

  @property
  def reward_probs(self) -> np.ndarray:
    return self._block_mean.copy()

  @property
  def n_actions(self) -> int:
    return self._n_actions


class BanditSession(NamedTuple):
  """Holds data for a single session of a bandit task."""
  choices: np.ndarray
  rewards: np.ndarray
  timeseries: np.ndarray
  n_trials: int

class KalmanData(NamedTuple):
  """Holds data for a single session of a bandit task."""
  post_mean: np.ndarray
  post_variance: np.ndarray
  V_t: np.ndarray
  n_trials: int

Agent = Union[AgentQ, AgentNetwork, ThompsonAgent, UCBAgent]
Environment = Union[EnvironmentBanditsFlips, EnvironmentBanditsDrift, GershmanBandit]


def run_experiment(agent: Agent,
                   environment: Environment,
                   n_trials: int) -> BanditSession:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_trials: The number of steps in the session you'd like to generate

  Returns:
    experiment: A BanditSession holding choices and rewards from the session
  """
  choices = np.zeros(n_trials)
  rewards = np.zeros(n_trials)
  reward_probs = np.zeros((n_trials, environment.n_actions))

  post_mean = np.zeros((n_trials, environment.n_actions))
  post_variance = np.zeros((n_trials, environment.n_actions))
  V_t = np.zeros((n_trials, environment.n_actions))


  for trial in np.arange(n_trials):
    # First record environment reward probs
    reward_probs[trial] = environment.reward_probs
    # First agent makes a choice
    choice = agent.get_choice(trial)
    # Then environment computes a reward
    reward = environment.step(choice)
    # Finally agent learns
    if agent != AgentNetwork:
      agent.update(choice, reward, trial)
    # Log choice and reward
    choices[trial] = choice
    rewards[trial] = reward
    if agent == AgentNetwork:
      agent.update(choice, reward)
      agent.update_kalman(choice, reward)
      post_mean[trial][choice] = agent.post_mean
      post_variance[trial][choice] = agent.post_variance
      V_t[trial] = agent.V_t

  experiment = BanditSession(n_trials=n_trials,
                             choices=choices,
                             rewards=rewards,
                             timeseries=reward_probs)
  kalman = KalmanData(post_mean=post_mean, post_variance=post_variance, V_t=V_t, n_trials=n_trials)
  if agent == AgentNetwork:
    return experiment, kalman
  else:
    return experiment


def plot_session(choices: np.ndarray,
                 rewards: np.ndarray,
                 timeseries: np.ndarray,
                 timeseries_name: str,
                 labels: Optional[List[str]] = None,
                 fig_ax: Optional = None):
  """Plot data from a single behavioral session of the bandit task.

  Tracks whether rewards from choices improve over time.
  """

  # Identify choices
  choose_high = choices == 1
  choose_low = choices == 0

  # Compare current reward with previous reward
  rewards_diff = np.diff(rewards, prepend=rewards[0])
  improved_reward = rewards_diff > 0
  decreased_reward = rewards_diff < 0

  """  improved_reward = rewards > 0
  decreased_reward = rewards < 0"""

  # Determine y-axis placement for markers
  y_high = np.max(timeseries) + 1
  y_low = np.min(timeseries) - 1

  # Make the plot
  if fig_ax is None:
    fig, ax = plt.subplots(1, figsize=(10, 4))
  else:
    fig, ax = fig_ax

  # Plot timeseries
  if labels is not None:
    if timeseries.ndim == 1:
      timeseries = timeseries[:, None]
    if len(labels) != timeseries.shape[1]:
      raise ValueError('labels length must match timeseries.shape[1].')
    for i in range(timeseries.shape[1]):
      ax.plot(timeseries[:, i], label=labels[i])
    ax.legend(bbox_to_anchor=(1, 1))
  else:
    ax.plot(timeseries)

  # Place markers for improvement or decrease in reward
  ax.scatter(
    np.argwhere(choose_high & improved_reward),
    y_high * np.ones(np.sum(choose_high & improved_reward)),
    color='green', marker='^', label="High Choice Improved"
  )
  ax.scatter(
    np.argwhere(choose_high & decreased_reward),
    y_high * np.ones(np.sum(choose_high & decreased_reward)),
    color='red', marker='v', label="High Choice Decreased"
  )
  ax.scatter(
    np.argwhere(choose_low & improved_reward),
    y_low * np.ones(np.sum(choose_low & improved_reward)),
    color='green', marker='^', label="Low Choice Improved"
  )
  ax.scatter(
    np.argwhere(choose_low & decreased_reward),
    y_low * np.ones(np.sum(choose_low & decreased_reward)),
    color='red', marker='v', label="Low Choice Decreased"
  )

  ax.set_xlabel('Trial')
  ax.set_ylabel(timeseries_name)
  ax.set_title("Session Choices and Reward Changes")
  ax.legend(loc="upper right")


def create_dataset(agent_cls: Callable[..., Agent],
                   env_cls: Callable[..., Environment],
                   n_trials_per_session: int,
                   n_sessions: int,
                   agent_kwargs: Optional[dict] = None,
                   env_kwargs: Optional[dict] = None,
                   batch_size: Optional[int] = None):
  """Generates a behavioral dataset from a given agent and environment.

  Args:
    agent: An agent object to generate choices
    environment: An environment object to generate rewards
    n_trials_per_session: The number of trials in each behavioral session to
      be generated
    n_sessions: The number of sessions to generate
    batch_size: The size of the batches to serve from the dataset. If None, 
      batch_size defaults to n_sessions

  Returns:
    A DatasetRNN object suitable for training RNNs.
    An experliment_list with the results of (simulated) experiments
  """
  xs = np.zeros((n_trials_per_session, n_sessions, 2))
  ys = np.zeros((n_trials_per_session, n_sessions, 1))
  experiment_list = []

  # Default kwargs if none are provided
  agent_kwargs = agent_kwargs or {}
  env_kwargs = env_kwargs or {}

  for sess_i in np.arange(n_sessions):
    # initialize agent and environment

    agent = agent_cls(**agent_kwargs)
    environment = env_cls(**env_kwargs)

    experiment = run_experiment(agent, environment, n_trials_per_session)
    experiment_list.append(experiment)
    prev_choices = np.concatenate(([0], experiment.choices[0:-1]))
    prev_rewards = np.concatenate(([0], experiment.rewards[0:-1]))
    xs[:, sess_i] = np.swapaxes(
        np.concatenate(([prev_choices], [prev_rewards]), axis=0), 0, 1)
    ys[:, sess_i] = np.expand_dims(experiment.choices, 1)

  dataset = DatasetRNN(xs, ys, batch_size)
  return dataset, experiment_list


###############
# DIAGNOSTICS #
###############


def show_valuemetric(experiment_list, label=None):
  """Plot value metric over time from data in experiment_list."""
  if experiment_list is None:
      print(('Skipping experiment because None value was found for experiment_lists.\n'
             'This is usually the case when using imported real data.'))
      return

  reward_prob_bins = np.linspace(-1, 1, 50)
  n_left = np.zeros(len(reward_prob_bins)-1)
  n_right = np.zeros(len(reward_prob_bins)-1)

  for sessdata in experiment_list:
    reward_prob_diff = sessdata.timeseries[:, 0] - sessdata.timeseries[:, 1]
    for reward_prob_i in range(len(n_left)):
      trials_in_bin = np.logical_and(
          (reward_prob_bins[reward_prob_i] < reward_prob_diff) ,
          (reward_prob_diff < reward_prob_bins[reward_prob_i+1]))
      n_left[reward_prob_i] += np.sum(
          np.logical_and(trials_in_bin, sessdata.choices == 0.))
      n_right[reward_prob_i] += np.sum(
          np.logical_and(trials_in_bin, sessdata.choices == 1.))

  choice_probs = n_left / (n_left + n_right)

  xs = reward_prob_bins[:-1] - (reward_prob_bins[1]-reward_prob_bins[0])/2
  ys = choice_probs
  plt.plot(xs, ys, label=label)
  plt.ylim((0, 1))
  plt.xlabel('Difference in Reward Probability (left - right)')
  plt.ylabel('Proportion of Leftward Choices')


def show_total_reward_rate(experiment_list):
  if experiment_list is None:
      print('Skipping showing reward rate for Non experiment_list')
      return
  rewards = 0
  trials = 0

  for sessdata in experiment_list:
    rewards += np.sum(sessdata.rewards)
    trials += sessdata.n_trials

  reward_rate = 100*rewards/trials
  print(f'Total Reward Rate is: {reward_rate:0.3f}%')


################################
# FITTING FUNCTIONS FOR AGENTS #
################################


class HkAgentQ(hk.RNNCore):
  """Vanilla Q-Learning model, expressed in Haiku.

  Updates value of chosen action using a delta rule with step-size param alpha. 
  Does not update value of the unchosen action.
  Selects actions using a softmax decision rule with parameter Beta.
  """

  def __init__(self, n_cs=4):
    super(HkAgentQ, self).__init__()

    # Haiku parameters
    alpha_unsigmoid = hk.get_parameter(
        'alpha_unsigmoid', (1,),
        init=hk.initializers.RandomUniform(minval=-1, maxval=1),
    )
    beta = hk.get_parameter(
        'beta', (1,), init=hk.initializers.RandomUniform(minval=0, maxval=2)
    )

    # Local parameters
    self.alpha = jax.nn.sigmoid(alpha_unsigmoid)
    self.beta = beta
    self._q_init = 0.5

  def __call__(self, inputs: jnp.array, prev_state: jnp.array):
    prev_qs = prev_state

    choice = inputs[:, 0]  # shape: (batch_size, 1)
    reward = inputs[:, 1]  # shape: (batch_size, 1)

    choice_onehot = jax.nn.one_hot(choice, num_classes=2)  # shape: (batch_size, 2)
    chosen_value = jnp.sum(prev_qs * choice_onehot, axis=1)  # shape: (batch_size)
    deltas = reward - chosen_value  # shape: (batch_size)
    new_qs = prev_qs + self.alpha * choice_onehot * jnp.expand_dims(deltas, -1)

    # Compute output logits
    choice_logits = self.beta * new_qs

    return choice_logits, new_qs

  def initial_state(self, batch_size):
    values = self._q_init * jnp.ones([batch_size, 2])  # shape: (batch_size, n_actions)
    return values



