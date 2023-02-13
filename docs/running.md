# Data Information

The ticker data is given in the file `capvol100.csv`. It is worth noting that the user can define the stocks (any stocks can be used) by sorting them based on the *gap*, where the gap is defined as the difference between the opening price on the day N and the closing price on day N-1 for a particular stock.

# Code

# `aiagentbase`

## Base classes for an **AI Agent** having the following architecture:
1.1 An **AI Agent** operates in a '**World**' that calls it via '**action<-act(state)**' requests and '**reward(reward)**' intimations. (Worlds are typically wrappers around traditional RL-environments; Worlds can also wrap supervised learning tasks.) The goal of the AI Agent is to maximise *long-term steady-state average reward*. Periodically the World may also update the Agent regarding the completion of an *episode* (e.g. an RL episode or completion of epoch).

1.2 An Agent has **Controller**, **Perception**, **Memory** and **Actor** components (In line with Lecun's "Archicture of an Autonomous AI Agent". World Model to be added later.) The Memory contains a Perceptual Memory as well as a State-Action-Reward memory. The Actor includes a **Model**. A Model includes a **Network** and **Trainer** (class that handles publishes training procedure(s) for the Network). Overall orchestration of all components including the Agent's public interface is handled by the Controller. Further, the learning schedule, to train the Network, be it done only intitally, periodically, or continually online, is decided by the Controller. 

1.3 Each component of an Agent may be customised for a specific World by inheriting from the default base class for that component. Agent's *begin* method indicates that a new episode/epoch is starting and resets/increments the time/ep counters (see below); the *clear* method clears (e.g. removes all storage) from applicable components.

## Controller

2.1 Control flow goes as follows: Agent tracks a *time* counter and a list *ep*; the latter tracks episodes/epochs and is set to the first time counter for each epoch/episode e.g. ep[e]=starting time of epoch/epsode e. World calls *act*(world-state) on Agent, which is routed to Controller's *act*. The incoming *world-state* is mapped to a percept using the *perceive_state* function published by the Perception module, and stored in the perceptual memory (via Memory's *add_percept*, against the current *time*). 

2.2 The Actor reads from the perceptual memory and creates an actor-state by processing the current percept (and possibly also using prior rewards and actions, e.g. for meta-RL). The Actor also updates the state-action-reward memory with the previous time's percept after mapping it to an actor-state.

2.3 The Actor calls its Model to decide the action to return. Before returning the action, it is stored in the perceptual memory; a new entry is also created in the state-action-reward memory with the current actor-state and action. Also, the action is mapped to a world_action using the Perception component's *action_to_world* function.

2.4 Before completing the *act* (or *reward*) flow, Actor checks to see if any periodic or online training is needed, and updates the *training* flag accordingly. It also updates the *time* counter.

2.5 On receiving a *world_reward* from the World, the Actor passes it to the Controller that extracts information using Perception's *perceive_reward* function.  These are stored in the perceptual memory (for the prior time step, since by now the Actor's time step has been update as soon as its action was completed) as well as appended to the latest entry (prev time step) of the state-action-reward memory. Note: e.g. additional information might include, in addition to state, labels in case of a supervised learning scenario. 

## Memory

3.1 ***Memory*** stores are nested dictionaries indexed by *time*. Each entry is a dictionary with keys *typically* being *'percept','action','reward'* and *'state','action','reward','next_state'* for perceptual memory / state-action-reward memory respectively. However these may be extended for specifc kinds of worlds, e.g. supervised learning.

3.2 The default **Perception** class just copies world states/actions/rewards to actor states/actions/rewards. Should be subclassed for a given World.

3.3 The default **Actor** has no Model and returns a fixed action (can be set). It copies the percept from perceptual memory directly into the actor_state. This should be subclassed and/or method *percept_to_state* or *create_actor_state* overridden for a given World.

## Guidelines on overriding/augmenting base classes for world-specific agents

### Generic (On-policy) Reinforcement Learning Agent including Windowing and Meta-RL
Create a local environment called by a Monitor thread running within Agent, which implements or
re-uses an on-policy RL training procedure (such as PPO etc.). The env has an input and output queue. If a training flag is set, the Agent uses a ProxyModel place of its normal model to compute actions: The actor-state is placed in the env's input queue and an action is awaited from the env's output queue.

The monitor starts by calling the env.reset method that waits on the input queue to receive 
and then return a state. The monitor thread computes an action on the current state and calls
env.step(action), which places the action in the output queue and awaits a reward from the input
queue. After receiving a reward, step again waits for the next state on the input queue.
Once this is also received, both next stte and reward are returned.

The generic RL agent incorporates windowing, i.e., past win states are concatenated, and meta-RL via the $RL^2$ algorithm (metarl parameter). Note that if win>1 then use_memory has to be true as the window computation uses memory. Further metarl=True requires win>=2.

# `rlagents`

## `StratAgent`

The StratAgent class extends the AIAgent class. It defines an AI agent that can be used for trading strategies. It provides the interface for the tradeserver to control the agent and get the decisions for entry and exit.
### Attributes

`agent`: a boolean attribute indicating whether the agent is active.

`tidx`: an integer attribute representing the index of the current ticker.

`owner`: an attribute to store the reference to the owner object.

`use_alt_data`: a boolean attribute to indicate whether to use alternative data.

`use_memory`: a boolean attribute to indicate whether to use memory.

`logL`: a list to store the log data.
`action_space`: a spaces.Discrete object to define the action space. It has 3 possible actions.

`data_cols`: a list of columns to define the data.

`model_type`: a string attribute to store the model type (RL in this case).

`actor`: an attribute to store the reference to the actor object.

### Methods

* __init__: The constructor method initializes the attributes of the class. It also calls the constructor of the parent class (AIAgent) using the super() method. It sets the values for attributes such as agent, tidx, owner, logL, action_space, data_cols and model_type. It also sets the reference of the actor object to the self.actor attribute.
* percept_to_state: A method to convert the perceived state to the state. It takes in a dictionary perceived_state and returns the state.
* initialize: A method to initialize the values of the attributes agent, tidx, owner, use_memory, logL, action_space, data_cols and model_type.
* compress_alt_data: A method to compress the alternative data. It takes in a dictionary gdata and returns a dictionary gstateD.
* set_alt_data: A method to set the alternative data. It takes in a function alt_data_func and a boolean remote indicating whether the data should be obtained remotely. If remote is True, the function calls the remote function to get the alternative data. Otherwise, it directly calls the function. It sets the obtained data to the self.gdata attribute and calls the compress_alt_data method to get the compressed alternative data.
* act_on_entry: A method to check if the agent should act on entry. If self.owner is None or the status of the current ticker is not 'deployed', it returns True. Otherwise, it returns False.
* act_on_exit: A method to check if the agent should act on exit. If self.owner is None or the status of the current ticker is not 'active', it returns True. Otherwise, it returns False.
* check_entry_batch: A method to check the entry decisions for a batch of tickers. It takes in a dictionary dfD and returns a tuple of dictionaries decisionsD, stopD, and `target

## RandomStratAgent

The RandomStratAgent class is a subclass of StratAgent class and implements a random trading strategy. The class has the following attributes and methods:

### Attributes:

`logL`: list to store log data.

`rewL`: list to store reward data.

### Methods:


* __init__: This method initializes the RandomStratAgent object and sets the perception.perceive_reward attribute to self.perceive_reward method.

* call_model: This method is used to return a random action from the action space.

* perceive_reward: This method takes in the reward and returns it.


## RLStratAgentDyn

The RLStratAgentDyn class is a subclass of both RLAgent and StratAgent classes. The class implements a reinforcement learning based trading strategy. The class has the following attributes and methods:

### Attributes:

Some of the attributes are:
`use_alt_data`: a flag indicating whether to use alternate data or not.
`win`: a scalar value representing the window size.

### Methods:

* __init__: This method initializes the RLStratAgentDyn object and sets the following attributes:

    * action_space: a MultiDiscrete action space.
    * observation_space: a Box observation space with high and low limits set to infinity.
    * actor.percept_to_state: set to self.percept_to_state method.

    The method also calls the __init__ method of the superclass RLAgent.

* load_model: This method is used to load a model from the given filepath.

* percept_to_state: This method takes in the perceived state and returns the state tensor.

* clear_sar_memory: This method clears the SAR memory by resetting the state, action, and reward values.

* check_entry_batch: This method returns the entry decisions, stop values, and target values for a given dataframe.

* reward: This method takes in the reward and returns it after transforming it if required.

# `rlagents_train`

A script for training and running a RL based trading agent. The script does the following tasks:

* Importing the required libraries: pickle is used to serialize and deserialize data structures, so that they can be saved and loaded to disk.
* Initializing the data feed: There are two types of data feed - BackFeed and DataFeed. The BackFeed class is used to create a feed using data from a csv file. The DataFeed class is used to create a feed using an existing csv file. The script checks two parameters loadfeed and datafeed to determine which type of feed to use. If loadfeed is True, it loads an existing feed that has been serialized to disk using pickle. If loadfeed is False, it creates a new feed.
* Initializing the reinforcement learning agent: The script creates an instance of the RLStratAgentDyn class. This class is used to implement a reinforcement learning-based trading agent. The agent is initialized with the algorithm parameter, which defines the algorithm used for training. The use_alt_data parameter determines whether the agent should use alternative data sources during training.
* Loading a pre-existing model: If the modelname parameter is passed, the script checks if the specified model exists in the ./saved_models/ directory. If the model exists, it loads the model into the agent.
* Creating a backtesting environment: The script creates an instance of the Backtest class. This class is used to implement a backtesting environment, which is used to evaluate the performance of the trading agent. The script sets the parameters for the backtesting environment, such as tickers, target, stop, txcost, and others.
* Running the backtesting environment: The script calls the run_all method of the Backtest class, which runs the backtesting environment and trains the reinforcement learning agent.
* Saving and Generating Plots: The script also provides functions to save the agents and visualize the plots

Script parameters:

* `-l` or `--load`: whether to load ther feed from a pre-existing file (boolean True/False)
* `-f` or `--feed`: Specifies the type of data feed to use. If set to `back`, the script will create and use BackFeed data, and if set to `data`, the script will use a specified data file.
* `-d` or `--datafile`: The name of the data file to use
* `-m` or `--modelname`: The name of the model to load or save. 
* `-s` or `--synthetic`: If set to True, the script will use synthetic data. If set to False, the script will use real data.
* `-w` or `--weeks`: The number of weeks of data to use.
* `-t` or `--training_steps`: The number of training steps to perform.
* `-p` or `--deploy`: True or False denoting whether to deploy or not
* `-u` or `--use_alt_data`: True or False denoting whether to use alternative data or not

# `rlagents_test`

This script is the testing script corresponding to the training script `rlagents_train.py` (`Backtest` is used to simulate trades and evaluate the performance of the agent). It also contains utilities to plot the results including `buy/sell`annotations in the generated graphs.