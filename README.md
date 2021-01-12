# RL-car

Training a reinforcement learning agent for OpenAI's Car Racing environment.

## Algorithms used
1. Deep Q-Learning[1]

We implement a Deep Q-Network and its forward pass in the DQN class in model.py. Our network takes a single frame as input.

The training loop for the DeepQ network is defined in deepq.py file. The target network updations and the deepQ step are defined in the learning.py file.

The action space is defined in the action.py file. We experimented with various action sets and eventually decided to stick with the 7 actions as defined in the file.

schedule.py is the script that defines the exploration-exploitation tradeoff. We begin with a p_initial value of 1 which means we would like to focus on exploration early on during the training.

2. Double Deep Q-Learning[2]

We implement a Double Deep Q-Network and its forward pass in the DQN class in model.py. Our network takes a single frame as input similar to the Deep Q learning experiment.

The traning loop for the Double Deep Q network is defined in the file deepq_double.py. The target network updation and double deepQ step is defined in the learning_double.py file.

For this experiment, we use the same action spaces as the DeepQ experiment.

We use the same scheme for exploration-exploitation tradeoff as in the Deep Q leanring experiment.


## Noticable techniques
1. Replay buffer for storing agent's memories
2. Target q-network to make q-learning stable



## To install the gym environment<br>
1) extract sdc_gym.zip
2) cd sdc_gym
3) pip install -e .["box2d"]

## To run the evaluation code
1) python evaluate_racing.py score


## References
1. [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)

