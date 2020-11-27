# Homework 3 - Reinforcement Learning

## 3.1 Base Implementation

### a) Deep Q-Network (2)

> Implement a Deep Q-Network and its forward pass in the DQN class in model.py. Your network should take a single frame as input. In addition, you may again utilize the extract sensor values function. Describe your architecture in your report.

	> i) Would it be a problem if we only trained our network on a crop of the game pixels, which would not include the bottom status bar and would not use the extracted sensor values as an additional input to the network? Hint: Think about the Markov assumption.

Yes! If we crop out the image, and lose the bottom status bar, it would be a problem. The bottom status bar gives us the following information, current steering values, and acceleration. To construct a markovian state, we need this information from either the bottom status bar in the image or from the extracted sensor values. If not, a single frame state would not be markovian.  To make the state markovian in that case, we could use a stack of frames as state.

	> ii) Why do we not need to use a stack of frames as input to our network as proposed to play Atari games in the original Deep Q-Learning papers?

In the original DL papers, they create a single agent that can play multiple Atari games, and so a stack of frames is a general way to achieve a markovian state for all those games. In our case, the sensor values of acceleration and steer along with the image are sufficient to construct a markovian state.


### b) Deep Q-Learning (2)

	> i) Why do we utilize fixed targets with a separate policy and target network?


	> ii) Why do we sample training data from a replay memory instead of using a batch of past consecutive frames?


### c) Action selection (2)

	> i) Why do we need to balance exploration and exploitation in a reinforcement learning agent and
	how does the e-greedy algorithm accomplish this?


### d) Training (2)

	> Train a Deep Q-Learning agent using the train racing.py file with the provided default
	parameters. Describe your observations of the training process.

	> In particular, show the generated loss and reward curves and describe how they develop over the course of training. Some points of interest to describe should be: How quickly is the agent able to consistently achieve positive rewards? What is the relationship between the e-greedy exploration schedule and the development of the cumulative reward which the agent achieves over time?

	> How does the loss curve compare to the loss curve that you would expect to see on a standard supervised learning problem?



### e) Evaluation (2)

> Evaluate the trained Deep Q-Learning agent by running the evaluate racing.py script. Observe the performance of the agent by running the script on your local machine. Where does the agent do well and where does it struggle?


> How does its performance compare to the imitation learning agent you have trained for Exercise 1? Discuss possible reasons for the observed improvement/decline in performance compared to your imitation learning agent from Exercise 1.





