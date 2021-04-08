# Rainbow documentation

Here we will try to explain the code of the [Rainbow algorithm](https://arxiv.org/pdf/1710.02298.pdf) that we tried to implement here.

The code is splitted into 4 files : **The agent** (which contain the training loop and the communication to the Gym environment), the network (which is the DQN topology that we are training in the agent file) which also implements a custom 'Noisy layer' instead of a classical fully connected layer (or Dense layer), and finally we have an implementation of a certain kind of **Experience Replay Buffer** which is called **Prioritized Replay Buffer**

## Part 1 : Why is it important to understand what is a **Markov Decision Process** ?
A Deep Q-Network (DQN) is something called a **Q-learning** algorithm. Q-Learning attempts to learn the value of being in a given state, and taking a specific action there.

But before going into Q-learning (why does Q mean by the way ?), it is important to understand what is an MDP (Markov Decision Process) which is the foundation of Q-learning.

## Part 2 : MDP 101

Take a moment to locate the nearest big city around you. If you were to go there, how would you do it? Go by car, take a bus, take a train? Maybe ride a bike, or buy an airplane ticket?

Making this choice, you **incorporate probability into your decision-making process**. Perhaps there’s a 70% chance of rain or a car crash, which can cause traffic jams. If your bike tire is old, it may break down – this is certainly a large probabilistic factor. 

On the other hand, there are **deterministic costs** – for instance, the cost of gas or an airplane ticket – as well as deterministic rewards – like much faster travel times taking an airplane.

These types of problems – in which an agent must balance probabilistic and deterministic rewards and costs – are common in decision-making. **Markov Decision Processes** are used to model these types of optimization problems, and can also be applied to more complex tasks in Reinforcement Learning.

### Defining Markov Decision Processes in Machine Learning

To illustrate a Markov Decision process, think about a dice game:

* Each round, you can either **continue** or **quit**.

* If you **quit**, you receive $5 and the game ends.

* If you **continue**, you receive $3 and roll a 6-sided die. If the die comes up as 1 or 2, the game ends. Otherwise, the game continues onto the next round.

There is a clear trade-off here. For one, we can trade a deterministic gain of $2 for the chance to roll dice and continue to the next round.

To create an MDP to model this game, first we need to define a few things:

* A **state** is a status that the agent (decision-maker) can hold. In the dice game, the agent can either be in the game or out of the game. 

* An **action** is a movement the agent can choose. It moves the agent between states, with certain penalties or rewards.

* **Transition probabilities** describe the probability of ending 
up in a state $$s'$$ (s prime) given an action a. These will be often denoted as a function $P(s, a, s’)$ that outputs the probability of ending up in $$s’$$ given current state $s$ and action $a$.
For example, $P(s=playing the game, a=choose to continue playing, s’=not playing the game)$ is ⅓, since there is a two-sixths (one-third) chance of losing the dice roll.

* **Rewards** are given depending on the action. The reward for continuing the game is 3, whereas the reward for quitting is $5. The 'overall' reward is to be optimized.


```python

```