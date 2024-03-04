# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)


"""
Project 2: MDP and Q-Learning
Team Members:
    1: Kalpit Vadnerkar
    2: Dhananjay Nikam
"""

"""
In this file, you should test your Q-learning implementation on the robot crawler environment 
that we saw in class. It is suggested to test your code in the grid world environments before this one.

The package `matplotlib` is needed for the program to run.


The Crawler environment has discrete state and action spaces
and provides both of model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
  

Once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
"""


# use random library if needed
import random

def q_learning(env, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: CrawlerEnv
        the environment
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """

    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    gamma = 0.95   
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    # maximum number of training iterations
    max_iterations = 5000
    #########################

### Please finish the code below ##############################################
###############################################################################
    q = [ [0] * NUM_ACTIONS for _ in range(NUM_STATES)]
    
### Please finish the code below ##############################################
###############################################################################
    for k in range(max_iterations):
        s = env.reset()
        j=0
        while True:
            ways = [0,1] # 0 is random and 1 is according to policy
            choice = random.choices(ways, [eps,1-eps])[0]
            if choice == 1:
                chosen_action = pi[s]
            else:
                chosen_action = random.choices(range(NUM_ACTIONS))[0]        
            s_, r, terminal, info = env.step(chosen_action)
            if choice == 1:
                next_action = q[s_].index(max(q[s_]))
            else:
                next_action = random.choices(range(NUM_ACTIONS))[0]
            if terminal:
                target = r
            else:
                target = r + gamma * q[s_][next_action] #q[_s][next best choice]
            q[s][chosen_action] = (1 - alpha) * q[s][chosen_action] + alpha * target
            v[s] = max(q[s])
            pi[s] = q[s].index(max(q[s]))
            s = s_
            if terminal:
                break
            logger.log(j+1, v, pi)
            j = j + 1
            # Linear decay function
            eps = 1 - j*0.0005 
            if eps <= 0:
                eps = 0
###############################################################################
    return pi


if __name__ == "__main__":
    from app.crawler import App
    import tkinter as tk
    
    algs = {
        "Q Learning": q_learning,
    }

    root = tk.Tk()
    App(algs, root)
    root.mainloop()