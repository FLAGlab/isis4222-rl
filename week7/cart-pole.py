#!/usr/bin/env python3 # -*- coding: utf-8 -*-
## Cartpole_DQL.v2.py
## Balancing a cartpole using Double Q-learning (without a neural network) and using
## a discretized state space. The algorithm is trained using epsilon-greedy approach
## with Replay Memory ##
## @author: Rohan Sarkar (sarkarr@purdue.edu)
import gym
import numpy as np
import time
import matplotlib.pyplot as plt 
import sys
import random
seed = 0 
random.seed(seed) 
np.random.seed(seed)
## Initialize parameters and environment variables: 
env = gym.make('CartPole-v0')
GAMMA = 0.85
MAX_ITER = 1000  #500
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 1+np.log(EXPLORATION_MIN)/MAX_ITER 
REP_MEM_SIZE = 100000
MINIBATCH_SIZE = 64
N_states = 162
N_actions = 2
Q1 = np.zeros((N_states, N_actions)) 
Q2 = np.zeros((N_states, N_actions)) 
alpha = 0.001
eps_rate = EXPLORATION_MAX 
cum_reward = 0 
train_reward_history = []
## Map the four dimensional continuous state-space to discretized state-space: ## Credit: http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf.
def map_discrete_state(cs):
    ds_vector = np.zeros(4); 
    ds = -1
    # Discretize x (position) 
    if abs(cs[0]) <= 0.8:
        ds_vector[0] = 1 
    elif cs[0] < -0.8:
        ds_vector[0] = 0 
    elif cs[0] > 0.8:
        ds_vector[0] = 2
    # Discretize x' (velocity) 
    if abs(cs[1]) <= 0.5:
        ds_vector[1] = 1 
    elif cs[1] < -0.5:
        ds_vector[1] = 0 
    elif cs[1] > 0.5:
        ds_vector[1] = 2
    # Discretize theta (angle) 
    angle = 180/3.1428*cs[2] 
    if -12 < angle <= -6:
        ds_vector[2] = 0 
    elif -6 < angle <= -1: 
        ds_vector[2] = 1 
    elif -1 < angle <= 0:
        ds_vector[2] = 2 
    elif 0 < angle <= 1: 
        ds_vector[2] = 3 
    elif 1 < angle <= 6: 
        ds_vector[2] = 4
    elif 6 < angle <= 12: 
        ds_vector[2] = 5
    # Discretize theta' (angular velocity) 
    if abs(cs[3]) <= 50:
        ds_vector[3] = 1 
    elif cs[3] < -50:
        ds_vector[3] = 0 
    elif cs[3] > 50:
        ds_vector[3] = 2
    ds = int(ds_vector[0]*54+ds_vector[1]*18+ds_vector[2]*3+ds_vector[3]) 
    return ds

## Return the most optimal action and the corresponding Q value: 
def optimal_action(Q, state, eps_rate, env):
    p = np.random.random()
    # Choose random action if in 'exploration' mode 
    if p < eps_rate:
        action = env.action_space.sample() # choose randomly between 0 and 1 
    else:
    # Choose optimal action based on learned weights if in 'exploitation' mode
        action = np.argmax(Q[state,:]) 
    return action

## Update Qvalues based on the logic of the Double Q-learning:
def updateQValues(state, action, reward, next_state, alpha, eps_rate, env):
    p = np.random.random()
    if (p < .5):
    # Update Qvalues for Table Q1
        next_action = optimal_action(Q1, next_state, eps_rate, env) 
        Q1[state][action] = Q1[state][action] + alpha * (reward + GAMMA * Q2[next_state][next_action] - Q1[state][action])
    else:
    # Update Qvalues for Table Q2
        next_action = optimal_action(Q2, next_state, eps_rate, env) 
        Q2[state][action] = Q2[state][action] + alpha * (reward + GAMMA * Q1[next_state][next_action] - Q2[state][action]) 
    return next_action


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, s, a , r, ns):
        self.memory.append([s, a, r, ns])
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, MINIBATCH_SIZE):
        return random.sample(self.memory, MINIBATCH_SIZE) 
    
    def __len__(self):
        return len(self.memory)
    
if __name__ == "__main__":
    ## Learn the weights for Double Q learning: 
    max_t = 0
    duration_history = []
    history = []
    epsilon_history = []
    repmemory = ReplayMemory(REP_MEM_SIZE)
    for i_episode in range(MAX_ITER):
        ## state is a 4-tuple: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        state = env.reset() ### state set to uniformly dist random bet -0.05 and 0.05 done = False
        # Initial state and action selection when environment restarts
        state = map_discrete_state(state)
        action = optimal_action(0.5*(Q1+Q2), state, 0, env)
        t=0
        # Decay the exploration parameter in epsilon-greedy approach.
        eps_rate *= EXPLORATION_DECAY
        eps_rate = max(EXPLORATION_MIN, eps_rate)
        while True:
            #env.render()
            next_state, reward, done, info = env.step(action) 
            if done:
                reward = -10
            next_state = map_discrete_state(next_state)
            repmemory.push(state, action, reward, next_state)
            # Update Q table using Double Q learning and get the next optimal action. 
            next_action = updateQValues(state, action, reward, next_state, alpha, eps_rate, env) # Update Q values by randomly sampling experiences in Replay Memory
            if len(repmemory)> MINIBATCH_SIZE:
                experiences = repmemory.sample(MINIBATCH_SIZE) 
                for experience in experiences:
                    ts, ta, tr, tns = experience
                    updateQValues(ts, ta, tr, tns, alpha, eps_rate, env) 
            state = next_state
            action = next_action 
            t += 1
            if done:
                break 
        history.append(t) 
        if i_episode > 50:
            latest_duration = history[-50:] 
        else:
            latest_duration = history 
        #print(latest_duration)
        duration_run_avg = np.mean(latest_duration) #print(duration_run_avg) duration_history.append([t, duration_run_avg]) epsilon_history.append(eps_rate)
        cum_reward += t
        if(t>max_t):
            max_t = t
        train_reward_history.append([cum_reward/(i_episode+1), max_t])
        print("\nEpisode: %d Episode duration: %d timesteps | epsilon %f" % (i_episode+1, t+1, eps_rate))
    np.save('Q1.npy', Q1)
    np.save('Q2.npy', Q2)
    fig = plt.figure(1)
    fig.canvas.set_window_title("DQL Training Statistics") 
    plt.clf()
    plt.subplot(1, 2, 1) 
    plt.title("Training History") 
    plt.plot(np.asarray(duration_history)) 
    plt.xlabel('Episodes') 
    plt.ylabel('Episode Duration')
    plt.subplot(1, 2, 2)
    plt.title("Epsilon for Episode") 
    plt.plot(np.asarray(epsilon_history)) 
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Value') 
    plt.savefig('Cartpole_DoubleQ_Learning.png') 
    plt.show()

    ## Finished exploration. Start testing now.
    import pymsgbox
    response = pymsgbox.confirm("Finished learning the weights for the Double-Q Algorithm. Start testing?") 
    if response == "OK":
        pass 
    else:
        sys.exit("Exiting")
    print("\n\n\nThe Testing Phase:\n\n")
    ## Control the cartpole using the learned weights using Double Q learning 
    play_reward =0
    for i_episode in range(100):
        observation = env.reset()
        done = False
        # Initial state and action selection when environment restarts state = map_discrete_state(observation)
        action = optimal_action(0.5*(Q1+Q2), state, 0, env) 
        t=0
        eps_rate = 0
        time.sleep(0.25) 
        while not done: 
            env.render()
            time.sleep(0.1)
            next_state, reward, done, info = env.step(action) 
            next_state = map_discrete_state( next_state ) 
            next_action = optimal_action(Q1, state, 0, env) 
            state = next_state
            action = next_action
            t += 1
        play_reward += t
        print("Episode ",i_episode," Episode duration: {} timesteps".format(t+1))
    env.close()