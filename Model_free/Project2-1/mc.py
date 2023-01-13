#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ast import Break
from pickle import FALSE, TRUE
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    # action
    if(score>=20):
        action = 0
    else:
        action= 1
    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for epi in range(n_episodes):
        #Printing current episode number
        print("Episode number :",epi)
        # initialize the episode
        current_state = env.reset()      

        
        # generate empty episode list
        episode_list=[]
       
        # loop until episode generation is done
        while(True):

            # select an action
            action = initial_policy(current_state)

            # return a reward and new state
            next_state,reward,done,info,trans_prob= env.step(action)
            
            # append state, action, reward to episode
            episode_list.append((current_state,action,reward))
            
            if(done):
                break

            # update state to new state
            current_state = next_state

        # compute G
        #Traversing the epsiode from episode termination and calculating return values 
        return_values=[]        
        ret = 0
        for (s,a, r) in reversed(episode_list):
            ret = gamma* ret + r
            return_values.append(ret)
        return_values.reverse()

        # loop for each step of episode          
        states_visited =[]
        for index, (s, a, r) in enumerate(episode_list):
            # unless state_t appears in states
            if s in states_visited:
                continue
            states_visited.append(s)

            # update return_count
            returns_count[s]+= 1

            # update return_sum
            returns_sum[s]+= return_values[index]

            # calculate average return for this state over all sampled episodes
            V[s] = returns_sum[s]/returns_count[s]


    ############################

    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #    
    
    probabilities = np.ones(nA, dtype=float) * epsilon / nA    #Random policy
    greedy_indices = np.argmax(Q[state]) 
    probabilities[greedy_indices] += 1.0 - epsilon   #Greedy Policy 
    #Choosing the action based on random choice given the values of epsilon greedy policy probabilities 
    action = np.random.choice(np.arange(len(probabilities)),p=probabilities)   
    

    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n
    ############################
    # YOUR IMPLEMENTATION HERE #

    # define decaying epsilon    
    decay_epsilon =lambda epsilon : epsilon - (0.1/n_episodes)
    while epsilon > 0:
        for epi in range(n_episodes):
            #Printing current episode number
            print("Episode number :",epi)

            # initialize the episode
            current_state = env.reset()

            # generate empty episode list
            episode_list=[]
            

            # loop until one episode generation is done
            while(True):

                # get an action from epsilon greedy policy
                action = epsilon_greedy(Q,current_state, nA, epsilon)

                # return a reward and new state
                next_state,reward,done,info,trans_prob= env.step(action)

                # append state, action, reward to episode
                episode_list.append((current_state,action,reward))

                if(done):
                    break

                # update state to new state
                current_state = next_state
            #decaying episode
            epsilon = decay_epsilon(epsilon)
            
            
            # Compute G
            #Traversing the epsiode from episode termination and calculating return values 
            return_values=[]            
            ret = 0
            for (s,a, r) in reversed(episode_list):
                ret = gamma* ret + r
                return_values.append(ret)
            return_values.reverse()

            # loop for each step of episode
            statesactions_visited =[]
            for index, (s, a, r) in enumerate(episode_list):
                

                # unless the pair state_t, action_t appears in <state action> pair list
                if (s,a) in statesactions_visited:
                    continue
                statesactions_visited.append((s,a))
                
                # update return_count
                returns_count[(s,a)]+= 1

                # update return_sum
                returns_sum[(s,a)]+= return_values[index]

                # calculate average return for this state over all sampled episodes
                Q[s][a] = returns_sum[(s, a)]/returns_count[(s, a)]
            # epsilon = epsilon - 0.1 / n_episodes

    return Q
