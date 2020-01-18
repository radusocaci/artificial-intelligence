# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, copy

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here

        #Computes V*(s) (with Bellman Equation) - expected utility starting in s and acting optimally
        valuesNext = util.Counter()
        states = mdp.getStates()

        for i in range(0, self.iterations): # for the number of iterations given
            for state in states:
                values = []
                if self.mdp.isTerminal(state):
                    values.append(0)
                #compute possible actions from Q-states
                possibleActions = mdp.getPossibleActions(state)
                for action in possibleActions:
                    values.append(self.computeQValueFromValues(state, action)) #reward + discout * value of state
                valuesNext[state] = max(values) #maximum value of qstates

            self.values = valuesNext.copy() #put in values a copy of the counter, update for each iteration

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        #Computes Q*(state,action) - expected utility that is taking action a from a state s and acts optimally
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0

        for nextSt, probability in transitionStatesAndProbs: #uses Bellman Equation to compute Q values
            value += probability * (self.mdp.getReward(state, action, nextSt) + self.discount * self.getValue(nextSt))
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        if len(possibleActions) == 0:
            return None

        policy = 0
        maxVal = 0
        for action in possibleActions:
            value = self.computeQValueFromValues(state, action)
            if value > maxVal or maxVal == 0: #find the maximum of values
                maxVal = value
                policy = action

        return policy #return the optimal action with the best Q-value

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

