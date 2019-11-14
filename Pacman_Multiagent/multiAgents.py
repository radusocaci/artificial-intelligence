# multiAgents.py
# --------------
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


import random

import util
from game import Agent, manhattanDistance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodPos = newFood.asList()

        ghostPos = [s.getPosition() for s in newGhostStates]

        distsFood = []
        distsGhosts = []
        remainingFood = len(newFood.asList())

        if remainingFood == currentGameState.getFood().count():
            #if it doesnt eat this action
            foodCapsules = min([manhattanDistance(fpos, newPos) for fpos in foodPos])
        else:
            foodCapsules = 0 # this action eats

        #impact of ghost
        ghosts = min([manhattanDistance(newPos, gPos) for gPos in ghostPos])

        #time ghosts are scared
        time = newScaredTimes[0]

        return successorGameState.getScore() - foodCapsules + ghosts - time


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        v = [self.minvalue(gameState.generateSuccessor(0, action), 1, 0) for action in gameState.getLegalActions(0)]
        i = v.index(max(v))
        return gameState.getLegalActions(0)[i]  # the pacman will do the action chosen

    def minvalue(self, gameState, i, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        depth += 1 if i + 1 == gameState.getNumAgents() else 0
        values = [self.minvalue(gameState.generateSuccessor(i, action), (i + 1) % gameState.getNumAgents(), depth) for
                  action in gameState.getLegalActions(i)]

        return max(values) if i == 0 else min(values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alfabetavalue(gameState, 0, 0, -999999, 999999)[1]  # the pacman will do the action chosen

    # a - MAX's best option on path to root
    # b - MIN's best option on path to root
    def alfabetavalue(self, gameState, i, depth, a, b):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None) # max depth is reached or the game is ended

        depth += 1 if i + 1 == gameState.getNumAgents() else 0 # if we reached the last ghost, we increase the depth of the graph

        if i == 0: # if agent is pacman , maximize
            best = (-999999, None) #tuple will hold the state values and the coresponding action
            for action in gameState.getLegalActions(i): #iterate through agent's legal actions
                v = self.alfabetavalue(gameState.generateSuccessor(i, action), (i + 1) % gameState.getNumAgents(), depth, a ,b)[0]#minimax value for next agent
                if best[0] < v:
                    best = (v, action)
                if v > b:
                    return best #if node becomes worse than a, stop considering it's children
                a = max(a, best[0])
            return best
        else: #if agent is ghost, minimize
            best = (999999, None)
            for action in gameState.getLegalActions(i):
                v = self.alfabetavalue(gameState.generateSuccessor(i, action), (i + 1) % gameState.getNumAgents(), depth, a, b)[0]#minimax value for the next agent
                if best[0] > v:
                    best = (v, action)
                if v < a:
                    return best #if node becomes worse than a, MAX avoids it
                b = min(b, best[0])
            return best



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
