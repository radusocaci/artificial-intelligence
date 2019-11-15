# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    stack = Stack()
    stack.push((problem.getStartState(), []))  # start state with the associated actions list

    visitedStates = []  # visited states list # TODO implement as set

    while not stack.isEmpty():  # as long as there are items left to process
        currentState, actions = stack.pop()  # process current node

        if currentState not in visitedStates:  # if it was already visited by DFS => skip this node
            visitedStates.append(currentState)

            if problem.isGoalState(currentState):  # we return the actions if we discovered a goal state
                return actions  # TODO check goal when successor is generated

            for nextState, nextAction, cost in problem.getSuccessors(currentState):
                stack.push((nextState, actions + [nextAction]))  # compute successors in graph and add the to the stack


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queue = Queue()  # the same as DFS, but using a queue instead
    queue.push((problem.getStartState(), []))

    visitedStates = []

    while not queue.isEmpty():
        currentState, actions = queue.pop()

        if currentState not in visitedStates:
            visitedStates.append(currentState)

            if problem.isGoalState(currentState):
                return actions

            for nextState, nextAction, cost in problem.getSuccessors(currentState):
                queue.push((nextState, actions + [nextAction]))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    priorityQueue = PriorityQueue()  # same as BFS but nodes are explored in the order of their cost
    priorityQueue.push((problem.getStartState(), [], 0), 0)  # the state also has a total cost path to keep track of the cost

    visitedStates = []

    while not priorityQueue.isEmpty():
        currentState, actions, costUntilHere = priorityQueue.pop()

        if currentState not in visitedStates:
            visitedStates.append(currentState)

            if problem.isGoalState(currentState):
                return actions

            for nextState, nextAction, cost in problem.getSuccessors(currentState):  # the cost is also updated
                priorityQueue.update((nextState, actions + [nextAction], costUntilHere + cost), costUntilHere + cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    priorityQueue = PriorityQueue()
    priorityQueue.push((problem.getStartState(), [], 0), 0)  # state is the same as UCS

    visitedStates = []

    while not priorityQueue.isEmpty():
        currentState, actions, costUntilHere = priorityQueue.pop()

        if currentState not in visitedStates:
            visitedStates.append(currentState)

            if problem.isGoalState(currentState):
                return actions

            for nextState, nextAction, cost in problem.getSuccessors(currentState):
                priorityQueue.update((nextState, actions + [nextAction], costUntilHere + cost),
                                     costUntilHere + cost + heuristic(nextState, problem))  # priority queue takes into consideration the heuristic


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
