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


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    point = problem.getStartState()  #
    reached = set()
    stack = []
    cache = {}
    reached.add(point)
    goal = None
    for i in problem.getSuccessors(point):  # put all the end point into stack at first
        cache[i] = (point, None, None)
        stack.append(i)

    while stack:
        u = stack.pop()
        if u[0] in reached:  # has already been in u[0] ignore
            continue
        if problem.isGoalState(u[0]):  # if this u[0] is target we can break
            goal = u
            break

        # else
        reached.add(u[0])  # u[0] has been visited
        for i in problem.getSuccessors(
            u[0]
        ):  # get next can be reached place push into stack
            cache[i] = (
                u  # P is the path dict, we can find the next node's Previous node
            )
            stack.append(i)

    # until we meet the target or the stack become empty, every place we have been reached
    path, current_node = [], goal
    while current_node != (point, None, None):  # we add all the
        path.insert(
            0, current_node[1]
        )  # From the goal back to get the path, add into path
        current_node = cache[current_node]
    # print(path)
    return path
    # util.raiseNotDefined()


class Node:
    def __init__(self, state, prev, action, priority=0) -> None:
        self.state = state
        self.prev = prev
        self.action = action
        self.priority = priority


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    closed = set()
    cache = {}
    queue = [Node(problem.getStartState(), None, None)]

    while queue:
        node = queue.pop(0)

        if problem.isGoalState(node.state) == True:  # reached the end
            actions = []
            while node.action is not None:
                actions.insert(0, node.action)
                node = node.prev
            return actions

        # still running
        if node.state in closed:
            continue
        closed.add(node.state)
        for s in problem.getSuccessors(node.state):
            queue.append(Node(s[0], node, s[1]))
    return []

    # util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    closed = set()
    queue = util.PriorityQueue()
    queue.push(Node(problem.getStartState(), None, None), 0)

    while queue:
        node = queue.pop()
        if problem.isGoalState(node.state) == True:  # reached the end
            actions = []
            while node.action is not None:
                actions.insert(0, node.action)
                node = node.prev
            return actions

        # still running
        if node.state in closed:
            continue
        closed.add(node.state)
        for s in problem.getSuccessors(node.state):
            queue.push(
                Node(s[0], node, s[1], s[2] + node.priority), s[2] + node.priority
            )
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    closed = set()
    queue = util.PriorityQueue()
    state = problem.getStartState()

    queue.push(
        Node(state, None, None, heuristic(state, problem)), heuristic(state, problem)
    )
    while queue:
        node = queue.pop()
        if problem.isGoalState(node.state):
            ret = []
            while node.action:
                ret.insert(0, node.action)
                node = node.prev
            return ret

        if node.state in closed:
            continue

        closed.add(node.state)
        for s in problem.getSuccessors(node.state):
            queue.push(
                Node(s[0], node, s[1], s[2] + node.priority),
                s[2] + node.priority + heuristic(s[0], problem),
            )

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
