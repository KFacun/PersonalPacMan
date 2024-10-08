# ASSIGNMENT 3
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
       # util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
       # util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
       # util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
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
    "*** YOUR CODE HERE ***"
    
    stack = util.Stack()  # Initialize a stack for DFS
    visited = set()  # Set to keep track of visited states

    startNode = (problem.getStartState(), [], 0)  # (state, actions, cost)
                                                  # It's the starting state of the problem
    stack.push(startNode)   # Starts the DFS

    while not stack.isEmpty():  # While loop runs until no more nodes to explore
        current_state, actions, _ = stack.pop() # current is pacman, actions is list of actions
                                                # _ is the cost (but in dfs is not used)

        if problem.isGoalState(current_state):  #Checks if pacman is at the goal.
            return actions  #If goal reached, function returns the list of actions from start to finish.

        if current_state not in visited:    # Checks if pacman has visited in state.
            visited.add(current_state)      # If not, it adds to visited set.

            for successor, action, cost in problem.getSuccessors(current_state): #Returns a list of triples
                # Loop iterates over all the successor states of the current state.
                
                new_actions = actions + [action]    # Creates a new list of actions by appending to current action
                                                    # to the list of actions taken so far
                                                    # This ensures the correct sequence of actions is maintained deep in search tree.
                stack.push((successor, new_actions, cost))  # successor state, along with associated action is pushed onto stack
                                                            # to be explored later.

    return []   # If stack is empty and no solution (goal state) is found, function returns empty list instead.
    
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"


    from util import Queue  # Use a FIFO queue
    # Initialize the queue for BFS and visited set
    queue = Queue()  # Queue for BFS
    visited = set()  # Set to track visited states
    
    # The start node contains the initial state, an empty list of actions, and a cost of 0
    start_node = (problem.getStartState(), [], 0)  # (state, actions, cost)
    
    # Push the start node into the queue
    queue.push(start_node)
    
    # While the queue is not empty, process each node
    while not queue.isEmpty():
        current_state, actions, currentCost = queue.pop()  # Pop from the queue
        
        # If the current state is the goal, return the list of actions
        if problem.isGoalState(current_state):
            return actions
        
        # If the current state hasn't been visited, expand it
        if tuple(current_state) not in visited:
            visited.add(tuple(current_state))  # Mark this state as visited
            # Loop over successors of the current state
            for successor, action, cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    # Generate the new action sequence
                    new_actions = actions + [action]
                    
                    # Push the successor into the queue
                    queue.push((successor, new_actions, cost + currentCost))
    # If no solution is found, return an empty list
    return []

    


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
     # Initialize a priority queue to hold nodes to be explored
    priority_queue = util.PriorityQueue()
    # Create a set to keep track of visited nodes
    visited = set()
    # Enqueue the start state into the priority queue as a tuple (state, actions, cost)
    priority_queue.push((problem.getStartState(), [], 0), 0)

# This loop continues until the pririty queue is empty (i.e all possible states have been explored)
    while not priority_queue.isEmpty():
        state, actions, cost = priority_queue.pop() #Lowest cumulative cost is popped from priority
        # checks if the current state is goal state.
        if problem.isGoalState(state):
            return actions # if goal, the function returns the sequence of actions that led to this state
        # Condition checks if the current state has already been visited.  If it hasn't, keep exploring
        if state not in visited:
            visited.add(state) # added to visited as "explored"
            successors = problem.getSuccessors(state) # retrieves the list of successor states from current state.  Each successor is represented as a tuple. (succesor, action, step_cost)
            # This loop processes each successor of the current state.
            for successor, action, step_cost in successors:
                if successor not in visited: # This ensures that the sucessor state hasn't been visited yet, to avoid re-exploration
                    new_actions = actions + [action]    #creates a new list of actions by appending the current action to the sequence of actions so far.
                    new_cost = cost + step_cost #Calculates the total cumulative cost.
                    priority_queue.push((successor, new_actions, new_cost), new_cost) # The sucessor, along with the actions and total cost is pushed onto the priority queue.
                    # Priority is set to new_cost, so UCS will ALWAYS priritize nodes with the least total cost.

    return []  # Return an empty list if no solution is found
    
  #  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    priority_queue = util.PriorityQueue()

    visited = set()

    start_state = problem.getStartState()

    priority_queue.push((start_state, [], 0), heuristic(start_state, problem))

    while not priority_queue.isEmpty():
      #  Pop the node with the lowest cost + heuristic
        current_state, actions, cost = priority_queue.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited:
            visited.add(current_state)

            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_actions = actions + [action]
                new_cost = cost + step_cost
                # Push the successor to the priority queue with new cost + heuristic
                priority_queue.push((successor, new_actions, new_cost), new_cost + heuristic(successor, problem))

    return []
    
  #  util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
