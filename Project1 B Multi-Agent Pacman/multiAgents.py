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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.val_and_act(gameState, 0, 0)[1]

    def val_and_act(self, state, index, depth):
        act = None
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), act
        elif index == state.getNumAgents() - 1:
            val = 999999
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                next_val = self.val_and_act(successor, 0, depth + 1)[0]
                if val > next_val:
                    act = action
                val = min(next_val, val)
            return val, act
        elif index == 0:
            val = -999999
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                next_val = self.val_and_act(successor, index + 1, depth)[0]
                if val < next_val:
                    act = action
                val = max(next_val, val)
            return val, act
        else:
            val = 999999
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                next_val = self.val_and_act(successor, index + 1, depth)[0]
                if val > next_val:
                    act = action
                val = min(next_val, val)
            return val, act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.val_and_act(gameState, -999999, 999999, 0, 0)[1]

    def val_and_act(self, state, alpha, beta, index, depth):
        act = None
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), act
        elif index == state.getNumAgents() - 1:
            val = 999999
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                next_val = self.val_and_act(successor, alpha, beta, 0, depth + 1)[0]
                if val > next_val:
                    act = action
                val = min(val, next_val)
                if val < alpha:
                    return val, act
                beta = min(beta, val)
            return val, act
        elif index == 0:
            val = -999999
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(0, action)
                next_val = self.val_and_act(successor, alpha, beta, index+1, depth)[0]
                if val < next_val:
                    act = action
                val = max(next_val, val)
                if val > beta:
                    return val, act
                alpha = max(alpha, val)
            return val, act
        else:
            val = 999999
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                next_val = self.val_and_act(successor, alpha, beta, index + 1, depth)[0]
                if val > next_val:
                    act = action
                val = min(val, next_val)
                if val < alpha:
                    return val, act
                beta = min(beta, val)
            return val, act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.val_and_act(gameState, 0, 0)[1]

    def val_and_act(self, state, index, depth):
        act = None
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), act
        elif index == state.getNumAgents() - 1:
            val = 0.0
            possible = 0.0
            for action in state.getLegalActions(index):
                possible = possible + 1.0
                successor = state.generateSuccessor(index, action)
                val = val + self.val_and_act(successor, 0, depth + 1)[0]
            val = val / possible
            return val, act
        elif index == 0:
            val = -999999.0
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                next_val = self.val_and_act(successor, index + 1, depth)[0]
                if val < next_val:
                    act = action
                val = max(next_val, val)
            return val, act
        else:
            val = 0.0
            possible = 0.0
            for action in state.getLegalActions(index):
                possible = possible + 1.0
                successor = state.generateSuccessor(index, action)
                val = val + self.val_and_act(successor, index+1, depth)[0]
            val = val / possible
            return val, act

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 4).

      DESCRIPTION: We do not want to die, so we must run away when a ghost is close enough.
      When the manhattan distance is smaller than 2, we think we must run away.

    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    cur_food = currentGameState.getFood()
    cur_position = currentGameState.getPacmanPosition()
    cur_capsules = currentGameState.getCapsules()

    list = []
    for i in range(cur_food.width):
        for j in range(cur_food.height):
            if cur_food[i][j]:
                list.append((i, j))
    cur_food = list

    food_closest = 999999
    for food in cur_food:
        distance = util.manhattanDistance(cur_position, food)
        food_closest = min(food_closest, distance)

    colst_ghost = 999999
    for ghost in currentGameState.getGhostStates():
        distance = util.manhattanDistance(cur_position, ghost.getPosition())
        colsest_ghost = min(colst_ghost, distance)

    cap_closest = 999999
    for cap in cur_capsules:
        distance = util.manhattanDistance(cur_position, cap)
        cap_closest = min(cap_closest, distance)

    return score + 10.0*(1.0/food_closest) + 100.0*(1.0/cap_closest) + colsest_ghost




# Abbreviation
better = betterEvaluationFunction

