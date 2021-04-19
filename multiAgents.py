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

MAXIMUM_VALUE = 9999999
MINIMUM_VALUE = -MAXIMUM_VALUE


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        for ghost in newGhostStates:
            if newPos == ghost.configuration.pos:
                if ghost.scaredTimer == 0:
                    return MINIMUM_VALUE
                else:
                    # Go for it!
                    return MAXIMUM_VALUE

        if action == Directions.STOP:
            return MINIMUM_VALUE

        dist_to_food = min([manhattanDistance(newPos, food) for food in currentGameState.getFood().asList()])
        return -dist_to_food


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

    """
    Returns tuple in form (action, value)
    """

    def pacmanTurn(self, gameState, turn=0):
        action_minimax_list = []
        for action in gameState.getLegalActions(0):
            successor_gameState = gameState.generateSuccessor(0, action)
            action_minimax_list.append((action, self.minimax(successor_gameState, turn + 1)))
        return max(action_minimax_list, key=lambda t: t[1])

    def ghostTurn(self, gameState, turn):
        action_minimax_list = []
        agentIndex = turn % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            ghostSuccessor_gameState = gameState.generateSuccessor(agentIndex, action)
            action_minimax_list.append((action, self.minimax(ghostSuccessor_gameState, turn + 1)))
        return min(action_minimax_list, key=lambda t: t[1])

    def minimax(self, gameState, turn):
        if gameState.isWin() or gameState.isLose() or turn is gameState.getNumAgents() * self.depth:
            return self.evaluationFunction(gameState)

        # It's pacman
        if turn % gameState.getNumAgents() == 0:
            _, value = self.pacmanTurn(gameState, turn)
            return value
        else:
            _, value = self.ghostTurn(gameState, turn)
            return value

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

        action, value = self.pacmanTurn(gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimax_with_pruning(self, gameState, turn, alpha, beta):
        if gameState.isWin() or gameState.isLose() or turn is gameState.getNumAgents() * self.depth:
            return self.evaluationFunction(gameState)

        # It's pacman
        if turn % gameState.getNumAgents() == 0:
            _, value = self.pacmanTurn(gameState, turn, alpha, beta)
            return value
        else:
            _, value = self.ghostTurn(gameState, turn, alpha, beta)
            return value

    def pacmanTurn(self, gameState, turn=0, alpha=MINIMUM_VALUE, beta=MAXIMUM_VALUE):
        action_minimax_list = []
        for action in gameState.getLegalActions(0):
            successor_gameState = gameState.generateSuccessor(0, action)
            max_value = self.minimax_with_pruning(successor_gameState, turn + 1, alpha, beta)
            action_minimax_list.append((action, max_value))
            alpha = max(alpha, max_value)
            if alpha > beta:
                break
        return max(action_minimax_list, key=lambda t: t[1])

    def ghostTurn(self, gameState, turn, alpha=MINIMUM_VALUE, beta=MAXIMUM_VALUE):
        action_minimax_list = []
        agentIndex = turn % gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            ghostSuccessor_gameState = gameState.generateSuccessor(agentIndex, action)
            min_value = self.minimax_with_pruning(ghostSuccessor_gameState, turn + 1, alpha, beta)
            action_minimax_list.append((action, min_value))
            beta = min(beta, min_value)
            if alpha > beta:
                break
        return min(action_minimax_list, key=lambda t: t[1])

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        action, _ = self.pacmanTurn(gameState)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def pacmanTurn(self, gameState, turn=0):
        action_minimax_list = []
        for action in gameState.getLegalActions(0):
            successor_gameState = gameState.generateSuccessor(0, action)
            action_minimax_list.append((action, self.expectimax(successor_gameState, turn + 1)))
        return max(action_minimax_list, key=lambda t: t[1])

    def ghostTurn(self, gameState, turn):
        scores = []
        agentIndex = turn % gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            ghostSuccessor_gameState = gameState.generateSuccessor(agentIndex, action)
            scores.append(self.expectimax(ghostSuccessor_gameState, turn + 1))

        avg = sum(scores) / len(scores)
        action = random.choice(gameState.getLegalActions(agentIndex))
        return action, avg

    def expectimax(self, gameState, turn):
        if gameState.isWin() or gameState.isLose() or turn is gameState.getNumAgents() * self.depth:
            return self.evaluationFunction(gameState)

        # It's pacman
        if turn % gameState.getNumAgents() == 0:
            _, value = self.pacmanTurn(gameState, turn)
            return value
        else:
            _, value = self.ghostTurn(gameState, turn)
            return value

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action, _ = self.pacmanTurn(gameState)
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Starts from the score of the game.

    More score while close to the lovely ghosts (scared that can be reached),
    less while close to the angry ghosts

    Better score while there is less food, that gets as game score and win the game.
    Much better score while this move taking the capsules number down,
    eating scared ghost is the best way to get points.

    And of course, the closet to food you are, you score get higher :)
    """

    pacman_pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    angryGhosts = [ghost.configuration.pos for ghost in ghostStates if
                   ghost.scaredTimer == 0
                   or manhattanDistance(pacman_pos,
                                        ghost.configuration.pos) >= ghost.scaredTimer]
    lovelyGhost = [ghost.configuration.pos for ghost in ghostStates if ghost.configuration.pos not in angryGhosts]

    score = currentGameState.getScore()
    score += 1 / (1 + sum([manhattanDistance(pacman_pos, ghost) for ghost in lovelyGhost]))
    score -= 1 / (1 + sum([manhattanDistance(pacman_pos, ghost) for ghost in angryGhosts]))

    score -= len(currentGameState.getFood().asList())
    score -= 10 * len(currentGameState.getCapsules())

    dist_to_food = 0
    if len(currentGameState.getFood().asList()) != 0:
        dist_to_food = min([manhattanDistance(pacman_pos, food) for food in currentGameState.getFood().asList()])
    score += 1 / float(1 + dist_to_food)

    return score


# Abbreviation
better = betterEvaluationFunction
