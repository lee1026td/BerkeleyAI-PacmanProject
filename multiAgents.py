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


from game import Directions
import random, util

from game import Agent
from pacman import GameState
DEBUG = False


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Evaluation function for your reflex agent (question 1).

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
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        "for food"
        currentFoodList = currentGameState.getFood().asList()
        newFoodList = newFood.asList()

        if newFoodList :   # if foodList is not empty
            # get closest food's distance between pacman's new position
            closestFoodDistance = min(util.manhattanDist(newPos, food) for food in newFoodList)
            remainingFoods = len(newFoodList)

            score += 3.5 * (1.0 / (closestFoodDistance + 1)) - 0.5 * remainingFoods
            if len(newFoodList) < len(currentFoodList) :
                score += 2.0

        "to avoid freezing"

        if action == Directions.STOP :
            score -= 5.0

        "for ghosts"

        aliveGhosts = [g for g in newGhostStates if g.scaredTimer == 0]
        scaredGhosts = [g for g in newGhostStates if g.scaredTimer > 0]

        # for alive ghosts
        if aliveGhosts :
            closestAliveGhostDist = min(util.manhattanDist(newPos, g.getPosition()) for g in aliveGhosts)

            if closestAliveGhostDist < 2 :
                score -= 500.0
            else :
                score -= 4.0 * (1.0 / closestAliveGhostDist)

        # for scared ghosts
        if scaredGhosts :
            closestScaredGhostDist = min(util.manhattanDist(newPos, g.getPosition()) for g in scaredGhosts)

            score += 6.0 * (1.0 / closestScaredGhostDist)

        "for capsule"

        currentCapsules = currentGameState.getCapsules()

        if newCapsules :
            closestCapsuleDistance = min(util.manhattanDist(newPos, capsule) for capsule in newCapsules)
            score += 2.5 * (1.0 / (closestCapsuleDistance + 1))

            if len(newCapsules) < len(currentCapsules) :
                score += 3.5

        "to avoid backtracking loops"
        currentPacmanState = currentGameState.getPacmanState()
        currentPacmanDir = currentPacmanState.configuration.direction

        if (currentPacmanDir == Directions.NORTH and action == Directions.SOUTH or \
            currentPacmanDir == Directions.EAST and action == Directions.WEST or \
            currentPacmanDir == Directions.WEST and action == Directions.EAST or \
            currentPacmanDir == Directions.SOUTH and action == Directions.NORTH) :
            score -= 1.0

        print(f"[Score] {score}") if DEBUG else None
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    Your minimax agent (question 2)
    """

    def isTerminalState(self, gameState: GameState, depth) :
        return gameState.isWin() or gameState.isLose() or (depth >= self.depth)
    
    # return value, bestAction
    def miniMax(self, gameState: GameState, depth, agentIdx) :

        if self.isTerminalState(gameState, depth) :
            return self.evaluationFunction(gameState), None

        legalAction = gameState.getLegalActions(agentIdx)
        if not legalAction :
            return self.evaluationFunction(gameState), None

        numAgent = gameState.getNumAgents()

        nextAgent = (agentIdx + 1) % numAgent
        nextDepth = depth + (1 if nextAgent == 0 else 0)

        if agentIdx == 0 :
            # max (pacman)
            value, bestAction = float('-inf'), None
            for action in legalAction :
                successorState = gameState.generateSuccessor(agentIdx, action)
                val, _ = self.miniMax(successorState, nextDepth, nextAgent)

                if val > value :
                    value, bestAction = val, action
            return value, bestAction

        else :
            # min (ghosts)
            value, bestAction = float('inf'), None
            for action in legalAction :
                successorState = gameState.generateSuccessor(agentIdx, action)
                val, _ = self.miniMax(successorState, nextDepth, nextAgent)

                if val < value :
                    value, bestAction = val, action
            return value, bestAction


    def getAction(self, gameState: GameState):
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

        _, action = self.miniMax(gameState, depth=0, agentIdx=0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def isTerminalState(self, gameState: GameState, depth) :
        return gameState.isWin() or gameState.isLose() or (depth >= self.depth)
    
    def maxValue(self, gameState: GameState, depth, alpha, beta) :
        legalActions = gameState.getLegalActions(0)

        if not legalActions :
            return self.evaluationFunction(gameState), None
        
        value, bestAction = float('-inf'), None

        for action in legalActions :
            successorState = gameState.generateSuccessor(0, action)
            val, _ = self.value(successorState, depth, 1, alpha, beta)

            if val > value :
                value, bestAction = val, action
            
            alpha = max(alpha, value)
            
            # pruning
            if value > beta :
                break

        return value, bestAction
    
    def minValue(self, gameState: GameState, depth, agentIdx, alpha, beta) :
        legalActions = gameState.getLegalActions(agentIdx)

        if not legalActions :
            return self.evaluationFunction(gameState), None
        
        numAgents = gameState.getNumAgents()

        value, bestAction = float('inf'), None

        for action in legalActions :
            successorState = gameState.generateSuccessor(agentIdx, action)

            nextAgent = (agentIdx + 1) % numAgents
            nextDepth = depth + (1 if nextAgent == 0 else 0)

            val, _ = self.value(successorState, nextDepth, nextAgent, alpha, beta)

            if val < value :
                value, bestAction = val, action

            beta = min(beta, value)

            # pruning
            if value < alpha :
                break
        
        return value, bestAction

    def value(self, gameState: GameState, depth, agentIdx, alpha, beta) :

        if self.isTerminalState(gameState, depth) :
            return self.evaluationFunction(gameState), None
        
        if agentIdx == 0 :
            return self.maxValue(gameState, depth, alpha, beta)
        else :
            return self.minValue(gameState, depth, agentIdx, alpha, beta)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha, beta = float('-inf'), float('inf')
        _, bestAction = self.value(gameState, depth=0, agentIdx=0, alpha=alpha, beta=beta)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    
    score = currentGameState.getScore()

    if foodList :
        closestFoodDistance = min(util.manhattanDist(pacmanPos, food) for food in foodList)
        remainingFoods = len(foodList)

        score += 3.5 * (1.0 / (closestFoodDistance + 1)) - 0.5 * remainingFoods

    aliveGhosts = [g for g in ghostStates if g.scaredTimer == 0]
    scaredGhosts = [g for g in ghostStates if g.scaredTimer > 0]

    if aliveGhosts :
        closestAliveGhostDist = min(util.manhattanDist(pacmanPos, g.getPosition()) for g in aliveGhosts)

        if closestAliveGhostDist < 2 :
            score -= 500.0
        else :
            score -= 4.0 * (1.0 / closestAliveGhostDist)
    
    if scaredGhosts :
        closestScaredGhostDist = min(util.manhattanDist(pacmanPos, g.getPosition()) for g in scaredGhosts)

        score += 6.0 * (1.0 / closestScaredGhostDist)
    
    return score

# Abbreviation
better = betterEvaluationFunction
