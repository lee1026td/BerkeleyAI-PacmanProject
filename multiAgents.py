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

    def isTerminalState(self, gameState: GameState, depth) :
        return gameState.isWin() or gameState.isLose() or (depth >= self.depth)

    def maxValue(self, gameState: GameState, depth) :
        legalActions = gameState.getLegalActions(0)

        if not legalActions :
            return self.evaluationFunction(gameState), None

        value, bestAction = float('-inf'), None

        for action in legalActions :
            successorState = gameState.generateSuccessor(0, action)
            
            val, _ = self.value(successorState, depth, agentIdx=1)

            if val > value :
                value, bestAction = val, action

        return value, bestAction

    def expectedValue(self, gameState: GameState, depth, agentIdx) :
        legalActions = gameState.getLegalActions(agentIdx)

        if not legalActions :
            return self.evaluationFunction(gameState), None
        
        numAgent = gameState.getNumAgents()

        prob = 1.0 / float(len(legalActions))
        expected = 0.0

        for action in legalActions :
            successorState = gameState.generateSuccessor(agentIdx, action)

            nextAgent = (agentIdx + 1) % numAgent
            nextDepth = depth + (1 if nextAgent == 0 else 0)

            val, _ = self.value(successorState, nextDepth, nextAgent)

            expected += prob * float(val)

        return expected, None

    def value(self, gameState: GameState, depth, agentIdx) :
        if self.isTerminalState(gameState, depth) :
            return self.evaluationFunction(gameState), None

        if agentIdx == 0 :
            return self.maxValue(gameState, depth)
        else :
            return self.expectedValue(gameState, depth, agentIdx)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, bestAction = self.value(gameState, depth=0, agentIdx=0)
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentCapsules = currentGameState.getCapsules()
    currentGhostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    currentFoodList = currentFood.asList()

    "for foods"
    if currentFoodList :
        foodDists = [util.manhattanDist(currentPos, food) for food in currentFoodList]
        closestFoodDist = min(foodDists)

        "consider average distance of 3 nearest foods for better planning"
        avgNearestFoodDist = sum(sorted(foodDists)[:min(3, len(foodDists))]) / min(3, len(foodDists))

        score += 10.0 / (closestFoodDist + 1)       # high bonus for getting close to closest food
        score += 5.0 / (avgNearestFoodDist + 1)     # little lower bonus for getting close to 3 nearest foods

        score -= 4.0 * len(currentFoodList)         # penalty for remaining foods
    else :
        score += 500                                # if there is no remaining foods, give high bonus


    "for ghosts"
    aliveGhosts = [g for g in currentGhostStates if g.scaredTimer == 0]
    scaredGhosts = [g for g in currentGhostStates if g.scaredTimer > 0]

    "for alive ghosts"
    if aliveGhosts :
        aliveGhostDists = [util.manhattanDist(currentPos, g.getPosition()) for g in aliveGhosts]
        closestAliveGhostDist = min(aliveGhostDists)

        '''
        when the distance between pacman and closest ghost is less than 1, give high penalty (-1000)
        for the distance of 2, little bit lower penalty
        for the distance of 4 or more, give penalty in inverse proportion to distance
        '''
        if closestAliveGhostDist <= 1 :
            score -= 1000
        elif closestAliveGhostDist == 2 :
            score -= 300
        elif closestAliveGhostDist <= 4 :
            score -= 100 / closestAliveGhostDist
        else :
            score -= 10 / closestAliveGhostDist

        "consider second closest ghost too"
        if len(aliveGhostDists) > 1 :
            secondClosestDist = sorted(aliveGhostDists)[1]
            if secondClosestDist <= 3 :
                score -= 50 / secondClosestDist

    "for scared ghosts"
    if scaredGhosts :
        for scaredGhost in scaredGhosts :
            scaredGhostDist = util.manhattanDist(currentPos, scaredGhost.getPosition())
            scaredTime = scaredGhost.scaredTimer

            '''
            if there is enough time to get scared ghost, chase the scared one
            '''
            if scaredTime > scaredGhostDist :
                score += 200 / (scaredGhostDist + 1)                # to chase the scared ghost
                score += min(scaredTime - scaredGhostDist, 10) * 5  # when the pacman catches the scared ghost in enough time remains, give bonus
            else :
                if scaredGhostDist <= 2 :                           # if there is not sufficient time, move carefully
                    score -= 50

    "for capsules"
    if currentCapsules :
        capsuleDists = [util.manhattanDist(currentPos, capsule) for capsule in currentCapsules]
        closestCapsuleDist = min(capsuleDists)

        "if there are alive ghosts in nearby, prefer capsules more"
        if aliveGhosts :
            closestAliveGhostdist = min(util.manhattanDist(currentPos, g.getPosition()) for g in aliveGhosts)

            if closestAliveGhostDist <= 5 :
                score += 150 / (closestCapsuleDist + 1)     # ghost is too close, get capsule
            else :
                score += 50 / (closestCapsuleDist + 1)      # less dangerous situation

        else :
            score += 30 / (closestCapsuleDist + 1)          # there is no alive ghost : lower priority on capsule
    
        score -= 20 * len(currentCapsules)                  # penalty for remaining capsules

    "if all of the ghosts are scared, play more aggressive"
    if scaredGhosts and not aliveGhosts :
        score += 100

    return score

# Abbreviation
better = betterEvaluationFunction
