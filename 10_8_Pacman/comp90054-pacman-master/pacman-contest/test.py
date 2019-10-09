from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):
    def __init__(self, index):
        self.index = index
        self.observationHistory = []

    # Follows from getSuccessor function of ReflexCaptureAgent
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Follows from chooseAction function of ReflexCaptureAgent
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    # Follows from evaluate function of ReflexCaptureAgent
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        # Start like getFeatures of OffensiveReflexAgent
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Get other variables for later use
        food = self.getFood(gameState)
        capsules = gameState.getCapsules()
        foodList = food.asList()
        walls = gameState.getWalls()
        x, y = gameState.getAgentState(self.index).getPosition()
        vx, vy = Actions.directionToVector(action)
        newx = int(x + vx)
        newy = int(y + vy)

        # Get set of invaders and defenders
        enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        # Check if pacman has stopped
        if action == Directions.STOP:
            features["stuck"] = 1.0

        # Get ghosts close by
        for ghost in invaders:
            ghostpos = ghost.getPosition()
            neighbors = Actions.getLegalNeighbors(ghostpos, walls)
            if (newx, newy) == ghostpos:
                if ghost.scaredTimer == 0:
                    features["scaredGhosts"] = 0
                    features["normalGhosts"] = 1
                else:
                    features["eatFood"] += 2
                    features["eatGhost"] += 1
            elif ((newx, newy) in neighbors) and (ghost.scaredTimer > 0):
                features["scaredGhosts"] += 1
            elif (successor.getAgentState(self.index).isPacman) and (ghost.scaredTimer > 0):
                features["scaredGhosts"] = 0
                features["normalGhosts"] += 1

        # How to act if scared or not scared
        if gameState.getAgentState(self.index).scaredTimer == 0:
            for ghost in defenders:
                ghostpos = ghost.getPosition()
                neighbors = Actions.getLegalNeighbors(ghostpos, walls)
                if (newx, newy) == ghostpos:
                    features["eatInvader"] = 1
                elif (newx, newy) in neighbors:
                    features["closeInvader"] += 1
        else:
            for ghost in enemies:
                if ghost.getPosition() != None:
                    ghostpos = ghost.getPosition()
                    neighbors = Actions.getLegalNeighbors(ghostpos, walls)
                    if (newx, newy) in neighbors:
                        features["closeInvader"] += -10
                        features["eatInvader"] = -10
                    elif (newx, newy) == ghostpos:
                        features["eatInvader"] = -10

        # Get capsules when nearby
        for cx, cy in capsules:
            if newx == cx and newy == cy and successor.getAgentState(self.index).isPacman:
                features["eatCapsule"] = 1.0

        # When to eat
        if not features["normalGhosts"]:
            if food[newx][newy]:
                features["eatFood"] = 1.0
            if len(foodList) > 0:
                tempFood = []
                for food in foodList:
                    food_x, food_y = food
                    adjustedindex = self.index - self.index % 2
                    check1 = food_y > (adjustedindex / 2) * walls.height / 3
                    check2 = food_y < ((adjustedindex / 2) + 1) * walls.height / 3
                    if (check1 and check2):
                        tempFood.append(food)
                if len(tempFood) == 0:
                    tempFood = foodList
                mazedist = [self.getMazeDistance((newx, newy), food) for food in tempFood]
                if min(mazedist) is not None:
                    walldimensions = walls.width * walls.height
                    features["nearbyFood"] = float(min(mazedist)) / walldimensions
        features.divideAll(10.0)

        return features

    def getWeights(self, gameState, action):
        return {'eatInvader': 5, 'closeInvader': 0, 'teammateDist': 1.5, 'nearbyFood': -1, 'eatCapsule': 10.0,
                'normalGhosts': -20, 'eatGhost': 1.0, 'scaredGhosts': 0.1, 'stuck': -5, 'eatFood': 1}


class DefensiveAgent(CaptureAgent):
    def __init__(self, index):
        self.index = index
        self.observationHistory = []

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        if (successor.getAgentState(self.index).scaredTimer > 0):
            features['numInvaders'] = 0
            if (features['invaderDistance'] <= 2): features['invaderDistance'] = 2
        teamNums = self.getTeam(gameState)
        initPos = gameState.getInitialAgentPosition(teamNums[0])
        # use the minimum noisy distance between our agent and their agent
        features['DistancefromStart'] = myPos[0] - initPos[0]
        if (features['DistancefromStart'] < 0): features['DistancefromStart'] *= -1
        if (features['DistancefromStart'] >= 10): features['DistancefromStart'] = 10
        if (features['DistancefromStart'] <= 4): features['DistancefromStart'] += 1
        if (features['DistancefromStart'] == 1):
            features['DistancefromStart'] == -9999
        features['DistancefromStart'] *= 2.5
        features['stayApart'] = self.getMazeDistance(gameState.getAgentPosition(teamNums[0]),
                                                     gameState.getAgentPosition(teamNums[1]))
        features['onDefense'] = 1
        features['offenseFood'] = 0

        if myState.isPacman:
            features['onDefense'] = -1

        if (len(invaders) == 0 and successor.getScore() != 0):
            features['onDefense'] = -1
            features['offenseFood'] = min(
                [self.getMazeDistance(myPos, food) for food in self.getFood(successor).asList()])
            features['foodCount'] = len(self.getFood(successor).asList())
            features['DistancefromStart'] = 0
            features['stayAprts'] += 2
            features['stayApart'] *= features['stayApart']
        if (len(invaders) != 0):
            features['stayApart'] = 0
            features['DistancefromStart'] = 0
        return features

    def getWeights(self, gameState, action):
        return {'foodCount': -20, 'offenseFood': -1, 'DistancefromStart': 3, 'numInvaders': -40000, 'onDefense': 20,
                'stayApart': 45, 'invaderDistance': -1800, 'stop': -400, 'reverse': -250}
