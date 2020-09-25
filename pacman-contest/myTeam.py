# myTeam.py
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions
import copy


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    This is the ancestor class for the agent we used.
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.midWidth = gameState.data.layout.width / 2
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        self.midHeight = gameState.data.layout.height / 2
        self.foodEaten = 0
        self.initialnumberOfFood = len(self.getFood(gameState).asList())
        self.lastEatenFoodPosition = None

        scanmap = ScanMap(gameState, self)
        foodList = scanmap.getFoodList(gameState)
        self.safeFoods = scanmap.getSafeFoods(foodList)  # a list of tuple contains safe food location
        self.dangerFoods = scanmap.getDangerFoods(self.safeFoods)

        # for food in self.safeFoods:
        #   self.debugDraw(food, [100, 100, 255], False)
        # for food in self.dangerFoods:
        #   self.debugDraw(food, [255, 100, 100], False)

        self.blueRebornHeight = self.height - 1
        self.blueRebornWidth = self.width - 1
        self.edgeList = self.getHomeEdges(gameState)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor (Game state object)
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        self.locationOfLastEatenFood(gameState)  # detect last eaten food
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def distToFood(self, gameState):
        ''''
        get the nearestFood, remember to edit again, return  [nearestFood, nearestDistance]
        '''
        food = self.getFood(gameState).asList()
        if len(food) > 0:
            dist = 9999
            for a in food:
                tempDist = self.getMazeDistance(gameState.getAgentPosition(self.index), a)
                if tempDist < dist:
                    dist = tempDist
                    temp = a
            return dist
        else:
            return 0

    def distToHome(self, gameState):
        ''''
        return the distance to nearest boudndary
        '''
        myState = gameState.getAgentState(self.index)
        myPosition = myState.getPosition()
        boundaries = []
        if self.red:
            i = self.midWidth - 1
        else:
            i = self.midWidth + 1
        boudaries = [(i, j) for j in range(self.height)]
        validPositions = []
        for i in boudaries:
            if not gameState.hasWall((int)(i[0]), (int)(i[1])):
                validPositions.append(i)
        dist = 9999
        for validPosition in validPositions:
            tempDist = self.getMazeDistance(validPosition, myPosition)
            if tempDist < dist:
                dist = tempDist
                temp = validPosition
        return dist

    def boundaryPosition(self, gameState):
        ''''
        return a list of positions of boundary
        '''
        myState = gameState.getAgentState(self.index)
        myPosition = myState.getPosition()
        boundaries = []
        if self.red:
            i = self.midWidth - 1
        else:
            i = self.midWidth + 1
        boudaries = [(i, j) for j in range(self.height)]
        validPositions = []
        for i in boudaries:
            if not gameState.hasWall((int)(i[0]), (int)(i[1])):
                validPositions.append(i)
        return validPositions

    def distToCapsule(self, gameState):
        ''''
        return the nerest distance to capsule
        '''
        if len(self.getCapsules(gameState)) > 1:
            dist = 9999
            for i in self.getCapsules(gameState):
                tempDist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), i)
                if tempDist < dist:
                    dist = tempDist
                    self.debugDraw(i, [125, 125, 211], True)
            return dist

        elif len(self.getCapsules(gameState)) == 1:
            distToCapsule = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                                 self.getCapsules(gameState)[0])
            self.debugDraw(self.getCapsules(gameState)[0], [125, 125, 211], True)
            return distToCapsule

    def locationOfLastEatenFood(self, gameState):
        ''''
        return the location of the last eaten food
        '''
        if len(self.observationHistory) > 1:
            prevState = self.getPreviousObservation()
            prevFoodList = self.getFoodYouAreDefending(prevState).asList()
            currentFoodList = self.getFoodYouAreDefending(gameState).asList()
            if len(prevFoodList) != len(currentFoodList):
                for food in prevFoodList:
                    if food not in currentFoodList:
                        self.lastEatenFoodPosition = food

    def getNearestGhostDistance(self, gameState):
        ''''
        return the distance of the nearest ghost
        '''
        myPosition = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in ghosts]
            return min(dists)
        else:
            return None

    def getNearestinvader(self, gameState):
        ''''
        return the distance of the nearest ghost
        '''
        myPosition = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in invaders]
            return min(dists)
        else:
            return None

    def opponentscaredTime(self, gameState):
        opponents = self.getOpponents(gameState)
        for opponent in opponents:
            if gameState.getAgentState(opponent).scaredTimer > 1:
                return gameState.getAgentState(opponent).scaredTimer

        return 0

    def nullHeuristic(self, state, problem=None):
        return 0

    """
    A Genral start search that can be used to solve any problem and any heuristics 
    """

    def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        from util import PriorityQueue
        start_state = problem.getStartState()
        # store the fringe use priority queue to ensure pop out lowest cost
        fringe = PriorityQueue()
        h = heuristic(start_state, gameState)
        g = 0
        f = g + h
        start_node = (start_state, [], g)
        fringe.push(start_node, f)
        explored = []
        while not fringe.isEmpty():
            current_node = fringe.pop()
            state = current_node[0]
            path = current_node[1]
            current_cost = current_node[2]
            if state not in explored:
                explored.append(state)
                if problem.isGoalState(state):
                    return path
                successors = problem.getSuccessors(state)
                for successor in successors:
                    current_path = list(path)
                    successor_state = successor[0]
                    move = successor[1]
                    g = successor[2] + current_cost
                    h = heuristic(successor_state, gameState)
                    if successor_state not in explored:
                        current_path.append(move)
                        f = g + h
                        successor_node = (successor_state, current_path, g)
                        fringe.push(successor_node, f)
        return []

    def GeneralHeuristic(self, state, gameState):

        """

        This heuristic is used for to avoid ghoost, we give the
        position which close to ghost a higher heuristic to avoid
        colission with ghost

        """
        heuristic = 0
        if self.getNearestGhostDistance(gameState) != None:
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            # pacmans = [a for a in enemies if a.isPacman]
            ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() != None]
            if ghosts != None and len(ghosts) > 0:
                ghostpositions = [ghost.getPosition() for ghost in ghosts]
                # pacmanPositions = [pacman.getPosition() for pacman in pacmans]
                ghostDists = [self.getMazeDistance(state, ghostposition) for ghostposition in ghostpositions]
                ghostDist = min(ghostDists)
                if ghostDist < 2:
                    # print ghostDist
                    heuristic = pow((5 - ghostDist), 5)

        return heuristic

    leftMoves = 300
    lastLostFoodPostion = (0, 0)
    lastLostFoodEffect = 0  # use to measure the effective of last lost food position
    corners = []
    edgeList = []

    def updateSafeFood(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            distanceToGhost = self.getDistanceToGhost(gameState)
            self.corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)
            self.safeFoods = self.getSafeFood(gameState)
            self.dangerFoods = self.getDangerFood(gameState)
        else:
            self.corners = self.removeCornersBaseOnDistance(gameState, 15)
            self.safeFoods = self.getSafeFood(gameState)
            self.dangerFoods = self.getDangerFood(gameState)
        '''
              for food in self.safeFoods:
            self.debugDraw(food, [100, 100, 255], False)
        for food in self.dangerFoods:
            self.debugDraw(food, [255, 100, 100], False)
        '''

    def getDistanceToHome(self, gameState):

        myPos = gameState.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in self.edgeList])
        return minDistanceToHome

    def getDistanceToGhost(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        if len(invaders) == 0:
            return 999
        if len(invaders) > 0:
            distanceToGhost = min(
                [self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                 for i in invaders])
            return distanceToGhost

    def getHomeEdges(self, gameState):
        # use to calculate the min distance to home
        edgeList = []
        height = gameState.data.layout.height
        length = gameState.data.layout.width

        if self.red:
            edge = (int)((length - 1) / 2)
            i = 0
            while i < height - 1:
                i += 1
                if not gameState.hasWall(edge, i):
                    edgeList.append((edge, i))
        else:
            edge = (int)((length + 1) / 2)
            i = 0
            while i < height - 1:
                i += 1
                if not gameState.hasWall(edge, i):
                    edgeList.append((edge, i))
        return edgeList

    def getLostFood(self, gameState):
        currentFood = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
        if (self.getPreviousObservation() is not None):
            previousFood = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()

        else:
            return (0, 0)
        if len(currentFood) < len(previousFood):
            for i in previousFood:

                if i not in currentFood:
                    return i
        else:
            return (0, 0)

    def getSafeFood(self, gameState):
        allcroners = self.corners
        foodList = self.getFood(gameState).asList()
        safeFood = []
        for i in foodList:
            if i not in allcroners:
                safeFood.append(i)
        return safeFood

    def getDangerFood(self, gameState):
        self.dangerFoods = []
        foodList = self.getFood(gameState).asList()
        for food in foodList:
            if food not in self.safeFoods:
                self.dangerFoods.append(food)
        return self.dangerFoods

    def getScareTime(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()

        minDistance = 400

        for i in self.getOpponents(gameState):
            if not gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None:
                if self.distancer.getDistance(gameState.getAgentState(i).getPosition(), myPos) < minDistance:
                    minDistance = self.distancer.getDistance(gameState.getAgentState(i).getPosition(), myPos)
                    ghostIndex = i
        try:
            return gameState.getAgentState(ghostIndex).scaredTimer
        except:
            return 0

    def repeatActionDetect(self):
        repeatTimes = 3
        count = 1
        testLength = 4
        breakFlag = False
        j = 1
        while j < 7:
            j += 1
            testLength = j
            goStrightFlag = True

            if len(self.historyAction) == 0 or len(self.historyAction) < (repeatTimes + 1) * testLength:
                return False
            while True:
                i = 0
                while i < testLength:

                    if not self.historyAction[len(self.historyAction) - count * testLength - i - 1] == \
                           self.historyAction[
                               len(self.historyAction) - i - 1]:
                        breakFlag = True
                        break
                    if count > repeatTimes - 1:
                        breakFlag = True
                        break
                    i += 1
                if breakFlag:
                    breakFlag = False
                    break

                count += 1
            k = 0
            while k < j:
                k += 1
                if not self.historyAction[
                           len(self.historyAction) - 1] == self.historyAction[
                           len(self.historyAction) - k - 1]:
                    goStrightFlag = False
            if count > repeatTimes - 1 and not goStrightFlag:
                return True
            else:
                return False

    def getDistanceToCenter(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - (height + 1) / 2 < nearest):
                centerLocation = location
                nearest = abs(location[1] - (height + 1) / 2 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    def getDistanceToTop(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - height + 1) < nearest:
                centerLocation = location
                nearest = abs(location[1] - height + 1) < nearest
        return self.getMazeDistance(myPos, centerLocation)

    def distanceToHighEntry(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - 3 * (height + 1) / 4 < nearest):
                centerLocation = location
                nearest = abs(location[1] - 3 * (height + 1) / 4 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    def getHighEntry(self, gameState):

        height = gameState.data.layout.height
        entryPoint = ()
        nearest = height
        myPos = gameState.getAgentState(self.index).getPosition()
        for location in self.edgeList:
            if abs(location[1] - 3 * (height + 1) / 4 < nearest):
                centerLocation = location
                nearest = abs(location[1] - 3 * (height + 1) / 4 < nearest)
                entryPoint = location

        return entryPoint

    def distanceToLowEntry(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - 1 * (height + 1) / 4 < nearest):
                centerLocation = location
                nearest = abs(location[1] - 1 * (height + 1) / 4 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    # This is the function that find the corners in layout
    def removeAllCorners(self, gameState):
        cornerList = []

        myPos = gameState.getAgentState(self.index).getPosition()
        height = gameState.data.layout.height
        length = gameState.data.layout.width

        loopTimes = 15
        while loopTimes > 0:
            loopTimes -= 1
            i = 1
            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three directionΩ
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in cornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in cornerList:
                            cornerList.append((i, j))
                    j += 1
                i += 1
        return cornerList

    def removeCornersBaseOnDistance(self, gameState, distanceToGhost):
        cornerList = []
        removeCornerList = []
        removeCornerListCopy = []
        capsuleList = self.getCapsules(gameState)
        myPos = gameState.getAgentState(self.index).getPosition()
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        loopTimes = 0
        loopTimes = 1 + (distanceToGhost - 4) / 2

        while loopTimes >= 1:
            loopTimes -= 1
            i = 1

            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three directionΩ
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in removeCornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in removeCornerList:
                            removeCornerListCopy.append((i, j))
                    j += 1
                i += 1
                for x in removeCornerListCopy:
                    if x not in removeCornerList:
                        removeCornerList.append(x)
        loopTimes = 30

        while loopTimes > 0:
            loopTimes -= 1
            i = 1
            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three directionΩ
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in cornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in cornerList and (i, j) not in capsuleList:
                            cornerList.append((i, j))
                    j += 1
                i += 1

        for i in removeCornerList:
            try:
                cornerList.remove(i)
            except:
                continue
        cornerDeep = 15
        while cornerDeep > 0:
            cornerDeep -= 1
            for corner in removeCornerList:
                i = corner[0]
                j = corner[1]
                numberOfWalls = 0
                if (i + 1, j) in cornerList:
                    numberOfWalls += 1
                if (i - 1, j) in cornerList:
                    numberOfWalls += 1
                if (i, j + 1) in cornerList:
                    numberOfWalls += 1
                if (i, j - 1) in cornerList:
                    numberOfWalls += 1
                if numberOfWalls >= 1 and (i, j) not in cornerList and (i, j) not in capsuleList:
                    cornerList.append((i, j))
        return cornerList

    def removeCorners(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        cornerList = []
        height = gameState.data.layout.height
        length = gameState.data.layout.width

        loopTimes = 10
        while loopTimes > 0:
            loopTimes -= 1
            i = 1
            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasFood(i, j) or gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three direction
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in cornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in cornerList:
                            cornerList.append((i, j))

                    j += 1
                i += 1
        return cornerList

    def stopAction(self):
        features = util.Counter()
        features['stop'] = 100000
        return features

    def justEatFood(self, gameState):
        if len(self.getFood(gameState).asList()) > 0:
            problem = SearchFood(gameState, self, self.index)
            return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
        else:
            return self.goHome(gameState)

    def eatSafeFood(self, gameState):
        problem = SearchSafeFood(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)
        else:
            finalAction = self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
        return finalAction

    def eatCapsule(self, gameState):
        problem = SearchCapsule(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)
        else:
            finalAction = self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
        return finalAction

    def goHome(self, gameState):

        problem = SearchEscape(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)
        else:
            finalAction = self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
        return finalAction

    def escape(self, gameState):
        problem = SearchEscape(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)
        else:
            finalAction = self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
        return finalAction

    ###########################################
    #            Astar Attacker               #
    ###########################################
    """
  
    Below is the a* attacker agent
  
    """


class OffensiveReflexAgent(DummyAgent):
    escapeEffect = 0
    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    repeatFlag = 0
    isOffensive = True
    historyAction = []
    goOffensive = True

    def recordInformation(self, gameState):
        self.leftMoves -= 1
        corners = self.removeCorners(gameState)
        if self.lastLostFoodEffect > 0:
            self.lastLostFoodEffect -= 1
        if self.escapeEffect > 0:
            self.escapeEffect -= 1

        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.repeatFlag = 0
            self.carryDots = 0
            self.backHomeTimes += 1
            self.escapeEffect = 0

    def recordInformationAfterCurrentStep(self, gameState, finalAction):
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.repeatFlag = 0
            self.escapeEffect = 0

        self.historyAction.append(finalAction)

    def chooseAction(self, gameState):

        start = time.time()

        self.recordInformation(gameState)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        minDistanceToHome = self.getDistanceToHome(gameState)
        distanceToGhost = self.getDistanceToGhost(gameState)

        self.updateSafeFood(gameState)
        if len(self.safeFoods) > 0:
            minDistanceToFood = min([self.distancer.getDistance(food, gameState.getAgentState(self.index).getPosition())
                                     for food in self.safeFoods])
        else:
            minDistanceToFood = 99

        scaredTimes = self.getScareTime(gameState)
        if len(self.getFood(gameState).asList()) <= 2:
            finalAction = self.goHome(gameState)
        elif scaredTimes > 3:
            # when the oppsite is scared, we just eat food
            finalAction = self.justEatFood(gameState)
        elif len(invaders) == 0:
            finalAction = self.justEatFood(gameState)
            # eat capsule 往后放
        elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) != 0 and self.opponentscaredTime(
                gameState) < 10:
            finalAction = self.eatCapsule(gameState)

        elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) == 0 and gameState.getAgentState(
                self.index).numCarrying > 1:
            finalAction = self.goHome(gameState)

        elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) > 0):
            finalAction = self.eatSafeFood(gameState)

        elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) == 0):
            finalAction = self.justEatFood(gameState)

        elif len(self.getFood(gameState).asList()) < 3 or gameState.data.timeleft < self.distToHome(gameState) + 10 \
                or gameState.getAgentState(
            self.index).numCarrying > 9 + self.backHomeTimes * 8 and minDistanceToFood > 5:
            finalAction = self.goHome(gameState)

        elif distanceToGhost < 5 and len(self.getCapsules(gameState)) > 0:
            finalAction = self.eatCapsule(gameState)
            # 这里如果ghost在去往capsule的路上会出现问题
        elif len(self.getSafeFood(gameState)) > 0:
            finalAction = self.eatSafeFood(gameState)
        elif len(self.getCapsules(gameState)) > 0:
            finalAction = self.eatCapsule(gameState)
        elif len(self.getCapsules(gameState)) == 0:
            if self.carryDots > 0:
                finalAction = self.goHome(gameState)
            else:
                finalAction = self.justEatFood(gameState)
        elif distanceToGhost < 3:
            finalAction = self.escape(gameState)

        elif self.leftMoves < minDistanceToHome:
            if len(self.getCapsules(gameState)) > 0:
                finalAction = self.eatCapsule(gameState)
            else:
                finalAction = self.goHome(gameState)
                # 这里可以尝试用自杀回家防守优化

        else:
            finalAction = self.justEatFood(gameState)
        self.recordInformationAfterCurrentStep(gameState, finalAction)
        print("Time", self.index, time.time() - start)
        return finalAction
    def getFeatures(self, gameState, action):

        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        ourFoodLeft = self.getFoodYouAreDefending(gameState).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)

        ourFoodList = self.getFoodYouAreDefending(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice

        if not self.isOffensive:
            return self.getFeaturesAsDefensive(gameState, action)

        if not successor.getAgentState(self.index).isPacman:
            features['distanceToEntry'] = self.distanceToHighEntry(gameState, myPos)
        # initial the score
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['finalDistanceToHome'] = 100 - minDistanceToHome
        features['leftCapsules'] = 100 - len(self.getCapsules(successor))
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # get the feature to the closest food
        if len(foodList) > 0:  # and self.repeatFlag==0 :  # This should always be True,  but better safe than sorry
            features['distanceToFood'] = self.distributeDots(successor)
        else:
            if (int)(self.repeatFlag / 7) < len(foodList):
                features['distanceToFood'] = self.distancer.getDistance(foodList[(int)(self.repeatFlag / 7)], myPos)
            else:
                features['distanceToFood'] = 0
                features['distanceToHome'] = 100 - minDistanceToHome

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        if myPos in self.corners:
            features['inCorner'] = 1

        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if action == 'Stop':
            try:
                x = min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                         for i in invaders])
                if x != 2:
                    return self.stopAction()
            except:
                return self.stopAction()
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = self.getScareTime(successor)
        if scaredTimes > 3:
            # when the oppsite is scared, we just eat food
            features['inCorner'] = 0
            features['distanceToGhost'] = 0
            features['distanceToHome'] = 0
        elif scaredTimes <= 2:
            # when the oppsite is not scared
            if self.repeatFlag > 0 and self.carryDots > 0:
                return self.getFeaturesGoHome(gameState, action)

            if self.repeatFlag > 0:
                return self.getFeaturesLowerScoreRepeat(gameState, action)

            if len(invaders) == 0:
                distanceToGhost = 0
                successroDistanceToGhost = 0
            if len(invaders) > 0:
                successroDistanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                                for i in invaders])
                distanceToGhost = min(
                    [self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                     for i in invaders])
                for i in invaders:
                    if myPos == i.getPosition():
                        features['meetGhost'] = 1
                if distanceToGhost < 5:
                    try:
                        distanceToCapsule = 5 * min(
                            [self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                        features['distanceToCapsule'] = distanceToCapsule

                    except:
                        distanceToCapsule = -1
                try:
                    distanceToCapsule = min(
                        [self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = distanceToCapsule

                except:
                    distanceToCapsule = -1
            if distanceToGhost < 15:
                # If we found ghost, all the corners should be avoid
                corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)

                if myPos in corners:
                    features['inCorner'] = 1

            if distanceToGhost < 8:
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = (distanceToCapsule) * 80
                except:
                    features['distanceToHome'] = - minDistanceToHome
            if distanceToGhost < 6:
                features['distanceToGhost'] = 100 - successroDistanceToGhost
                features['successorScore'] = 0

        if self.leftMoves < minDistanceToHome + 4 and self.carryDots > 0:
            # should go home directly

            features['distanceToHome'] = - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves < minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] = (distanceToCapsule) * 80
            except:
                distanceToCapsule = -1
        if self.carryDots > 1 + 7 * self.backHomeTimes:
            features['distanceToHome'] = - minDistanceToHome
        if (not successor.getAgentState(self.index).isPacman):
            features['distanceToGhost'] = 0

        return features

    def getFeaturesAsDefensive(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # when it is defensive but is a pacman, go home firstly
        if gameState.getAgentState(self.index).isPacman:
            return self.getFeaturesGoHome(gameState, action)
        # Computes whether we're on defense (1) or offense (0)
        if not successor.getAgentState(self.index).isPacman:
            features['onDefense'] = 1

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = 200 - min(dists)
        if len(invaders) == 0:
            distanceToCenter = self.getDistanceToCenter(gameState, myPos)
            try:
                features['distanceToCenter'] = self.getMazeDistance(myPos,
                                                                    self.getCapsulesYouAreDefending(gameState)[0])
            except:
                features['distanceToCenter'] = self.getDistanceToTop(gameState, myPos)

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        distanceToLostFood = 0
        # before find the invaders, we use  lost food position to estimate the position of invaders
        if len(invaders) == 0:
            lostFoodPosition = self.getLostFood(gameState)
            if lostFoodPosition != (0, 0):
                distanceToLostFood = self.getMazeDistance(myPos, lostFoodPosition)
                self.lastLostFoodPostion = lostFoodPosition
                self.lastLostFoodEffect = 10
            else:
                if self.lastLostFoodPostion != (0, 0) and self.lastLostFoodEffect > 0:
                    distanceToLostFood = self.getMazeDistance(myPos, self.lastLostFoodPostion)
        features['distanceToLostFood'] = distanceToLostFood
        return features

    def getFeaturesLowerScoreRepeat(self, gameState, action):
        print("getFeaturesLowerScoreRepeat")
        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)
        myPos = successor.getAgentState(self.index).getPosition()

        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice
        '''
        if self.repeatActionDetect() or self.changeEntryEffect > 0:
            # features['changeEntryPoint']=self.changeEntryPoint(edgeList,gameState,myPos)
            self.changeEntryEffect = 10
        '''
        # initial the score
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['finalDistanceToHome'] = - minDistanceToHome
        features['leftCapsules'] = 100 - len(self.getCapsules(successor))

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # get the feature to the closest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            safeFoodList = self.getSafeFood(gameState)
            print(safeFoodList)
            try:
                features['distanceToFood'] = min([self.distancer.getDistance(food, myPos)
                                                  for food in safeFoodList])
            except:
                features['distanceToFood'] = min([self.distancer.getDistance(food, myPos)
                                                  for food in foodList])
        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        if myPos in self.corners:
            features['inCorner'] = 1

        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间

        if action == 'Stop':
            try:
                x = min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                         for i in invaders])
                if x != 2:
                    return self.stopAction()
            except:
                return self.stopAction()
        scaredTimes = self.getScareTime(successor)
        if scaredTimes > 3:
            # when the oppsite is scared, we just eat food
            features['inCorner'] = 0
            features['distanceToGhost'] = 0

            features['distanceToHome'] = 0
        elif scaredTimes <= 2:
            # when the oppsite is not scared

            if len(invaders) == 0:
                distanceToGhost = 0
                successroDistanceToGhost = 0
            if len(invaders) > 0:
                successroDistanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                                for i in invaders])
                distanceToGhost = min(
                    [self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                     for i in invaders])

                for i in invaders:
                    if myPos == i.getPosition():
                        features['meetGhost'] = 1
                if distanceToGhost < 5:
                    try:
                        distanceToCapsule = 5 * min(
                            [self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                        features['distanceToCapsule'] = distanceToCapsule

                    except:
                        distanceToCapsule = -1
                try:
                    distanceToCapsule = min(
                        [self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = distanceToCapsule

                except:
                    distanceToCapsule = -1

            if distanceToGhost < 13:
                # If we found ghost, all the corners should be avoid
                corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)

                if myPos in corners:
                    features['inCorner'] = 1

            if distanceToGhost < 3:
                features['distanceToGhost'] = 100 - successroDistanceToGhost
                features['successorScore'] = 0

        if self.leftMoves < minDistanceToHome + 4 and self.carryDots > 0 or self.carryDots > 0 + self.backHomeTimes * abs(
                self.getScore(gameState)):
            # should go home directly

            features['distanceToHome'] = - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves < minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] = (distanceToCapsule) * 80
            except:
                distanceToCapsule = -1
        if (not successor.getAgentState(self.index).isPacman):
            features['distanceToGhost'] = 0

        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'leftCapsules': 100,  # Eat capsule
                    'distanceToGhost': -100, 'finalDistanceToHome': 5, 'distanceToCapsule': -1,
                    # distance attribute when come back home
                    'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'distanceToLostFood': -5,
                    'distanceToCenter': -3,  # defensive attribute
                    'stop': -10, 'inCorner': -100000, 'reverse': -2}
        return {'successorScore': 1000, 'leftCapsules': 200,  # Eat food or capsule when it can
                'distanceToGhost': -100, 'distanceToHome': 60, 'distanceToFood': -2, 'distanceToCapsule': -1,
                'distanceToEntry': -20,  # distance attribute when it is pacman
                'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'distanceToLostFood': -5,
                'distanceToCenter': -3,  # defensive attribute
                'inCorner': -100000, 'stop': -10, 'reverse': -2, 'meetGhost': -1000000,
                'changeEntryPoint': 1000}

#######################
#  Astart_defender   #
#######################

class DefensiveReflexAgent(DummyAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    escapeEffect = 0
    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    repeatFlag = 0
    isOffensive = True
    historyAction = []
    goOffensive = True

    def recordInformation(self, gameState):
        self.leftMoves -= 1
        if self.lastLostFoodEffect > 0:
            self.lastLostFoodEffect -= 1
        if self.escapeEffect > 0:
            self.escapeEffect -= 1

        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.repeatFlag = 0
            self.carryDots = 0
            self.backHomeTimes += 1
            self.escapeEffect = 0

        lostFoodPosition = self.getLostFood(gameState)

        if lostFoodPosition != (0, 0):
            self.lastLostFoodPostion = lostFoodPosition
            self.lastLostFoodEffect = 10

    def recordInformationAfterCurrentStep(self, gameState, finalAction):
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.repeatFlag = 0
            self.escapeEffect = 0

        self.historyAction.append(finalAction)

    def chooseAction(self, gameState):
        start = time.time()
        self.recordInformation(gameState)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        self.locationOfLastEatenFood(gameState)  # detect last eaten food
        actions = gameState.getLegalActions(self.index)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman]
        knowninvaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        # if number of invaders is less than two, we can go out and try to eat some food
        print(gameState.data.timeleft)
        # when number of invader > 0, we excute defendense strategy
        if gameState.data.timeleft > 1050:

            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)
        elif gameState.getAgentState(self.index).scaredTimer > 5 or len(invaders) < 1:

            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

            minDistanceToHome = self.getDistanceToHome(gameState)
            distanceToGhost = self.getDistanceToGhost(gameState)

            self.updateSafeFood(gameState)
            if len(self.safeFoods) > 0:
                minDistanceToFood = min(
                    [self.distancer.getDistance(food, gameState.getAgentState(self.index).getPosition())
                     for food in self.safeFoods])
            else:
                minDistanceToFood = 99
            scaredTimes = self.getScareTime(gameState)

            if scaredTimes > 3:
                # when the oppsite is scared, we just eat food
                finalAction = self.justEatFood(gameState)
            elif len(invaders) == 0:
                finalAction = self.justEatFood(gameState)

                # eat capsule 往后放
            elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) != 0 and self.opponentscaredTime(
                    gameState) < 10:
                finalAction = self.eatCapsule(gameState)

            elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) == 0 and gameState.getAgentState(
                    self.index).numCarrying > 1:
                finalAction = self.goHome(gameState)

            elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) > 0):
                finalAction = self.eatSafeFood(gameState)

            elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) == 0):
                finalAction = self.justEatFood(gameState)

            elif len(self.getFood(gameState).asList()) < 3 or gameState.data.timeleft < self.distToHome(gameState) + 30 \
                    or gameState.getAgentState(
                self.index).numCarrying > 9 + self.backHomeTimes * 8 and minDistanceToFood > 5:
                finalAction = self.goHome(gameState)

            elif distanceToGhost < 5 and len(self.getCapsules(gameState)) > 0:
                finalAction = self.eatCapsule(gameState)
                # 这里如果ghost在去往capsule的路上会出现问题
            elif len(self.getSafeFood(gameState)) > 0:
                finalAction = self.eatSafeFood(gameState)
            elif len(self.getCapsules(gameState)) > 0:
                finalAction = self.eatCapsule(gameState)
            elif len(self.getCapsules(gameState)) == 0:
                if self.carryDots > 0:
                    finalAction = self.goHome(gameState)
                else:
                    finalAction = self.justEatFood(gameState)
            elif distanceToGhost < 3:
                finalAction = self.escape(gameState)

            elif self.leftMoves < minDistanceToHome:
                if len(self.getCapsules(gameState)) > 0:
                    finalAction = self.eatCapsule(gameState)
                else:
                    finalAction = self.goHome(gameState)
                    # 这里可以尝试用自杀回家防守优化

            else:
                finalAction = self.justEatFood(gameState)

        elif gameState.getAgentState(self.index).isPacman:
            finalAction = self.goHome(gameState)
        else:

            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)

        print("Time", self.index, time.time() - start)

        self.recordInformationAfterCurrentStep(gameState, finalAction)
        return finalAction

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)

        if action == 'Stop':
            features['stop'] = 1000
        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = 100 - min(dists)
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # before find the invaders, we use  lost food position to estimate the position of invaders
        if len(invaders) == 0:
            if self.lastLostFoodPostion != (0, 0) and self.lastLostFoodEffect > 0:
                features['distanceToLostFood'] = self.getMazeDistance(myPos, self.lastLostFoodPostion)
            elif self.lastLostFoodEffect == 0:
                features['distanceToCenter'] = self.getDistanceToCenter(gameState, myPos)

        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'leftCapsules': 100,  # Eat capsule
                    'distanceToGhost': -100, 'finalDistanceToHome': 5, 'distanceToCapsule': -1,
                    # distance attribute when come back home
                    'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'distanceToLostFood': -5,
                    'distanceToCenter': -3,  # defensive attribute
                    'stop': -10, 'inCorner': -100000, 'reverse': -2}
        return {'successorScore': 1000, 'leftCapsules': 200,  # Eat food or capsule when it can
                'distanceToGhost': -100, 'distanceToHome': 60, 'distanceToFood': -2, 'distanceToCapsule': -1,
                'distanceToEntry': -20,  # distance attribute when it is pacman
                'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'distanceToLostFood': -20,
                'distanceToCenter': -10,  # defensive attribute
                'inCorner': -100000, 'stop': -10, 'reverse': -2, 'meetGhost': -1000000,
                'changeEntryPoint': 1000}


##############################################################
#  Helper Class  ----- defines multiple  Search problem      #
##############################################################


class PositionSearchProblem:
    """
    It is the ancestor class for all the search problem class.
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point.
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class SearchFood(PositionSearchProblem):
    """
     The goal state is to find all the food
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        # return state in self.food.asList() or state in self.capsule
        return state in self.food.asList()


class SearchFoodNotInCorners(PositionSearchProblem):
    """
       The goal state is to find all the food
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.foodNotInCorners = agent.getSafeFood(gameState)

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        # return state in self.food.asList() or state in self.capsule
        return state in self.foodNotInCorners


class SearchSafeFood(PositionSearchProblem):
    """
    The goal state is to find all the safe fooof
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.safeFood = agent.safeFoods

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        # return state in self.food.asList() or state in self.capsule
        return state in self.safeFood


class SearchDangerousFood(PositionSearchProblem):
    """
    Used to get the safe food
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.dangerousFood = agent.dangerFoods

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        # return state in self.food.asList() or state in self.capsule
        return state in self.dangerousFood


class SearchEscape(PositionSearchProblem):
    """
    Used to escape
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.homeBoundary = agent.boundaryPosition(gameState)
        self.safeFood = agent.safeFoods

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsule
        return state in self.homeBoundary or state in self.capsule


class SearchEntry(PositionSearchProblem):
    """
    Used to go back home
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.entry = agent.getHighEntry(gameState)

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsule
        return state in self.entry


class SearchHome(PositionSearchProblem):
    """
    Used to go back home
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.homeBoundary = agent.boundaryPosition(gameState)

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsule
        return state in self.homeBoundary


class SearchCapsule(PositionSearchProblem):
    """
    Used to search capsule
    """

    def __init__(self, gameState, agent, agentIndex=0):
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        # the goal state is the location of capsule
        return state in self.capsule


class SearchLastEatenFood(PositionSearchProblem):
    """
    Used to search capsule
    """

    def __init__(self, gameState, agent, agentIndex=0):
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.lastEatenFood = agent.lastEatenFoodPosition
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        # the goal state is the location of capsule
        return state == self.lastEatenFood


class SearchInvaders(PositionSearchProblem):
    """
    Used to search capsule
    """

    def __init__(self, gameState, agent, agentIndex=0):
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.lastEatenFood = agent.lastEatenFoodPosition
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.enemies = [gameState.getAgentState(agentIndex) for agentIndex in agent.getOpponents(gameState)]
        self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition != None]
        if len(self.invaders) > 0:
            self.invadersPosition = [invader.getPosition() for invader in self.invaders]
        else:
            self.invadersPosition = None

    def isGoalState(self, state):
        # # the goal state is the location of invader
        return state in self.invadersPosition


##########################
#    Scan Map Method     #
##########################


class ScanMap:
    """
    A Class Below is used for scanning the map to find
    Safe food and dangerousfood

    Note: Safe food is the food whitin the position can has
    at least two ways home
    """

    def __init__(self, gameState, agent):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState).asList()
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.homeBoundary = agent.boundaryPosition(gameState)
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width

    def getFoodList(self, gameState):
        foods = []
        for food in self.food:
            food_fringes = []
            food_valid_fringes = []
            count = 0
            food_fringes.append((food[0] + 1, food[1]))
            food_fringes.append((food[0] - 1, food[1]))
            food_fringes.append((food[0], food[1] + 1))
            food_fringes.append((food[0], food[1] - 1))
            for food_fringe in food_fringes:
                if not gameState.hasWall((int)(food_fringe[0]), (int)(food_fringe[1])):
                    count = count + 1
                    food_valid_fringes.append(food_fringe)
            if count > 1:
                foods.append((food, food_valid_fringes))
        return foods

    def getSafeFoods(self, foods):
        safe_foods = []
        for food in foods:
            count = self.getNumOfValidActions(food)
            if count > 1:
                safe_foods.append(food[0])
        return safe_foods

    def getDangerFoods(self, safe_foods):
        danger_foods = []
        for food in self.food:
            if food not in safe_foods:
                danger_foods.append(food)
        return danger_foods

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    def isGoalState(self, state):
        return state in self.homeBoundary

    def getNumOfValidActions(self, foods):
        food = foods[0]
        food_fringes = foods[1]
        visited = []
        visited.append(food)
        count = 0
        for food_fringe in food_fringes:
            closed = copy.deepcopy(visited)
            if self.BFS(food_fringe, closed):
                count = count + 1
        return count

    def BFS(self, food_fringe, closed):
        from util import Queue

        fringe = Queue()
        fringe.push((food_fringe, []))
        while not fringe.isEmpty():
            state, actions = fringe.pop()
            closed.append(state)
            if self.isGoalState(state):
                return True
            for successor, direction in self.getSuccessors(state):
                if successor not in closed:
                    closed.append(successor)
                    fringe.push((successor, actions + [direction]))