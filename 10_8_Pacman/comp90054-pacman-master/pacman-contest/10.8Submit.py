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


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    leftMoves = 300
    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    reward = 0
    negetiveReward = 0
    startPostion = (0, 0)
    passedPositionWeight = util.Counter()

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.refreshPassedPositionWeight(gameState)

    def refreshPassedPositionWeight(self, gameState):

        height, length = self.getHightAndLength(gameState)
        i = 0

        while i < height:
            j = 0
            while j < length:
                if not gameState.hasWall((int)(j), (int)(i)):
                    self.passedPositionWeight[(j, i)] = 0
                j += 1
            i += 1

    def NStep(self, gameState, step):
        i = 0
        actions = gameState.getLegalActions(self.index)

        learningFactor = 0.1
        if step > 1:
            values = [
                self.evaluate(gameState, action) + learningFactor * (float)(
                    self.NStep(self.getSuccessor(gameState, action),
                               step - 1)[0]) for action in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return maxValue, bestActions
        elif step == 1:
            values = [self.evaluate(gameState, action) for action in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return maxValue, bestActions

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        print(self.passedPositionWeight)
        myPos = gameState.getAgentState(self.index).getPosition()
        self.passedPositionWeight[myPos] += 1
        if self.startPostion == myPos:
            self.negetiveReward += 1

        if self.leftMoves == 300:
            self.startPostion = myPos

        self.leftMoves -= 1
        actions = gameState.getLegalActions(self.index)
        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.carryDots = 0
            self.backHomeTimes += 1
            self.reward += 1
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        manhattanDistanceToGhost = 0
        if len(invaders) > 0:
            manhattanDistanceToGhost = min([distanceCalculator.manhattanDistance(i.getPosition(), myPos)
                                            for i in invaders])

        if len(invaders) == 0:
            self.refreshPassedPositionWeight(gameState)
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)

        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        finalAction = random.choice(bestActions)

        foodLeft = len(self.getFood(gameState).asList())

        finalAction = random.choice(bestActions)
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.reward += 1
            self.refreshPassedPositionWeight(gameState)
        return finalAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

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

    def getHightAndLength(self, gameState):
        i = 0
        while i < 100:
            try:
                gameState.hasWall(1, i)
                i += 1
            except:
                height = i
                break
        i = 0
        while i < 100:
            try:
                gameState.hasWall(i, 1)
                i += 1
            except:
                length = i
                break
        return height, length

    def getHomeEdges(self, gameState):
        # use to calculate the min distance to home
        edgeList = []
        height, length = self.getHightAndLength(gameState)

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


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    # usually, the offensive should not stop
    def stopAction(self):
        features = util.Counter()
        features['successorScore'] = -100000
        return features

    def initialFeature(self, foodLeft):
        features = util.Counter()
        features['successorScore'] = 100 - foodLeft  # self.getScore(successor)
        features['inCorner'] = 0
        features['distanceToGhost'] = 0
        features['distanceToHome'] = 0
        features['distanceToFood'] = 0
        return features

    # we return the possible location of the ghost
    def getGhostPotentialLocation(self, ghostPos, gameState):
        newGhostPos = []
        for pos in ghostPos:
            if not gameState.hasWall((int)(pos[0] + 1), (int)(pos[1])) and (pos[0] + 1, pos[1]) not in ghostPos:
                newGhostPos.append((pos[0] + 1, pos[1]))
            if not gameState.hasWall((int)(pos[0] - 1), (int)(pos[1])) and (pos[0] - 1, pos[1]) not in ghostPos:
                newGhostPos.append((pos[0] - 1, pos[1]))
            if not gameState.hasWall((int)(pos[0]), (int)(pos[1] + 1)) and (pos[0], pos[1] + 1) not in ghostPos:
                newGhostPos.append((pos[0], pos[1] + 1))
            if not gameState.hasWall((int)(pos[0]), (int)(pos[1] - 1)) and (pos[0], pos[1] - 1) not in ghostPos:
                newGhostPos.append((pos[0], pos[1] - 1))
        for pos in newGhostPos:
            ghostPos.append(pos)
        return ghostPos

    def getFeatures(self, gameState, action):

        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)

        ourFoodList = self.getFoodYouAreDefending(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice

        if action == 'Stop':
            return self.stopAction()
        # initial the score
        features = self.initialFeature(foodLeft)

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        # get the feature to the closest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        #Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        corners=self.removeCorners(gameState)
        if myPos in corners:
          features['inCorner']=1


        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        invadersPosition = [a.getPosition() for a in invaders]
        invadersPosition = self.getGhostPotentialLocation(invadersPosition, gameState)
        if myPos in invadersPosition:
            features['meetGhost'] = 1

        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = max([i.scaredTimer for i in enemies])
        if scaredTimes > 3:
            # when the oppsite is scared, we just eat food
            features['inCorner'] = 0
            features['distanceToGhost'] = 0
            features['distanceToHome'] = 0
        elif scaredTimes <= 2:
            # when the oppsite is not scared
            ours = [successor.getAgentState(i) for i in self.getTeam(successor)]
            our_invaders = [a for a in ours if a.getPosition() != None]
            if len(invaders) == 0:
                distanceToGhost = 100
            if len(invaders) > 0:
                # min distance to ghost
                distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                       for i in invaders])
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = 100 - distanceToCapsule
                except:
                    distanceToCapsule = 0



            if distanceToGhost < 6 + self.negetiveReward * 2:

                # If we found ghost, all the corners should be avoid
                corners = self.removeCornersBaseOnDistance(gameState,distanceToGhost)
                if myPos in corners:
                  features['inCorner'] = 1

            if distanceToGhost < 2 + self.negetiveReward:
                features['distanceToHome'] = 150 - minDistanceToHome
            if distanceToGhost < 4 + self.negetiveReward:
                features['distanceToGhost'] = distanceToGhost
        if self.leftMoves < minDistanceToHome + 3 or self.carryDots > 0 + self.backHomeTimes * 5 \
                or (minDistanceToHome < 3 and self.carryDots > 0):
            # should go home directly

            features['distanceToHome'] = 150 - minDistanceToHome
            features['distanceToFood'] = 0

        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'successorScore': 0, 'distanceToFood': 0,
                    'distanceToGhost': 100, 'distanceToHome': 60, 'inCorner': -100000,
                    'meetGhost': -1000000, 'reverse': -2, 'repeatPosition': 0, 'distanceToCapsule': 1}
        return {'successorScore': 100, 'distanceToFood': -2,
                'distanceToGhost': 100, 'distanceToHome': 60, 'inCorner': -100000,
                'meetGhost': -1000000, 'reverse': -2, 'distanceToCapsule': 1, 'repeatPosition': 0}

    def removeCornersBaseOnDistance(self, gameState, distanceToGhost):
        cornerList = []
        removeCornerList = []
        myPos = gameState.getAgentState(self.index).getPosition()
        height, length = self.getHightAndLength(gameState)
        loopTimes = 0
        if distanceToGhost >= 3:
            loopTimes = 1

        if distanceToGhost >= 5:
            loopTimes = 2
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
                        if gameState.hasWall(i + 1, j):
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j):
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1):
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1):
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in removeCornerList:
                            removeCornerList.append((i, j))
                    j += 1
                i += 1
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

        for i in removeCornerList:
            cornerList.remove(i)

        return cornerList
    def removeCorners(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        cornerList = []
        height, length = self.getHightAndLength(gameState)

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

    # This is the function that find the corners in layout
    def removeAllCorners(self, gameState):
        cornerList = []

        myPos = gameState.getAgentState(self.index).getPosition()
        height, length = self.getHightAndLength(gameState)

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


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    lastLostFoodPostion = (0, 0)
    lastLostFoodEffect = 10  # use to measure the effective of last lost food position

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        if len(invaders) == 0:
            distanceToCenter = self.getDistanceToCenter(gameState, myPos)
            features['distanceToCenter'] = distanceToCenter
        if action == Directions.STOP: features['stop'] = 1
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
                    self.lastLostFoodEffect -= 1

        features['distanceToLostFood'] = distanceToLostFood
        return features

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

    def getDistanceToCenter(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height, length = self.getHightAndLength(gameState)
        nearest = height
        for location in centerList:
            if abs(location[1] - (height + 1) / 2 < nearest):
                centerLocation = location
                nearest = abs(location[1] - (height + 1) / 2 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2
            , 'distanceToLostFood': -5, 'distanceToCenter': -3}
