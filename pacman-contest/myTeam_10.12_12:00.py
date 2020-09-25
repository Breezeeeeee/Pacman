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
    AisRepeat =False
    BisRepeat = False
    reward = 0
    negetiveReward = 0
    startPostion = (0, 0)

    escapeFlag = 0
    lastLostFoodPostion = (0, 0)
    lastLostFoodEffect = 10  # use to measure the effective of last lost food position

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)



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

    def getDistanceToTop(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height, length = self.getHightAndLength(gameState)
        nearest = height
        for location in centerList:
            if abs(location[1] - 3) < nearest:
                centerLocation = location
                nearest = abs(location[1] - 3) < nearest
        return self.getMazeDistance(myPos, centerLocation)
    def getFeaturesGoHome(self, gameState, action):
        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)
        ourFoodLeft=self.getFoodYouAreDefending(successor).asList()
        ourFoodList = self.getFoodYouAreDefending(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice


        # initial the score
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['inCorner'] = 0
        features['distanceToGhost'] = 0
        features['distanceToHome'] = 0
        features['distanceToFood'] = 0
        features['finalDistanceToHome'] = 100 - minDistanceToHome
        features['leftCapsules'] = 100 - len(self.getCapsules(successor))
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # get the feature to the closest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry

            features['distanceToFood'] = self.distributeDots(successor)

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        corners = self.removeCorners(gameState)
        if myPos in corners:
            features['inCorner'] = 1

        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = min([i.scaredTimer for i in enemies])
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

                distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                       for i in invaders])
                x = min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                         for i in invaders])
                if action == 'Stop' and x != 2:
                    return self.stopAction()
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = distanceToCapsule

                except:
                    distanceToCapsule = -1

            if distanceToGhost < 13 :
                # If we found ghost, all the corners should be avoid
                corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)

                if myPos in corners:
                    features['inCorner'] = 1

            if distanceToGhost < 4:
                features['distanceToHome'] = 100 - minDistanceToHome
            if distanceToGhost < 4:
                features['distanceToGhost'] = 100 - distanceToGhost
                features['successorScore'] = 0



        elif self.leftMoves<minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] =-(50- distanceToCapsule)*80
            except:
                distanceToCapsule=-1
        if (not gameState.getAgentState(self.index).isPacman):
            features['distanceToGhost'] = 0
        features['distanceToHome'] = 100 - minDistanceToHome
        features['distanceToFood'] = 0
        return features
    def getFeaturesAsDefensive(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if gameState.getAgentState(self.index).isPacman:

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

            features['finalDistanceToHome'] = 100 - minDistanceToHome
            features['leftCapsules'] = 100 - len(self.getCapsules(successor))
            rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]


            # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
            corners = self.removeCorners(gameState)
            if myPos in corners:
                features['inCorner'] = 1

            # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            # 获取吃超级豆子之后 Ghost害怕还剩余的时间
            scaredTimes = min([i.scaredTimer for i in enemies])
            if scaredTimes > 3:
                # when the oppsite is scared, we just eat food
                features['inCorner'] = 0
                features['distanceToGhost'] = 0

            elif scaredTimes <= 2:
                # when the oppsite is not scared
                ours = [successor.getAgentState(i) for i in self.getTeam(successor)]
                our_invaders = [a for a in ours if a.getPosition() != None]
                if len(invaders) == 0:
                    distanceToGhost = 100
                if len(invaders) > 0:
                    distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                           for i in invaders])
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

                if distanceToGhost < 4:
                    features['distanceToHome'] = 100 - minDistanceToHome
                if distanceToGhost < 4:
                    features['distanceToGhost'] = 100 - distanceToGhost
                    features['successorScore'] = 0



            features['distanceToHome'] = 100 - minDistanceToHome
            features['distanceToFood'] = 0
            if self.leftMoves < minDistanceToHome:
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = -(50 - distanceToCapsule) * 80
                except:
                    distanceToCapsule = -1
            if (not gameState.getAgentState(self.index).isPacman):
                features['distanceToGhost'] = 0
            print(features)
            return features
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = 200-min(dists)
        if len(invaders) == 0:
            distanceToTop = self.getDistanceToCenter(gameState, myPos)
            features['distanceToCenter'] = distanceToTop

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
        print(features)
        return features

    def changeEntryPoint(self, centerList, gameState, myPos):
        height, length = self.getHightAndLength(gameState)
        nearest = height
        for location in centerList:
            if abs(location[1] - 3) < nearest:
                centerLocation = location
                nearest = abs(location[1] - 3) < nearest
        return self.getMazeDistance(myPos, centerLocation)

    changeEntryEffect = 0

    def getFeaturesLowerScoreRepeat(self, gameState, action):

        # get basic parameters
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)
        features['changeEntryPoint'] = 0

        ourFoodList = self.getFoodYouAreDefending(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice
        if self.repeatActionDetect() or self.changeEntryEffect > 0:
            # features['changeEntryPoint']=self.changeEntryPoint(edgeList,gameState,myPos)
            self.changeEntryEffect = 10
        if action == 'Stop':

            return self.stopAction()
            # initial the score
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['inCorner'] = 0
        features['distanceToGhost'] = 0
        features['distanceToHome'] = 0
        features['distanceToFood'] = 0
        features['finalDistanceToHome'] = 100 - minDistanceToHome
        features['leftCapsules'] = 100 - len(self.getCapsules(successor))
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # get the feature to the closest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry

            features['distanceToFood'] = self.distributeDots(successor)

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        corners = self.removeCorners(gameState)
        if myPos in corners:
            features['inCorner'] = 1

        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = min([i.scaredTimer for i in enemies])
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

                distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                       for i in invaders])
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

            if distanceToGhost < 4:
                features['distanceToHome'] = 100 - minDistanceToHome
            if distanceToGhost < 4:
                features['distanceToGhost'] = 100 - distanceToGhost
                features['successorScore'] = 0

        if self.leftMoves < minDistanceToHome + 3 and self.carryDots > 0 or self.carryDots > 0 + self.backHomeTimes * abs(
                self.getScore(gameState)):
            # should go home directly

            features['distanceToHome'] = 100 - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves < minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] = -(50 - distanceToCapsule) * 80
            except:
                distanceToCapsule = -1
        if (not gameState.getAgentState(self.index).isPacman):
            features['distanceToGhost'] = 0

        return features

    # def getFeaturesHigherScoreRepeat:
    # def getFeaturesSameScoreRepeat:
    def removeCornersBaseOnDistance(self, gameState, distanceToGhost):
        cornerList = []
        removeCornerList = []
        capsuleList=self.getCapsules(gameState)
        myPos = gameState.getAgentState(self.index).getPosition()
        height, length = self.getHightAndLength(gameState)
        loopTimes = 0
        if distanceToGhost > 3:
            loopTimes = 1
        if distanceToGhost > 6:
            loopTimes = 2
        if distanceToGhost >8:
            loopTimes=3
        if distanceToGhost >10:
            loopTimes=4
        if distanceToGhost >12:
            loopTimes=5
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
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in removeCornerList:
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
                        if numberOfWalls >= 3 and (i, j) not in cornerList and (i,j) not in capsuleList:
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


    def stopAction(self):
        features = util.Counter()
        features['stop'] = 100000
        return features

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    def recordInformation(self,gameState):
        self.leftMoves -= 1

        myPos = gameState.getAgentState(self.index).getPosition()
        if self.startPostion == myPos:
            self.negetiveReward += 1

        if self.leftMoves == 300:
            self.startPostion = myPos

        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.repeatFlag=0
            self.carryDots = 0
            self.backHomeTimes += 1
            self.reward += 1

    def recordInformationAfterCurrentStep(self,gameState,finalAction):
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.reward += 1
            self.repeatFlag=0
            self.BisRepeat = False
        if self.escapeFlag > 0:
            self.escapeFlag -= 1

        if self.changeEntryEffect > 0:
            self.changeEntryEffect -= 1
        self.historyAction.append(finalAction)
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.recordInformation(gameState)
        print(self.AisRepeat,self.BisRepeat,self.repeatFlag)
        actions = gameState.getLegalActions(self.index)

        self.solveRepeat(gameState)
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        finalAction = bestActions[0]
        self.recordInformationAfterCurrentStep(gameState,finalAction)
        return finalAction
    def repeatThenChooseAnotherFood(self,gameState):

        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        team = self.getTeam(gameState)
        meTeamMate = 0
        height, length = self.getHightAndLength(gameState)

        for i in team:
            if i != self.index:
                myTeamMatePos = gameState.getAgentState(i).getPosition()
                myTeamMate = i
        upAreaFood = []
        downAreaFood = []
        # 分为上下两个领域
        for food in foodList:
            if food[1] >= height / 2:
                upAreaFood.append(food)
            else:
                downAreaFood.append(food)
        minDisList=[]
        firstMinDis = self.distributeDots(gameState)
        i = 0
        while i<self.repeatFlag:
            i+=1

            minDis = 100
            minDisFood = ()
            if len(upAreaFood) > 0 and len(downAreaFood) > 0:
                for food in upAreaFood:
                    if self.getMazeDistance(myPos, food) < minDis and self.getMazeDistance(myPos, food) != firstMinDis and food not in minDisList:
                        minDis = self.getMazeDistance(myPos, food)
                        misDisFood = food

            elif len(upAreaFood) == 0:
                for food in downAreaFood:
                    if self.getMazeDistance(myPos, food) < minDis and self.getMazeDistance(myPos, food) != firstMinDis and food not in minDisList:
                        minDis = self.getMazeDistance(myPos, food)
                        misDisFood = food

            elif len(downAreaFood) == 0:
                for food in upAreaFood:
                    if self.getMazeDistance(myPos, food) < minDis and self.getMazeDistance(myPos, food) != firstMinDis and food not in minDisList:
                        minDis = self.getMazeDistance(myPos, food)
                        misDisFood = food

            minDisList.append(minDisFood)

        return minDis

    def distributeDots(self,gameState):
        foodList=self.getFood(gameState).asList()
        myPos=gameState.getAgentState(self.index).getPosition()
        team=self.getTeam(gameState)
        meTeamMate=0
        height, length=self.getHightAndLength(gameState)

        for i in team:
            if i != self.index:
                myTeamMatePos=gameState.getAgentState(i).getPosition()
                myTeamMate=i
        upAreaFood=[]
        downAreaFood=[]
        #分为上下两个领域
        for food in foodList:
            if food[1]>=height/2:
                upAreaFood.append(food)
            else:
                downAreaFood.append(food)
        minDis = 100
        minDisFood = ()
        if len(upAreaFood)>0 and len(downAreaFood)>0:


            for food in upAreaFood:
                if self.getMazeDistance(myPos, food) < minDis:
                    minDis = self.getMazeDistance(myPos, food)
                    misDisFood = food
            return minDis
        elif len(upAreaFood)==0:
            for food in downAreaFood:
                if self.getMazeDistance(myPos, food) < minDis:
                    minDis = self.getMazeDistance(myPos, food)
                    misDisFood = food
            return minDis
        elif len(downAreaFood)==0:
            for food in upAreaFood:
                if self.getMazeDistance(myPos, food) < minDis:
                    minDis = self.getMazeDistance(myPos, food)
                    misDisFood = food
            return minDis

    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    repeatFlag = 0
    isOffensive = True
    historyAction = []
    goOffensive = True

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

    def solveRepeat(self, gameState):
        print(self.getScore(gameState))
        myPos = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(invaders) == 0:
            distanceToGhost = 100
        if len(invaders) > 0:
            distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                   for i in invaders])
        if distanceToGhost > 10:
            self.isOffensive = True
        if self.getScore(gameState) < 0:
            self.isOffensive = True

        if len(self.getFood(gameState).asList()) <= 2 and (not gameState.getAgentState(self.index).isPacman):
            self.isOffensive = False

        if self.repeatActionDetect():
            self.repeatFlag +=1
            self.AisRepeat=True
            if len(self.getFood(gameState).asList()) < len(
                    self.getFoodYouAreDefending(gameState).asList()) or self.getScore(gameState) > 0:
                self.isOffensive = False
    def distanceToHighEntry(self,gameState,myPos):
        centerList = self.getHomeEdges(gameState)
        height, length = self.getHightAndLength(gameState)
        nearest = height
        for location in centerList:
            if abs(location[1] - 3*(height + 1) / 4< nearest):
                centerLocation = location
                nearest = abs(location[1] - 3*(height + 1) / 4 < nearest)
        return self.getMazeDistance(myPos, centerLocation)


    def getFeatures(self, gameState, action):

        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        ourFoodLeft=self.getFoodYouAreDefending(gameState).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)

        ourFoodList = self.getFoodYouAreDefending(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice

        if not self.isOffensive:
            print("this is desive")
            return self.getFeaturesAsDefensive(gameState,action)
                #return self.getRiskAttackFeature(gameState,action)
                #return self. getEscapeGhostFeature()
        if self.repeatFlag>0 and self.carryDots>0:
            self.getFeaturesGoHome(gameState)
        # initial the score
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['inCorner'] = 0
        features['distanceToGhost'] = 0
        features['distanceToHome'] = 0
        features['distanceToFood'] = 0
        features['finalDistanceToHome'] = 100-minDistanceToHome
        features['leftCapsules'] = 100 - len(self.getCapsules(successor))
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # get the feature to the closest food
        if len(foodList) > 0 and self.repeatFlag==0 :  # This should always be True,  but better safe than sorry
            features['distanceToFood'] = self.distributeDots(successor)
        else:
            if (int)(self.repeatFlag/10)<len(foodList):
                features['distanceToFood'] = self.distancer.getDistance(foodList[(int)(self.repeatFlag/10)], myPos)
            else:
                features['distanceToFood'] = 0
                features['distanceToHome'] = 100- minDistanceToHome

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        corners = self.removeCorners(gameState)
        if myPos in corners:
            features['inCorner'] = 1

        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = min([i.scaredTimer for i in enemies])
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
                if self.AisRepeat and self.BisRepeat:
                    features['distanceToTeammate']=self.distancer.getDistance(ours[0].getPosition,ours[1].getPosition())
                distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                           for i in invaders])
                x =min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                                           for i in invaders])
                if action == 'Stop' and x!=2:
                    return self.stopAction()

                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = distanceToCapsule
                except:
                    distanceToCapsule = -1
            if distanceToGhost < 15 :
                # If we found ghost, all the corners should be avoid
                corners = self.removeCornersBaseOnDistance(gameState,distanceToGhost)

                if myPos in corners:
                    features['inCorner'] = 1

            if distanceToGhost < 5 :
                features['distanceToHome'] =100- minDistanceToHome
            if distanceToGhost < 5 :
                features['distanceToGhost'] = 100 - distanceToGhost
                features['successorScore']=0

        if self.leftMoves < minDistanceToHome + 3 and self.carryDots>0:
            # should go home directly

            features['distanceToHome'] = 100 - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves<minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] =-(50- distanceToCapsule)*80
            except:
                distanceToCapsule=-1
        if (not gameState.getAgentState(self.index).isPacman):
            features['distanceToGhost']=0

        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'successorScore': 0,'leftCapsules':100, 'distanceToFood': 0,
                    'distanceToGhost': -100, 'distanceToTeammate':20, 'distanceToHome': 0, 'inCorner': -100000, 'finalDistanceToHome': 5,
                    'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'stop': -100, 'reverse': -2
            , 'distanceToLostFood': -5, 'distanceToCenter': -3,'distanceToCapsule': -1}
        return {'successorScore': 100, 'leftCapsules':100,'distanceToFood': -2,
                'distanceToGhost': -100, 'distanceToTeammate':20,'distanceToHome': 60, 'inCorner': -100000, 'finalDistanceToHome': 0,
                'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'stop': -100, 'reverse': -2
            , 'distanceToLostFood': -5, 'distanceToCenter': -3, 'distanceToEntry':1,'distanceToCapsule': -1,'changeEntryPoint':1000}




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    goOffensive=False

    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    repeatFlag = False
    isOffensive = True
    historyAction = []
    def recordInformation(self,gameState):
        self.leftMoves -= 1

        myPos = gameState.getAgentState(self.index).getPosition()
        if self.startPostion == myPos:
            self.negetiveReward += 1

        if self.leftMoves == 300:
            self.startPostion = myPos

        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.carryDots = 0
            self.repeatFlag=False
            self.backHomeTimes += 1
            self.reward += 1

    def recordInformationAfterCurrentStep(self,gameState,finalAction):
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.reward += 1
            self.repeatFlag=False
            self.BisRepeat=False

        if self.escapeFlag > 0:
            self.escapeFlag -= 1

        if self.changeEntryEffect > 0:
            self.changeEntryEffect -= 1
        self.historyAction.append(finalAction)
    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.recordInformation(gameState)

        actions = gameState.getLegalActions(self.index)

        self.solveRepeat(gameState)
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        finalAction = bestActions[0]
        self.recordInformationAfterCurrentStep(gameState,finalAction)
        print(self.index, self.leftMoves,finalAction,values)
        return finalAction

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

    def solveRepeat(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(invaders) == 0:
            distanceToGhost = 100
        if len(invaders) > 0:
            distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                   for i in invaders])
        if distanceToGhost > 10:
            self.isOffensive = True
        if self.getScore(gameState) < 0:
            self.isOffensive = True
        if len(self.getFood(gameState).asList()) <= 2 and (not gameState.getAgentState(self.index).isPacman):
            self.isOffensive = False

        if self.repeatActionDetect():
            self.repeatFlag = True
            self.BisRepeat=True
            if len(self.getFood(gameState).asList()) < len(
                    self.getFoodYouAreDefending(gameState).asList()) or self.getScore(gameState) > 0:
                self.isOffensive = False
    def distributeDots(self, gameState):
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        team = self.getTeam(gameState)
        meTeamMate = 0
        height, length = self.getHightAndLength(gameState)

        for i in team:
            if i != self.index:
                myTeamMatePos = gameState.getAgentState(i).getPosition()
                myTeamMate = i
        upAreaFood = []
        downAreaFood = []
        # 分为上下两个领域
        for food in foodList:
            if food[1] >= height / 2:
                upAreaFood.append(food)
            else:
                downAreaFood.append(food)
        minDis = 1000
        minDisFood = ()
        teamMateMinDis = 1000
        teamMateMinDisFood = ()
        if len(upAreaFood) > 0 and len(downAreaFood) > 0:


            for food in downAreaFood:
                if self.getMazeDistance(myPos, food) < minDis:
                    minDis = self.getMazeDistance(myPos, food)
                    misDisFood = food
            return minDis
        elif len(upAreaFood) == 0:

            for food in downAreaFood:
                if self.getMazeDistance(myTeamMatePos, food) < minDis:
                    minDis = self.getMazeDistance(myTeamMatePos, food)
                    misDisFood = food

            for food in downAreaFood:
                if self.getMazeDistance(myPos, food) < teamMateMinDis and food != misDisFood:
                    teamMateMinDis = self.getMazeDistance(myPos, food)
                    teamMateMinDisFood = food
            return teamMateMinDis
        elif len(downAreaFood) == 0:
            for food in upAreaFood:
                if self.getMazeDistance(myTeamMatePos, food) < minDis:
                    minDis = self.getMazeDistance(myTeamMatePos, food)
                    misDisFood = food

            for food in upAreaFood:
                if self.getMazeDistance(myPos, food) < teamMateMinDis and food != misDisFood:
                    teamMateMinDis = self.getMazeDistance(myPos, food)
                    teamMateMinDisFood = food
            return teamMateMinDis
    def distanceToLowEntry(self,gameState,myPos):
        centerList = self.getHomeEdges(gameState)
        height, length = self.getHightAndLength(gameState)
        nearest = height
        for location in centerList:
            if abs(location[1] - 1*(height + 1) / 4< nearest):
                centerLocation = location
                nearest = abs(location[1] - 1*(height + 1) / 4 < nearest)
        return self.getMazeDistance(myPos, centerLocation)




    def getFeatures(self, gameState, action):
        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        foodLeft = len(foodList)
        edgeList = self.getHomeEdges(gameState)
        ourFoodLeft=self.getFoodYouAreDefending(successor).asList()
        ourFoodList = self.getFoodYouAreDefending(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in edgeList])
        # Stop is meaningless, therefore, it should be bad choice
        if not self.isOffensive:
            return self.getFeaturesAsDefensive(gameState,action)
        if self.repeatFlag and self.carryDots>0:
            return self.getFeaturesGoHome(gameState,action)
        if self.repeatFlag:
            return self.getFeaturesLowerScoreRepeat(gameState,action)
                #return self.getRiskAttackFeature(gameState,action)
                #return self. getEscapeGhostFeature()

        # initial the score
        features['successorScore'] = -len(foodList)  # self.getScore(successor)
        features['inCorner'] = 0
        features['distanceToGhost'] = 0
        features['distanceToHome'] = 0
        features['distanceToFood'] = 0
        features['finalDistanceToHome'] = 100 - minDistanceToHome
        features['leftCapsules'] = 100 - len(self.getCapsules(successor))
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # get the feature to the closest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry

            features['distanceToFood'] = self.distributeDots(successor)

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        corners = self.removeCorners(gameState)
        if myPos in corners:
            features['inCorner'] = 1

        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = min([i.scaredTimer for i in enemies])
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

                distanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                       for i in invaders])
                x = min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                         for i in invaders])
                if action == 'Stop' and x != 2:
                    return self.stopAction()
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                    features['distanceToCapsule'] = distanceToCapsule

                except:
                    distanceToCapsule = -1

            if distanceToGhost < 13 :
                # If we found ghost, all the corners should be avoid
                corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)

                if myPos in corners:
                    features['inCorner'] = 1

            if distanceToGhost < 4:
                features['distanceToHome'] = 100 - minDistanceToHome
            if distanceToGhost < 4:
                features['distanceToGhost'] = 100 - distanceToGhost
                features['successorScore'] = 0

        if self.leftMoves < minDistanceToHome + 3 and self.carryDots>0:
            # should go home directly

            features['distanceToHome'] = 100 - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves<minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] =-(50- distanceToCapsule)*80
            except:
                distanceToCapsule=-1
        if (not gameState.getAgentState(self.index).isPacman):
            features['distanceToGhost'] = 0

        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'successorScore': 0,'leftCapsules':100, 'distanceToFood': 0,
                    'distanceToGhost': -100, 'distanceToHome': 0, 'inCorner': -100000, 'finalDistanceToHome': 5,
                    'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'stop': -100, 'reverse': -2
                , 'distanceToLostFood': -5, 'distanceToCenter': -3,'distanceToCapsule': -1}
        return {'successorScore': 100,'leftCapsules':100, 'distanceToFood': -2,
                'distanceToGhost': -100, 'distanceToHome': 60, 'inCorner': -100000, 'finalDistanceToHome': 0,
                'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'stop': -100, 'reverse': -2
            , 'distanceToLostFood': -5, 'distanceToCenter': -3, 'distanceToEntry': 1, 'distanceToCapsule': -1,
                'changeEntryPoint': 1000}
