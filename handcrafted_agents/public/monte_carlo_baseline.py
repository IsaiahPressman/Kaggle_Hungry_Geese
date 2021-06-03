from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
import random
import copy

frame = 0
opposites = {Action.EAST: Action.WEST, Action.WEST: Action.EAST, Action.NORTH: Action.SOUTH, Action.SOUTH: Action.NORTH}
action_meanings = {Action.EAST: (1, 0), Action.WEST: (-1, 0), Action.NORTH: (0, -1), Action.SOUTH: (0, 1)}
action_names = {(1, 0): Action.EAST, (-10, 0): Action.EAST, (-1, 0): Action.WEST, (10, 0): Action.WEST,
                (0, -1): Action.NORTH, (0, 6): Action.NORTH, (0, -6): Action.SOUTH, (0, 1): Action.SOUTH}
strValue = {Action.EAST: 'EAST', Action.WEST: 'WEST', Action.NORTH: 'NORTH', Action.SOUTH: 'SOUTH'}
all_last_actions = [None, None, None, None]
revert_last_actions = [None, None, None, None]
last_observation = None


class Obs:
    pass


def setLastActions(observation, configuration):
    global frame, revert_last_actions, all_last_actions
    if not frame == 0:
        for i in range(4):
            setLastAction(observation, configuration, i)
    revert_last_actions = copy.deepcopy(all_last_actions)


def revertLastActions():
    global revert_last_actions, all_last_actions
    all_last_actions = copy.deepcopy(revert_last_actions)


def setLastAction(observation, configuration, gooseIndex):
    global last_observation, all_last_actions, action_names
    if len(observation.geese[gooseIndex]) > 0:
        oldGooseRow, oldGooseCol = row_col(last_observation.geese[gooseIndex][0], configuration.columns)
        newGooseRow, newGooseCol = row_col(observation.geese[gooseIndex][0], configuration.columns)
        all_last_actions[gooseIndex] = action_names[
            ((newGooseCol - oldGooseCol) % configuration.columns, (newGooseRow - oldGooseRow) % configuration.rows)]


def getValidDirections(observation, configuration, gooseIndex):
    global all_last_actions, opposites
    directions = [Action.EAST, Action.WEST, Action.NORTH, Action.SOUTH]
    returnDirections = []
    for direction in directions:
        row, col = getRowColForAction(observation, configuration, gooseIndex, direction)
        if not willGooseBeThere(observation, configuration, row, col) and not all_last_actions[gooseIndex] == opposites[
            direction]:
            returnDirections.append(direction)
    if len(returnDirections) == 0:
        return directions
    return returnDirections


def randomTurn(observation, configuration, actionOverrides, rewards, fr):
    newObservation = cloneObservation(observation)
    for i in range(4):
        if len(observation.geese[i]) > 0:
            if i in actionOverrides.keys():
                newObservation = performActionForGoose(observation, configuration, i, newObservation,
                                                       actionOverrides[i])
            else:
                newObservation = randomActionForGoose(observation, configuration, i, newObservation)

    checkForCollisions(newObservation, configuration)
    updateRewards(newObservation, configuration, rewards, fr)
    hunger(newObservation, fr)
    return newObservation


def hunger(observation, fr):
    if fr % 40 == 0:
        for g, goose in enumerate(observation.geese):
            goose = goose[0:len(goose) - 1]


def updateRewards(observation, configuration, rewards, fr):
    for g, goose in enumerate(observation.geese):
        if len(goose) > 0:
            rewards[g] = 2 * fr + len(goose)


def checkForCollisions(observation, configuration):
    killed = []
    for g, goose in enumerate(observation.geese):
        if len(goose) > 0:
            for o, otherGoose in enumerate(observation.geese):
                for p, part in enumerate(otherGoose):
                    if not (o == g and p == 0):
                        if goose[0] == part:
                            killed.append(g)

    for kill in killed:
        observation.geese[kill] = []


def cloneObservation(observation):
    newObservation = Obs()
    newObservation.index = observation.index
    newObservation.geese = copy.deepcopy(observation.geese)
    newObservation.food = copy.deepcopy(observation.food)
    return newObservation


def randomActionForGoose(observation, configuration, gooseIndex, newObservation):
    validActions = getValidDirections(observation, configuration, gooseIndex)
    action = random.choice(validActions)
    row, col = getRowColForAction(observation, configuration, gooseIndex, action)
    newObservation.geese[gooseIndex] = [row * configuration.columns + col] + newObservation.geese[gooseIndex]
    if not isFoodThere(observation, configuration, row, col):
        newObservation.geese[gooseIndex] = newObservation.geese[gooseIndex][0:len(newObservation.geese[gooseIndex]) - 1]
    return newObservation


def performActionForGoose(observation, configuration, gooseIndex, newObservation, action):
    row, col = getRowColForAction(observation, configuration, gooseIndex, action)
    newObservation.geese[gooseIndex][:0] = [row * configuration.columns + col]
    if not isFoodThere(observation, configuration, row, col):
        newObservation.geese[gooseIndex] = newObservation.geese[gooseIndex][0:len(newObservation.geese[gooseIndex]) - 1]
    return newObservation


def isFoodThere(observation, configuration, row, col):
    for food in observation.food:
        foodRow, foodCol = row_col(food, configuration.columns)
        if foodRow == row and foodCol == col:
            return True
    return False


def willGooseBeThere(observation, configuration, row, col):
    for goose in observation.geese:
        for p, part in enumerate(goose):
            if not p == len(goose) - 1:
                partRow, partCol = row_col(part, configuration.columns)
                if partRow == row and partCol == col:
                    return True
    return False


def getRowColForAction(observation, configuration, gooseIndex, action):
    global action_meanings
    gooseRow, gooseCol = row_col(observation.geese[gooseIndex][0], configuration.columns)
    actionRow = (gooseRow + action_meanings[action][1]) % configuration.rows
    actionCol = (gooseCol + action_meanings[action][0]) % configuration.columns
    return actionRow, actionCol


def simulateMatch(observation, configuration, firstMove, depth):
    global frame
    actionOverrides = {observation.index: firstMove}
    revertLastActions()
    simulationFrame = frame + 1
    newObservation = cloneObservation(observation)
    rewards = [0, 0, 0, 0]
    count = 0
    while count < depth:
        newObservation = randomTurn(newObservation, configuration, actionOverrides, rewards, simulationFrame)
        actionOverrides = {}
        simulationFrame += 1
        count += 1
    return rewards


def simulateMatches(observation, configuration, numMatches, depth):
    options = getValidDirections(observation, configuration, observation.index)
    rewardTotals = []
    for o, option in enumerate(options):
        rewardsForOption = [0, 0, 0, 0]
        for i in range(numMatches):
            matchRewards = simulateMatch(observation, configuration, option, depth)
            for j in range(4):
                rewardsForOption[j] += matchRewards[j]
        rewardTotals.append(rewardsForOption)
    scores = []
    for o, option in enumerate(options):
        rewards = rewardTotals[o]
        if len(rewards) <= 0:
            mean = 0
        else:
            mean = sum(rewards) / len(rewards)
        if mean == 0:
            scores.append(0)
        else:
            scores.append(rewards[observation.index] / mean)

    """
    print('frame: ', frame)
    print('options: ', options)
    print('scores: ', scores)
    print('reward totals: ', rewardTotals)
    print('lengths: ')
    print('0: ', len(observation.geese[0]))
    print('1: ', len(observation.geese[1]))
    print('2: ', len(observation.geese[2]))
    print('3: ', len(observation.geese[3]))
    """

    return options[scores.index(max(scores))]


def agent(obs_dict, config_dict):
    global last_observation, all_last_actions, opposites, frame
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    setLastActions(observation, configuration)
    myLength = len(observation.geese[observation.index])
    if myLength < 5:
        my_action = simulateMatches(observation, configuration, 300, 3)
    elif myLength < 9:
        my_action = simulateMatches(observation, configuration, 120, 6)
    else:
        my_action = simulateMatches(observation, configuration, 85, 9)

    last_observation = cloneObservation(observation)
    frame += 1
    return strValue[my_action]
