from node_map import NODE_MAP
import random
import operator
from functools import reduce
import sys

sys.setrecursionlimit(768149)
# condition: eq neq and or
condition = [10, 15, 18, 19]


def number(s):
    return s['number']


def getchildNode(ast, childInFirst):
    second_1 = [0, 0, 0]
    if childInFirst == 0:
        second_action_ = [0, 0, 0]
    else:
        i = 1
        second_action_ = []
        second_ = []
        #childrenInSecond_ = []
        #childrenInSecond = []
        nodeInSecond = ast[int(childInFirst)]
        childrenInSecond_ = nodeInSecond['children']
        #childrenInSecond.append(childrenInSecond_)
        for childInSecond in childrenInSecond_:
            x = ast[int(childInSecond)]['name']
            second_.append(NODE_MAP[x] + 1)
        second_action_.append(0)
        for ind in range(ast[int(childInFirst)]['numOfChildren']):
            if second_[ind] in condition:
                second_1[0] = second_[ind]
                second_action_[0] = childrenInSecond_[ind]
            else:
                second_1[i] = second_[ind]
                second_action_.append(childrenInSecond_[ind])
                i = i + 1
        for index in range(3-len(second_action_)):
            second_action_.append(0)
    return second_1, second_action_


def getAstDict():
    # get dictionary of AST
    with open("./data/ast.txt", "r") as f:
        samples_before = f.read()
        samples = samples_before.split("\n")
        ast = []
        for sample in samples:
            if sample == '':
                print "error"
            nodeDict = {}
            nodeNumber = sample.split("node")[0]
            if sample == samples[0]:
                nodeDict['number'] = int(nodeNumber[-1])
            nodeDict['number'] = int(nodeNumber)
            node = sample.split("node:")[1].split(";")[0]
            nodeDict['name'] = node
            nodeDict['children'] = []
            children_ = sample.split("children:")[1]
            if children_[0] != 'n':         #null
                children_ = children_.split(';')
                for child in children_:
                    if child != '':
                        nodeDict['children'].append(child)
            else:
                nodeDict['children'] = []
            nodeDict['numOfChildren'] = len(nodeDict['children'])
            ast.append(nodeDict)
        ast_sorted = sorted(ast, key = number)
    return ast_sorted


def firstFloor(ast):
    nodeDict = ast[0]
    firstFloor = []
    action_layer1 = []
    for index in nodeDict['children']:
        nodeDict_ = ast[int(index)]
        x = nodeDict_['name']
        if x != 'wi' and x != 'cs':
            firstFloor.append(NODE_MAP[x] + 1)
            action_layer1.append(int(index))
    # for root wi
    # 7 = 5 + wi + cs
    for index in range(7 - nodeDict['numOfChildren']):
        firstFloor.append(0)
        action_layer1.append(0)
    return firstFloor, action_layer1


def secondFloor(ast, action_layer1):
    second_ = []
    childrenInFirst = action_layer1
    action_layer2 = []
    for childInFirst in childrenInFirst:
        # condition tr1 tr2
        second_1, second_action_ = getchildNode(ast, childInFirst)
        second_.append(second_1)
        action_layer2.append(second_action_)
    for index in range(5-len(second_)):
        second_ = [0, 0, 0]
        second_.append(second_)
    second = reduce(operator.add, second_)

    return second, action_layer2


def thirdFloor(ast, childrenInSecond):
    third = []
    action_layer3 = []
    for children in childrenInSecond:
            k = 0
            third_ = []
            action_layer3_1 = []
            for child in children:
                if k == 0:
                    third_cond = [0, 0]
                    third_cond_action = [0, 0]
                    if child != 0:
                        condThird = ast[int(child)]['children']
                        i = 0
                        for cond in condThird:
                            x = ast[int(cond)]['name']
                            third_cond[i] = NODE_MAP[x] + 1
                            third_cond_action[i] = cond
                            i = i + 1
                    third_.append(third_cond)
                    action_layer3_1.append(third_cond_action)
                else:
                    third_1, action_layer3_ = getchildNode(ast, child)
                    third_.append(third_1)
                    action_layer3_1.append(action_layer3_)
                k = k + 1
            for i in range(3 - len(third_)):
                third_temp = [0, 0, 0]
                third_.append(third_temp)
            third.append(third_)
            action_layer3.append(action_layer3_1)
    third = reduce(operator.add,reduce(operator.add, third))
    return third, action_layer3


def fourthFloor(ast,numThird):
    #reduce(operator.add, numThird)
    fourth = []
    action_layer4 = []
    for index in numThird:
        index = reduce(operator.add, index)
        for nodeNum in index:
            fourth_i = [0, 0]
            fourth_ = []
            action_layer4_ = [0, 0]
            if int(nodeNum) != 0:
                children = ast[int(nodeNum)]['children']
                k = 0
                for child in children:
                    action_layer4_[k] =child
                    x = ast[int(child)]['name']
                    fourth_.append(NODE_MAP[x] + 1)
                    k = k + 1
                i = 0
                for ind in range(len(fourth_)):
                    fourth_i[i] = fourth_[ind]
                    i = i + 1
            action_layer4.append(action_layer4_)
            fourth.append(fourth_i)
    fourth = reduce(operator.add,fourth)
    return fourth, action_layer4


def fifthFloor(ast, action_layer4):
    action_layer4 = reduce(operator.add, action_layer4)
    fifth = []
    for index in action_layer4:
        fifth_i = [0, 0]
        fifth_ = []
        if int(index) != 0:
            children = ast[int(index)]['children']
            for child in children:
                x = ast[int(child)]['name']
                fifth_.append(NODE_MAP[x] + 1)
            i = 0
            for ind in range(len(fifth_)):
                fifth_i[i] = fifth_[ind]
                i = i + 1
        fifth.append(fifth_i)
    fifth = reduce(operator.add, fifth)
    return fifth

def astEncoder(ast):
    first, action_layer1 = firstFloor(ast)
    second, action_layer2_ = secondFloor(ast, action_layer1)
    third, action_layer3_ = thirdFloor(ast, action_layer2_)
    # action_layer4 can not be choosed
    fourth, action_layer4 = fourthFloor(ast, action_layer3_)
    fifth = fifthFloor(ast, action_layer4)
    action_layer2 = []
    action_layer3 = []
    for action in action_layer2_:
        action = action[1:]
        action_layer2.append(action)
    for action_ in action_layer3_:
        action_ = action_[1:]
        for action in action_:
            action = action[1:]
            action_layer3.append(action)
    action_layer2 = reduce(operator.add, action_layer2)
    action_layer3 = reduce(operator.add, action_layer3)
    action_layer234 = action_layer1 + action_layer2 + action_layer3
    state = first + second + third + fourth + fifth
    return state, action_layer234

def setActSet():
    actSet = []
    for i in range(35):
        actSet.append(i)
    return actSet

def get_action1(action_layer234, ast, actionSet):
    # random.randint(a, b)
    # Return a random integer N such that a <= N <= b.
    treenode = ["assign", "if", "while"]
    actions = action_layer234
    # 0-29 30-768029 768030-768149
    if len(actionSet) == 0:
        print("error")
    node_chse = random.randint(0, (len(actionSet)-1))
    nodth = actionSet[node_chse]
    nodeNum = actions[node_chse]
    nodeNum_ = int(nodeNum)

    if nodeNum_ == 0 or ast[nodeNum_]['name'] not in treenode:
        #reward = -100
        actionSet.remove(nodth)
        nodeNum, nodth = get_action1(action_layer234, ast, actionSet)

    return nodeNum, nodth

def setAction1s(info_):
    treenode = ["assign", "if", "while"]
    actIndex = []
    for index in range(50):
        actIndex.append(0)
    for index in info_.astActNodes:
        a = int(index)
        if a != 0 and info_.ast[a]['name'] in treenode:
            actIndex[a] = 1
    # info_.actIndex = actIndex
    return actIndex


def getAction2(nodth):
    if nodth < 15 and nodth > 4:
        actionType = random.randint(0, 76799)
        layer = 3
    else:
        actionType = random.randint(0, 4)
        if nodth < 5:
            layer = 2
        else:
            layer = 4
    return actionType




'''
def get_action(action_layer234, ast, actionSet):
    # random.randint(a, b)
    # Return a random integer N such that a <= N <= b.
    treenode = ["assign", "if", "while"]
    actions = action_layer234
    # 0-29 30-768029 768030-768149
    #node_chse = random.randint(0, 35)
    #nodeNum = actions[node_chse]
    #while int(nodeNum) == 0 or ast[int(nodeNum)]['name'] not in treenode:
        #reward = -100
       # nodeNum = get_action(action_layer1, action_layer2, action_layer3, ast, actionSet)

    num = random.randint(0, (len(actionSet)-1))
    action = actionSet[num]
    actionSet.remove(action)

    if action < 30 :
        node_chse = action / 6
        action_type = action % 6
        layer = 2
    else:
        if action < 768030:
            node_chse = (action - 30) / 76800 + 5
            action_type = (action - 30) % 76800
            layer = 3
        else:
            node_chse = (action - 768030) / 6 + 5 + 10
            action_type = (action - 768030) % 6
            layer = 4
    nodeNum = actions[node_chse]
    print type(nodeNum)
    while int(nodeNum) == 0 or ast[int(nodeNum)]['name'] not in treenode:
        #reward = -100
        layer, action_type, nodeNum = get_action(action_layer234, ast, actionSet)
    return layer, action_type, nodeNum

'''



