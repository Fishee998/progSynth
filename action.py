commandtypeVarindex = [2, 3, 10, 20]
assignRandom = [0, 1, 2, 3, 10, 11, 12, 13, 14]

a = [3,4]
b = [1,2]
c = [10, 11, 12, 13, 14]
d = [0, 1, 2, 3]
condrandom = []
for a_ in a:
    strin = str(a_)
    for b_ in b:
        strin = strin + str(b_)
        for c_ in c:
            strin = strin + str(c_)
            for d_ in d:
                strin = strin + str(d_)
                for b_ in b:
                    strin = strin + str(b_)
                    print strin
                    for c_ in c:
                        strin = strin + str(c_)
                        for d_ in d:
                            strin = strin + str(d_)
                            condrandom.append(strin)
                            strin = strin[:-1]
                        strin = strin[:-2]
                    strin = strin[:-1]
                strin = strin[:-1]
            strin = strin[:-2]
        strin = strin[:-1]
    strin = strin[:-1]

'''
commandtypeVarindex2 = 2
commandtypeVarindex3 = 3
commandtypeVarindex10 = 10
commandtypeVarindex20 = 20

assignRandom0 = 0
assignRandom1 = 1
assignRandom2 = 2
assignRandom3 = 3
assignRandom10 = 10
assignRandom11 = 11
assignRandom12 = 12
assignRandom13 = 13
assignRandom14 = 14
'''

numofchNode_1 = 5
action1 = []
i = 0
for node_1 in range(numofchNode_1):
    for commandtypeVarindex_ in commandtypeVarindex:
        for assignRandom_ in assignRandom:
            action_ = {}
            action1.append(action_)
            action1[i]['commandtypeVarindex'] = commandtypeVarindex_
            action1[i]['assignRandom'] = assignRandom_
            i = i + 1

numofchNode_2 = 10
action2 = []
i = 0
for node_2 in range(numofchNode_1, numofchNode_1 + numofchNode_2):
    for commandtypeVarindex0 in commandtypeVarindex:
        for commandtypeVarindex1 in commandtypeVarindex:
            for assignRandom1 in assignRandom:
                for condrandom0 in condrandom:
                    for condrandom1 in condrandom:
                        action_ = {}
                        action2.append(action_)
                        action2[i]['commandtypeVarindex0'] = commandtypeVarindex0
                        action2[i]['commandtypeVarindex1'] = commandtypeVarindex1
                        action2[i]['assignRandom1'] = assignRandom1
                        action2[i]['condrandom0'] = condrandom0
                        action2[i]['condrandom1'] = condrandom1
                        i = i + 1






print 0

