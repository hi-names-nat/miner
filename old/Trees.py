import matplotlib.pyplot as plt
import pandas
import numpy as np
import IPython.display as display
# Global variables to drive stuff - not good practice but this is my code dangit
DEBUG: int = 1
RANDOMSEED: int = 10024


# Node tree class, used for the BTree
class BTreeNode:
    def __init__(self):
        self.right: BTreeNode = None
        self.left: BTreeNode = None
        self.data = None
        self.pivot: int = None
        self.moreOf: int = None

    def GenerateConfusionMatrix(self, isLeft):
        # do the thing
        confusion = dict()

        t = self.__FindPosNeg(0)
        print("final confusion tuple: " + str(t))
        confusion["Actual Positive"] = [t[0], t[3]]
        confusion["Actual Negative"] = [t[1], t[2]]

        confusionmatrix = pandas.DataFrame.from_dict(confusion, "index")
        confusionmatrix.columns = ["Predicted Positive", "Predicted Negative"]

        return confusionmatrix

    # walk the list and get all values
    def __FindPosNeg(self, isRight):  # return a tuple [TP, FP, TN, FN]
        returnTuple = [0, 0, 0, 0]
        print("\nFindPosNeg\n")
        if self.left is None and self.right is None:  # get the actual values
            x = self.__GetConfusionForNode(isRight)
            print("Confusion for node: isRight: " + str(isRight) + " return: " + str(x))
            return x
        else:
            l = self.left.__FindPosNeg(0)
            r = self.right.__FindPosNeg(1)
            print("left: " + str(l) + " right: " + str(r) + "\n")
            returnTuple[0] = returnTuple[0] + l[0] + r[0]
            returnTuple[1] = returnTuple[1] + l[1] + r[1]
            returnTuple[2] = returnTuple[2] + l[2] + r[2]
            returnTuple[3] = returnTuple[3] + l[3] + r[3]
            print("final tuple calculation for this block: " + str(returnTuple))
            return returnTuple

    def __GetConfusionForNode(self, isRight):
        mReadableData = self.data

        t = isRight

        countTruePos = 0
        countTrueNeg = 0
        countFalsePos = 0
        countFalseNeg = 0
        for i in range(0, len(mReadableData.index), 1):
            c = mReadableData.iloc[i, 0]

            print("iteration " + str(i) + " isright: " + str(isRight) + " true value: " + str(c))

            if c == 1:
                if c == t:
                    countTruePos = countTruePos + 1
                else:
                    countFalseNeg = countFalseNeg + 1
            else:
                if c != t:
                    countFalsePos = countFalsePos + 1
                else:
                    countTrueNeg = countTrueNeg + 1
        return [countTruePos, countFalsePos, countTrueNeg, countFalseNeg]

    def MakeConfusionWithTree(self, mReadableData):  # return a tuple [TP, FP, TN, FN]
        returnTuple = [0, 0, 0, 0]

        t = -1
        for i in range(0, len(mReadableData.index), 1):
            c = mReadableData.iloc[i, 0]
            root = self
            while root.left is not None and root.right is not None:

                pivot = mReadableData.iloc[i, root.pivot]

                if pivot == 1:
                    root = root.right
                    t = 1
                else:
                    root = root.left
                    t = 0

            if c == 1:
                if c == t:
                    returnTuple[0] = returnTuple[0] + 1
                else:
                    returnTuple[3] = returnTuple[3] + 1
            else:
                if c != t:
                    returnTuple[1] = returnTuple[1] + 1
                else:
                    returnTuple[2] = returnTuple[2] + 1

        return returnTuple

    def __GetMoreOf(self):  # 1: positive, 0: negative
        p = 0;
        np = 0
        for i in range(0, len(mReadableData.index), 1):
            c = mReadableData.iloc[i, 0]

            if (c == 1):
                p = p + 1
            else:
                np = np + 1

        if p >= np:
            return 1
        else:
            return 0

    def PopulateMoreRoots(self):
        if self.left is not None and self.right is not None:
            self.left.PopulateMoreRoots()
            self.right.PopulateMoreRoots()
        else:
            self.moreOf = self.__GetMoreOf()

    def MakePhiConfusion(self, mReadableData):
        returnTuple = [0, 0, 0, 0]

        t = -1
        for i in range(0, len(mReadableData.index), 1):
            c = mReadableData.iloc[i, 0]
            root = self
            while root.left is not None and root.right is not None:

                pivot = mReadableData.iloc[i, root.pivot]

                if pivot == 1:
                    root = root.right
                else:
                    root = root.left

            t = self.moreOf
            if c == 1:
                if c == t:
                    returnTuple[0] = returnTuple[0] + 1
                else:
                    returnTuple[3] = returnTuple[3] + 1
            else:
                if c != t:
                    returnTuple[1] = returnTuple[1] + 1
                else:
                    returnTuple[2] = returnTuple[2] + 1

        return returnTuple


# Set up all our stuff
def initialize():
    plt.rcParams["figure.figsize"] = [100, 50]
    plt.rcParams.update({'font.size': 15})

    data = pandas.read_csv("mutations.csv")

    print("csv loaded...")

    return data


# Turn data into a more machine readable format
def MakeMachineReadable(data):
    # Generate our datatable for confusion matrix

    d = data
    # first off we're going to generate a more machine readable dataset by transforming the first column into 1s (for cancer) and 0s (noncancer)
    for i in range(0, 230, 1):
        dataAt = data.iloc[i, 0]
        if dataAt[0] == 'N':
            d.iloc[i, 0] = 0
        else:
            d.iloc[i, 0] = 1
    display(d)
    return d
    # d now contains the machine-readable data so we'll now use that


def findMaxPhi(mReadableData):
    maxDict = list()

    numSamples = len(mReadableData.index)

    if (DEBUG):
        print("numsamples: " + str(numSamples))

    for j in range(1, len(mReadableData.columns), 1):
        sam = mReadableData.columns[j]
        # right is positive

        NumSamplesR = 0;
        NumSamplesL = 0
        PosSamplesL = 0;
        NegSamplesL = 0
        PosSamplesR = 0;
        NegSamplesR = 0

        for i in range(0, len(mReadableData.index), 1):
            if (mReadableData.iloc[i, j] == 1):  # right

                NumSamplesR = NumSamplesR + 1

                if (mReadableData.iloc[i, 0] == 1):
                    PosSamplesR = PosSamplesR + 1
                else:
                    NegSamplesR = NegSamplesR + 1
            else:  # left

                NumSamplesL = NumSamplesL + 1

                if (mReadableData.iloc[i, 0] == 1):
                    PosSamplesL = PosSamplesL + 1
                else:
                    NegSamplesL = NegSamplesL + 1

            PL = (NumSamplesL / numSamples)
            PR = (NumSamplesR / numSamples)
            PPosL = 0 if NumSamplesL == 0 else PosSamplesL / NumSamplesL
            PNegL = 0 if NumSamplesL == 0 else NegSamplesL / NumSamplesL
            PPosR = 0 if NumSamplesR == 0 else PosSamplesR / NumSamplesR
            PNegR = 0 if NumSamplesR == 0 else NegSamplesR / NumSamplesR
            Q = np.abs(PPosL - PPosR) + np.abs(PNegL - PNegR)
            Phi = (2 * PL * PR) * Q
            maxDict.append(
                [j, sam, NumSamplesL, NumSamplesR, PosSamplesL, NegSamplesL, PL, PR, PNegL, PPosL, PPosR, PNegR,
                 (2 * PL * PR), Q, Phi])

    maxDict.sort(key=lambda x: x[14])
    maxDict.reverse()

    print("Found Max Phi: " + str(maxDict[0]) + " or " + str(maxDict[1]))

    return maxDict


# Find the max TPSubFP of a machine readable dataframe
def FindMaxTPSubFP(mReadableData):
    trueSub = 0
    trueIndex = 0
    tpSubList = list()

    for j in range(1, len(mReadableData.columns), 1):

        countFalsePos = 0
        countTruePos = 0
        countFalseNeg = 0
        countTrueNeg = 0

        for i in range(0, len(mReadableData.index), 1):
            c = mReadableData.iloc[i, 0]
            t = mReadableData.iloc[i, j]

            if c == 1:
                if c == t:
                    countTruePos = countTruePos + 1
                else:
                    countFalseNeg = countFalseNeg + 1
            else:
                if c != t:
                    countFalsePos = countFalsePos + 1
                else:
                    countTrueNeg = countTrueNeg + 1

        tempSub = countTruePos - countFalsePos
        tpSubList.append([j, tempSub])

    tpSubList.sort(key=lambda x: x[1])
    tpSubList.reverse()

    print("Found MaxTPSubFP: " + str(tpSubList[0]) + " or " + str(tpSubList[1]))

    return tpSubList


# fine the max accuracy of a machine readable dataframe
def FindMaxAccuracy(mReadableData):
    trueSub = 0
    trueIndex = 0
    tpSubList = list()

    for j in range(1, len(mReadableData.columns), 1):

        countFalsePos = 0
        countTruePos = 0
        countFalseNeg = 0
        countTrueNeg = 0

        for i in range(0, len(mReadableData.index), 1):
            c = mReadableData.iloc[i, 0]
            t = mReadableData.iloc[i, j]

            if c == 1:
                if c == t:
                    countTruePos = countTruePos + 1
                else:
                    countFalseNeg = countFalseNeg + 1
            else:
                if c != t:
                    countFalsePos = countFalsePos + 1
                else:
                    countTrueNeg = countTrueNeg + 1

        Acc = (countTruePos + countTrueNeg) / (countTruePos + countTrueNeg + countFalsePos + countFalseNeg)
        tpSubList.append([j, Acc])

    tpSubList.sort(key=lambda x: x[1])
    tpSubList.reverse()

    print("Found Max Accuracy: " + str(tpSubList[0]) + " or " + str(tpSubList[1]))

    return tpSubList


# takes a tree node and finds its leaves
def GenerateNodes(node, idx):
    b2 = BTreeNode()
    b3 = BTreeNode()

    d = node.data

    PositiveIndexes = list()
    NegativeIndexes = list()

    for i in range(0, len(d.index), 1):
        if d.iloc[i, idx] == 0:
            NegativeIndexes.append(d.index[i])
        else:
            PositiveIndexes.append(d.index[i])

    b2.data = d.loc[NegativeIndexes]
    b2.pivot = idx
    b3.data = d.loc[PositiveIndexes]
    b3.pivot = idx

    node.left = b2
    node.right = b3

    return node


# make a tree
def GenerateTree(mReadableData, method):
    b = BTreeNode()
    b.pivot = -1
    b.data = mReadableData

    idx = method(b.data)

    b = BTreeNode()

    b.data = data

    b = GenerateNodes(b, idx[0][0])
    b.pivot = idx[0][0]

    print("data left:")
    display(b.left.data)
    print("data right:")
    display(b.right.data)

    maxL = method(b.left.data)
    maxR = method(b.right.data)

    b.left = GenerateNodes(b.left, maxL[0][0])
    print("left Chose pivot of " + str(maxL[0][0]))
    b.pivot = maxL[0][0]

    b.right = GenerateNodes(b.right, maxR[1][0])
    print("right Chose pivot of " + str(maxL[1][0]))
    b.pivot = maxR[1][0]

    return b


# driver
if __name__ == '__main__':
    data = initialize()
    mReadableData = MakeMachineReadable(data)
    print(mReadableData.columns[3198])

    # Inspired by Ryan's implimentation.
    s = mReadableData.sample(frac=1, replace="False", random_state=RANDOMSEED)
    splits = np.array_split(s, 3)
    g1 = splits[0]
    g2 = splits[1]
    g3 = splits[2]

    t = [g1, g2]
    manufacturePivot1 = pandas.concat(t)

    t2 = [g1, g3]
    manufacturePivot2 = pandas.concat(t2)

    t3 = [g2, g3]
    manufacturePivot3 = pandas.concat(t3)

    TPFP1 = GenerateTree(manufacturePivot1, FindMaxTPSubFP)
    TPFP2 = GenerateTree(manufacturePivot2, FindMaxTPSubFP)
    TPFP3 = GenerateTree(manufacturePivot3, FindMaxTPSubFP)

    Phi1 = GenerateTree(manufacturePivot1, findMaxPhi)
    Phi2 = GenerateTree(manufacturePivot2, findMaxPhi)
    Phi3 = GenerateTree(manufacturePivot3, findMaxPhi)

    Phi1.PopulateMoreRoots()
    Phi2.PopulateMoreRoots()
    Phi3.PopulateMoreRoots()

    print("TP-FP")
    display(TPFP1.data)
    display(TPFP2.left.data)
    display(TPFP3.right.data)

    print("Confusion TPFP")
    c = TPFP1.MakeConfusionWithTree(g3)
    d = TPFP2.MakeConfusionWithTree(g2)
    e = TPFP3.MakeConfusionWithTree(g1)

    display(c)
    display(d)
    display(e)
    TP = c[0]
    FN = c[3]
    FP = c[1]
    TN = c[2]
    confusion = dict()
    print("final confusion tuple: " + str(c))
    confusion["Actual Positive"] = [c[0], c[3]]
    confusion["Actual Negative"] = [c[1], c[2]]
    display(c)

    print("Confusion Phi")
    c = Phi1.MakePhiConfusion(g3)
    d = Phi2.MakePhiConfusion(g2)
    e = Phi3.MakePhiConfusion(g1)

    print("Root pivot: " + str(Phi1.pivot))
    print("Left pivot: " + str(Phi1.left.pivot))

    print("LL Pivot: " + str(Phi1.left.left.pivot) + "Moreof: " + str(Phi1.left.left.moreOf))
    print("LR Pivot: " + str(Phi1.left.right.pivot) + "Moreof: " + str(Phi1.left.right.moreOf))
    print("Right pivot: " + str(Phi1.right.pivot))
    print("RL Pivot: " + str(Phi1.right.left.pivot) + "Moreof: " + str(Phi1.right.left.moreOf))
    print("RR Pivot: " + str(Phi1.right.right.pivot) + "Moreof: " + str(Phi1.right.right.moreOf))

    print("phi")
    print("accuracy: " + str(c[0] + c[2] / c[0] + c[1] + c[2] + c[3]))
    print("sens: " + str(c[0] / c[0] + c[3]))
    print("spec: " + str(c[2] / c[2] + c[1]))
    print("prec: " + str(c[0] / c[0] + c[1]))
    print("miss: " + str(1 - ([0] / c[0] + c[3])))
    print("false disc: " + str(1 - str(c[0] / c[0] + c[1])))
    print("false omm: " + str(c[3] / c[3] + c[2]))

    display(c)
    display(d)
    display(e)
    confusion = dict()
    print("final confusion tuple: " + str(c))
    confusion["Actual Positive"] = [c[0], c[3]]
    confusion["Actual Negative"] = [c[1], c[2]]
    display(c)
    # return a tuple [TP, FP, TN, FN]
    print("TPFP")
    print("accuracy: " + str(c[0] + c[2] / c[0] + c[1] + c[2] + c[3]))
    print("sens: " + str(c[0] / c[0] + c[3]))
    print("spec: " + str(c[2] / c[2] + c[1]))
    print("prec: " + str(c[0] / c[0] + c[1]))
    print("miss: " + str(1 - ([0] / c[0] + c[3])))
    print("false disc: " + str(1 - str(c[0] / c[0] + c[1])))
    print("false omm: " + str(c[3] / c[3] + c[2]))
