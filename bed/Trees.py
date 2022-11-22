import matplotlib.pyplot as plt
import pandas
import numpy as np
import IPython.display as display
import pprint as pp
import node


# Global variables to drive stuff - not good practice but this is my code dangit
DEBUG: int = 1
RANDOMSEED: int = 10024

# Set up all our stuff
def initialize():
    plt.rcParams["figure.figsize"] = [100, 50]
    plt.rcParams.update({'font.size': 15})

    data = pandas.read_csv("mutations.csv")

    print("csv loaded...")

    pandas.set_option('display.max_columns', 30)

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
    display.display(d)
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

        NumSamplesR = 0
        NumSamplesL = 0
        PosSamplesL = 0
        NegSamplesL = 0
        PosSamplesR = 0
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

    return maxDict

def __FindEntropy(mReadableData):
    numSamples = len(mReadableData.index)
    NumPosSamples = 0
    NumNegSamples = 0
    for i in range(0, len(mReadableData.index), 1):
        if (mReadableData.iloc[i, 0] == 1):  # right
            NumPosSamples = NumPosSamples + 1
        else:  # left
            NumNegSamples = NumNegSamples + 1

    ProbPos = -1
    ProbNeg = -1

    if numSamples == 0:
        ProbPos = 0
        ProbNeg = 0

    else:
        ProbPos = NumPosSamples / numSamples
        ProbNeg = NumNegSamples / numSamples

    logRootPos = -1
    if (NumPosSamples == 0):
        logRootPos = 0
    else:
        logRootPos = np.log2(ProbPos)

    logRootNeg = -1
    if (NumNegSamples == 0):
        logRootNeg = 0
    else:
        logRootNeg = np.log2(ProbNeg)

    H = -((ProbPos * logRootPos) + (ProbNeg * logRootNeg))

    return [H, numSamples, NumPosSamples, NumNegSamples]


def FindMaxInformationGain(mReadableData):
    maxDict = list()

    numSamples = len(mReadableData.index)

    if (DEBUG):
        print("numsamples: " + str(numSamples))

    EntRoot = __FindEntropy(mReadableData)[0]

    for j in range(1, len(mReadableData.columns), 1):
        # right is positive

        PositiveIndexes = list()
        NegativeIndexes = list()

        for i in range(0, len(mReadableData.index), 1):
            if mReadableData.iloc[i, j] == 0:
                NegativeIndexes.append(mReadableData.index[i])
            else:
                PositiveIndexes.append(mReadableData.index[i])

        dataLeft = mReadableData.loc[NegativeIndexes]
        dataRight = mReadableData.loc[PositiveIndexes]

        EntL = __FindEntropy(dataLeft)
        NumSamplesL = EntL[1]
        NumPosSamplesL = EntL[2]
        NumNegSamplesL = EntL[3]
        EntL = EntL[0]

        EntR = __FindEntropy(dataRight)
        NumSamplesR = EntR[1]
        NumPosSamplesR = EntR[2]
        NumNegSamplesR = EntR[3]
        EntR = EntR[0]

        P_L = (NumSamplesL / numSamples)
        P_R = (NumSamplesR / numSamples)

        EntST = (P_L * EntL) + (P_R * EntR)

        Gain = EntRoot - EntST


        maxDict.append([j, data.columns[j], NumSamplesL, NumSamplesR, NumPosSamplesL, NumNegSamplesL,
                       NumPosSamplesR, NumNegSamplesR, P_L, P_R, EntST, EntST, Gain])

    maxDict.sort(key=lambda x: x[12])
    maxDict.reverse()

    print("Found Max IG: " + str(maxDict[0]) + " or " + str(maxDict[        1]))


    return maxDict


# takes a tree node and finds its leaves
def GenerateNodes(node, idx):
    b2 = node.BTreeNode()
    b3 = node.BTreeNode()

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

    if (DEBUG == 1):
        print("left:")
        display.display(b2.data)
        print("right:")
        display.display(b3.data)

    return node

    #enumerate

# make a tree
def GenerateTree(mReadableData, method):
    b = node.BTreeNode()
    b.pivot = -1
    b.data = mReadableData



    print("\n\nRoot:")
    idx = method(b.data)

    b = node.BTreeNode()

    b.data = data

    b = GenerateNodes(b, idx[0][0])
    b.pivot = idx[0][0]

    print("\n\nLeft: ")
    maxL = method(b.left.data)

    if (maxL[0][0] == b.pivot):
        b.left = GenerateNodes(b.left, maxL[1][0])
        b.pivot = maxL[1][0]
        print("left Chose pivot of " + str(maxL[1]))

    else:
        b.left = GenerateNodes(b.left, maxL[0][0])
        b.left.pivot = maxL[0][0]
        print("left Chose pivot of " + str(maxL[0]))


    print("\n\nRight: ")
    maxR = method(b.right.data)
    if (maxR[0][0] == b.pivot):
        b.right = GenerateNodes(b.right, maxR[1][0])
        b.pivot = maxR[1][0]
        print("right Chose pivot of " + str(maxR[1]))
    else:
        b.right = GenerateNodes(b.right, maxR[0][0])
        b.pivot = maxR[0][0]
        print("right Chose pivot of " + str(maxR[0]))

    return b

def EvalMethod(c, whatIs : str):
    # return a tuple [TP, FP, TN, FN]
    TP = c[0]
    FP = c[1]
    TN = c[2]
    FN = c[3]

    print("this is: " + whatIs)

    confusion = dict()
    confusion["Actual Positive"] = [TP, FN]
    confusion["Actual Negative"] = [FP, TN]
    conf = pandas.DataFrame.from_dict(confusion, "index")
    conf.columns = ["Test Positive", "Test Negative"]
    print("Confusion:")
    display.display(conf)

    acc = (TP + TN) / (TP + TN + FP + FN)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    prec = TP / (TP + FP)
    miss = FN / (FN + TP)
    disc = FP / (FP + TP)
    omm = FN / (FN + TN)

    evals = dict()
    evals[whatIs] = [acc, sens, spec, prec, miss, disc, omm]
    edf = pandas.DataFrame.from_dict(evals, "index")

    edf.columns = ["Accuracy","Sensitivity", "Specificity", "Precision", "Miss Rate",
                   "False Discovery", "False Ommision"]

    print("Evals:")
    display.display(edf)


# driver
if __name__ == '__main__':
    data = initialize()
    mReadableData = MakeMachineReadable(data)


    # Inspired by Ryan's implimentation.
    s = mReadableData.sample(frac=1, replace="True", random_state= RANDOMSEED)

