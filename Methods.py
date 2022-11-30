import numpy as np
import Main
import pandas


def findMaxPhi(mReadableData: pandas.DataFrame):
    maxDict = list()

    cleanedValues = mReadableData.loc[~mReadableData.index.duplicated(), :].copy()

    numSamples = len(mReadableData.index)

    if (Main.DEBUG):
        print("numsamples: " + str(numSamples))

    for j in mReadableData.columns:
        if j == 'Cancerous':
            continue
        # right is positive
        n: pandas.DataFrame = cleanedValues.loc[:, j]

        NumSamplesR = 0
        NumSamplesL = 0
        PosSamplesL = 0
        NegSamplesL = 0
        PosSamplesR = 0
        NegSamplesR = 0

        for i in cleanedValues.index:
            t = n.loc[i]
            if t == 1:  # right

                NumSamplesR = NumSamplesR + 1

                t = cleanedValues.loc[i]
                if t.loc['Cancerous'] == 1:
                    PosSamplesR = PosSamplesR + 1
                else:
                    NegSamplesR = NegSamplesR + 1
            else:  # left

                NumSamplesL = NumSamplesL + 1

                t = cleanedValues.loc[i]
                if t.iloc[0] == 1:
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
            [j, NumSamplesL, NumSamplesR, PosSamplesL, NegSamplesL, PL, PR, PNegL, PPosL, PPosR, PNegR,
             (2 * PL * PR), Q, Phi])

    maxDict.sort(key=lambda x: x[13])
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

    if (Main.DEBUG):
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


        maxDict.append([j, mReadableData.columns[j], NumSamplesL, NumSamplesR, NumPosSamplesL, NumNegSamplesL,
                       NumPosSamplesR, NumNegSamplesR, P_L, P_R, EntST, EntST, Gain])

    maxDict.sort(key=lambda x: x[12])
    maxDict.reverse()

    print("Found Max IG: " + str(maxDict[0]) + " or " + str(maxDict[        1]))


    return maxDict
