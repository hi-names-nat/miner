import numpy as np
from IPython.display import display
import pandas
import Main


# Node tree class, used for the BTree
class BTreeNode:
    def __init__(self):
        self.right: BTreeNode = None
        self.left: BTreeNode = None
        self.data: pandas.DataFrame = None
        self.outOfBag: pandas.DataFrame = None
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
    def __FindPosNeg(self, isRight: bool):  # return a tuple [TP, FP, TN, FN]
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

    def __GetConfusionForNode(self, isRight : bool):
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

    def MakeConfusionWithTree(self, mReadableData : pandas.DataFrame):  # return a tuple [TP, FP, TN, FN]
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
        p = 0
        np = 0
        for i in range(0, len(self.data.index), 1):
            c = self.data.iloc[i, 0]

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

                p = mReadableData.iloc[i]
                pivot = p.loc[self.pivot]


                if pivot == 1:
                    root = root.right
                else:
                    root = root.left

            t = root.moreOf
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

    def GenerateTreeSqrt(mReadableData: pandas.DataFrame, method):
        b = BTreeNode()
        b.pivot = -1

        b.data = mReadableData.sample(frac=1, replace=True)
        print("Bag:")
        display(b.data)
        b.outOfBag = mReadableData.merge(b.data, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        print("OOB:")
        display(b.outOfBag)

        sqrtn = int(np.sqrt(len(b.data.columns)))
        trueValues = b.data.loc[:, 'Unnamed: 0']
        s = b.data.drop(columns='Unnamed: 0')
        samples: pandas.DataFrame = s.sample(n=sqrtn, replace=False, axis='columns')
        samples.insert(0, column='Unnamed: 0', value=trueValues)

        print("\n\nRoot:")

        idx = method(samples)

        b.GenerateNodes(idx[0][0])
        b.pivot = idx[0][0]
        print("Root chose pivot of " + str(idx[0]))


        print("\n\nLeft: ")
        sqrtn = int(np.sqrt(len(b.left.data)))

        # sampling from the sorted dataset
        sqrtn = int(np.sqrt(len(b.left.data.columns)))
        trueValues = b.left.data.loc[:, 'Unnamed: 0']
        s = b.left.data.drop(columns='Unnamed: 0')
        samples: pandas.DataFrame = s.sample(n=sqrtn, replace=False, axis='columns')
        samples.insert(0, column='Unnamed: 0', value=trueValues)

        maxL = method(samples)
        i = 0

        for i in range(0, len(maxL), 1):
            if maxL[i][0] != b.pivot:
                break

        b.left.GenerateNodes(maxL[i][0])
        b.left.pivot = maxL[i][0]
        print("left Chose pivot of " + str(maxL[i]))

        print("\n\nRight: ")

        # sampling from the sorted dataset

        sqrtn = int(np.sqrt(len(b.right.data.columns)))
        trueValues = b.right.data.loc[:, 'Unnamed: 0']
        s = b.right.data.drop(columns='Unnamed: 0')
        samples: pandas.DataFrame = s.sample(n=sqrtn, replace=False, axis='columns')
        samples.insert(0, column='Unnamed: 0', value=trueValues)


        maxR = method(samples)
        i = 0

        for i in range(0, len(maxR), 1):
            if maxR[i][0] != b.pivot:
                break
        b.right.GenerateNodes(maxR[i][0])
        b.right.pivot = maxR[i][0]
        print("right Chose pivot of " + str(maxR[i]))

        b.PopulateMoreRoots()

        return b

    def GenerateNodes(self, idx):
        b2 = BTreeNode()
        b3 = BTreeNode()

        d = self.data.loc[~self.data.index.duplicated(), :].copy()

        PositiveIndexes = list()
        NegativeIndexes = list()

        for i in d.index:
            if d.loc[i, idx] == 0:
                NegativeIndexes.append(i)
            else:
                PositiveIndexes.append(i)

        b2.data = d.loc[NegativeIndexes]
        b2.pivot = -1
        b3.data = d.loc[PositiveIndexes]
        b3.pivot = -1

        self.left = b2
        self.right = b3

        if Main.DEBUG == 1:
            print("left:")
            display(b2.data)
            print("right:")
            display(b3.data)
