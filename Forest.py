import pandas
import Node
import Methods

class BTreeForest:

    def __init__(self):
        trees: list = list()
        numtrees: int
        treedepth: int

    def newInstance(numTrees: int, treeDepth: int, data: pandas.DataFrame):
        self = BTreeForest()
        self.numtrees = numTrees
        self.treedepth = treeDepth
        self.trees = list()

        for i in range(0, numTrees, 1):
            print("tree " + str(i))
            self.trees.append(Node.BTreeNode.GenerateTreeSqrt(data, Methods.findMaxPhi))

        return self

    def evalSample(self, data: pandas.DataFrame):

        # [TP, FP, TN, FN]
        positives = 0
        negatives = 0

        for tree in self.trees:
            t = tree.MakePhiConfusion(data)
            if t[0] + t[1] >= t[2] + t[3]:  #  [TP, FP, TN, FN]
                positives = positives + 1
            else:
                negatives = negatives + 1

        print("Positives: " + str(positives) + "\tNegatives: " + str(negatives))
        if positives > negatives:
            return True
        else:
            return False

    def evalOOB(self):
        samples = [0, 0, 0, 0]  # [TP, FP, TN, FN]

        for tree in self.trees:
            t = tree.MakePhiConfusion(tree.outOfBag)
            samples[0] = samples[0] + t[0]
            samples[1] = samples[1] + t[1]
            samples[2] = samples[2] + t[2]
            samples[3] = samples[3] + t[3]

        return samples
