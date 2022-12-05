import matplotlib.pyplot as plt
from IPython.display import display
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
import Forest

# Global variables to drive stuff - not good practice but this is my code dangit
DEBUG: int = 1
RANDOMSEED: int = 10024


# Set up all our stuff
def initialize():
    plt.rcParams["figure.figsize"] = [100, 50]
    plt.rcParams.update({'font.size': 15})
    pandas.set_option('display.expand_frame_repr', False)

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
    display(d)
    return d
    # d now contains the machine-readable data so we'll now use that


# run evaluators
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
    display(conf)

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

    edf.columns = ["Accuracy", "Sensitivity", "Specificity", "Precision",
                   "Miss Rate", "False Discovery", "False Omission"]

    print("Evals:")
    display(edf)


# driver
if __name__ == '__main__':
    data = initialize()
    mReadableData = MakeMachineReadable(data)

    # sets created.
    btf: Forest.BTreeForest = Forest.BTreeForest.newInstance(10, 2, mReadableData)

    data1 = data.loc[[0]]
    data2 = data.loc[[14]]
    data3 = data.loc[[99]]
    data4 = data.loc[[12]]
    data5 = data.loc[[31]]

    print("C1")
    btf.evalSample(data1)
    print("C10")
    btf.evalSample(data2)
    print("C50")
    btf.evalSample(data3)
    print("NC5")
    btf.evalSample(data4)
    print("NC15")
    btf.evalSample(data5)

    print("Out of bag:")
    EvalMethod(btf.evalOOB(), 'Out of Bag')
