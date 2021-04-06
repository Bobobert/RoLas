from rofl.functions.const import *
import matplotlib.pyplot as plt
from rofl.utils.utils import timeFormated

def graphResults(accRewards, var, meanRand:int, varRand:int, iters:int, testFreq:int, 
                name:str = "", dpi:int = 250, show:bool = False):
    """
    Put a smile on that face!
    pyplot gx function for results from the train and testRun functions
    """
    if show: plt.ion()
    fig = plt.figure(figsize=(11,6), dpi=dpi)
    x = np.arange(0, iters, testFreq)
    accRewards = np.array(accRewards)
    std = np.sqrt(np.array(var))
    meanRand = np.array([meanRand for _ in x])
    stdRnd = np.array(np.array([varRand for _ in x]))
    plt.plot(x, accRewards, label="TrainedPolicy", color=CLRPI)
    plt.fill_between(x, accRewards - std, accRewards + std, alpha=ALPHA,
                        ls = "-.", lw=LINEWDT, color=CLRPI)
    
    plt.plot(x, meanRand, ls = "-", lw=LINEWDT, color=CLRRDM, label="RandomPolicy")
    lMn, lMx = np.min(accRewards - std), np.max(accRewards + std)
    sMn, sMx = np.min(meanRand - stdRnd), np.max(meanRand + stdRnd)
    if (sMn >= lMn) and (sMx <= lMx):
        plt.fill_between(x, meanRand - stdRnd, meanRand + stdRnd, alpha=ALPHA, color =CLRRDM)
    plt.title("Training results " + name)
    plt.xlabel("Updates")
    plt.ylabel("Mean Accumulated reward")
    plt.legend()
    plt.savefig("mean_acc_reward_"+ name +"_"+timeFormated(), dpi=dpi)
    if show: plt.show()