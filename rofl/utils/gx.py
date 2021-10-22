from rofl.functions.const import *
from rofl.functions.functions import nprnd
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

def showBuffer(memory, samples:int = 20, Wait:int = 3):
    """
        Meant to be used with dqnMemory, as prints contents 
        from the 'frame' contents of each experience
    """
    # Drawing samples
    for i in nprnd.randint(memory._li_, memory._i_, size=samples):
        plt.ion()
        fig = plt.figure(figsize=(10,3))
        item = memory[i]
        plt.title('Non-terminal' if item['done'] else 'Terminal')
        plt.axis('off')
        for n, j in enumerate(range(memory.lhist)):
            fig.add_subplot(1, memory.lhist, n + 1)
            plt.imshow(item['frame'][j])
            plt.axis('off')
        plt.pause(Wait)
        plt.close(fig)

def showFrameFromTensor(obs, i = 0):
    obsClone = obs.clone().detach().cpu().squeeze().numpy()
    plt.imshow(obsClone[i])
    plt.show(block = False)
