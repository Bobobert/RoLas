from .const import *

def getDevice(cudaTry:bool = True):
    if torch.cuda.is_available() and cudaTry:
        return Tdevice("cuda")
    return DEVICE_DEFT

# Custom functions
def toT(arr: ARRAY, device = DEVICE_DEFT, dtype = F_TDTYPE_DEFT, grad: bool = False):
        arr = np.squeeze(arr)
        return torch.as_tensor(arr, dtype = dtype, device = device).unsqueeze(0).requires_grad_(grad)

def copyDictState(net, grad:bool = True):
    newSD = dict()
    sd = net.state_dict()
    for i in sd.keys():
        t = sd[i]
        newSD[i] = t.new_empty(t.shape).copy_(t).requires_grad_(grad)
    return newSD

def cloneNet(net):
    new = net.new()
    new.load_state_dict(copyDictState(net), strict = True)
    return new.to(net.device)

def updateNet(net, targetLoad):
    #net.opParams = copyStateDict(net)
    if isinstance(targetLoad, dict):
        net.load_state_dict(targetLoad)
    elif isinstance(targetLoad, list):
        for p, pt in zip(targetLoad, net.parameters()):
            pt.requires_grad_(False) # This is a must to change the values properly
            pt.copy_(p).detach_()
            pt.requires_grad_(True)

def getDictState(net, cpu:bool = True):
    stateDict = net.state_dict()
    if cpu:
        for key in stateDict.keys():
            stateDict[key] = stateDict[key].to(DEVICE_DEFT)
    return stateDict

def getListState(net, cpu:bool = True):
    params = []
    for p in net.parameters():
        params += [p if not cpu else p.clone().to(DEVICE_DEFT)]
    return params

def getListParams(net):
    params = []
    for p in net.parameters():
        params += [p.clone().detach_()]
    return params

def analysisGrad(net, calculateMean: bool = False, calculateMax: bool = True):
    """
        Should return 
    """
    max_grad, mean_grad = 0.0, 0.0
    if calculateMax:
        max_grad = max(p.grad.detach().abs().max() for p in net.parameters()).item()
    if calculateMean:
        mean_grad = Tmean(torch.tensor([Tmean(p.grad.detach()) for p in net.parameters()])).item()
    return max_grad, mean_grad

def zeroGrad(net):
    for p in net.parameters():
        p.grad = p.new_zeros(p.shape)

def noneGrad(net):
    for p in net.parameters():
        p.grad = None

def clipGrads(net, clip:float):
    nn.utils.clip_grad_value_(net.parameters(), clip)

def cloneState(states, grad: bool = True, ids = None):
    if ids is not None:
        assert isinstance(ids, (ARRAY, TENSOR)), "ids must be a ndarray or a torch.Tensor"
    def cloneT(T):
        return torch.clone(T).detach_().requires_grad_(grad)
    if isinstance(states, TENSOR):
        if ids is not None:
            states = states[ids]
        return cloneT(states)
    elif isinstance(states, dict):
        new = dict()
        for key in states.keys():
            sk = states[key]
            if ids is not None:
                sk = sk[ids]
            new[key] = cloneT(sk)
        return new
    else:
        raise TypeError("State type {} not supported".format(type(states)))

def convert2flat(x):
    shapes = []
    flat = []
    for p in x:
        shapes += [p.shape]
        flat += [p.flatten()]
    return Tcat(flat, dim=0), shapes

def totSize(t):
    tot = 1
    for i in t:
        tot *= i
    return tot

def convertFromFlat(x, shapes):
    newX, iL, iS = [], 0, 0
    for s in shapes:
        iS = iL + totSize(s)
        newX += [x[iL:iS].reshape(s)]
        iL = iS
    return newX