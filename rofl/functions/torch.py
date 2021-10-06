from .const import *
from .functions import Tmean, Tcat, nn, optim

def getDevice(cudaTry:bool = True):
    if torch.cuda.is_available() and cudaTry:
        print("Using CUDA")
        return Tdevice("cuda")
    return DEVICE_DEFT

# Custom functions
def array2Tensor(arr: ARRAY, device = DEVICE_DEFT, dtype = F_TDTYPE_DEFT, grad: bool = False, batch: bool = False):
    arr = arr if batch else np.squeeze(arr)
    tensor = torch.from_numpy(arr).to(device).to(dtype).requires_grad_(grad)
    tensor = tensor if batch else tensor.unsqueeze(0)
    return tensor

def list2Tensor(arr:list, device = DEVICE_DEFT, dtype = F_TDTYPE_DEFT, grad: bool = False):
    # expecting simple lists with single items (int, float, bool)
    return torch.tensor(arr, dtype = dtype, device = device).unsqueeze(1).requires_grad_(grad)

def copyDictState(net, grad:bool = True):
    newSD = dict()
    sd = net.state_dict()
    for i in sd.keys():
        t = sd[i]
        t = t.new_empty(t.shape).copy_(t)
        if t.dtype == F_TDTYPE_DEFT:
            t.requires_grad_(grad)
        newSD[i] = t
    return newSD

def newNet(net, config = {}):
    netClass = net.__class__
    if config == {}:
        config = net.config
    new = netClass(config)
    return new

def cloneNet(net):
    new = newNet(net, net.config)
    new.load_state_dict(copyDictState(net), strict = True)
    return new.to(net.device)

# from https://docs.ray.io/en/master/serialization.html
# TODO; managing parameters from main policy in __main__ could be 
# done efficiently if the parameters are saved as a list of numpy
# then the update can be done with a 'zero copy read'
# as the torch shared memory is futile with Ray! 
# (seems that TENSOR is not consider native data type)

def getListParams(net):
    params = []
    shared = getattr(net, 'isShared', False)
    for p in net.parameters():
        if not shared:
            targetP = p.clone().detach_()
        else:
            targetP = p.data.numpy()# excludes grad
        params.append(targetP)
    return params

def updateNet(net, targetLoad):
    #net.opParams = copyStateDict(net)
    if isinstance(targetLoad, dict):
        net.load_state_dict(targetLoad)
    elif isinstance(targetLoad, list):
        shared = getattr(net, 'isShared', False)
        for p, pt in zip(targetLoad, net.parameters()):
            pt.requires_grad_(False) # This is a must to change the values properly
            if not shared:
                pt.copy_(p).detach_()
            else:
                # This should be used only when net.device is CPU
                pt.data = torch.from_numpy(p)
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

def oldgetListParams(net):
    params = []
    for p in net.parameters():
        params += [p.clone().detach_()]
    return params

def getParams(policy):
    pi = policy.actor
    baseline = getattr(policy, 'baseline')
    isAC = getattr(policy, 'actorHasCritic', False)
    piParams = getListParams(pi)
    blParams = [] if baseline is None or isAC else getListParams(baseline)
    return piParams, blParams

def maxGrad(net):
    return max(p.grad.detach().abs().max() for p in net.parameters()).item()

def meanGrad(net):
    return Tmean(torch.tensor([Tmean(p.grad.detach()) for p in net.parameters()])).item()

def zeroGrad(net):
    zeroGradParams(net.parameters())

def zeroGradParams(parameters):
    for p in parameters:
        if p.grad is None:
            p.grad = p.new_zeros(p.shape)
        else:
            p.grad.fill_(0)

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
    
def getGradients(net):
    grads = []
    for p in net.parameters():
        grads.append(p.grad.clone().detach_())
    return grads

def accumulateGrad(net, *grads):
    for grad in grads:
        for p, g in zip(net.parameters(), grad):
            p.grad.add_(g)

def tryCopy(T: TENSOR):
    if isinstance(T, TENSOR): 
        return T.clone().detach()
    elif isinstance(T, ARRAY):
        return np.copy(T)
    else:
        from copy import deepcopy
        return deepcopy(T)

def getOptimizer(config: dict, network, deftLR = OPTIMIZER_LR_DEF, key: str = 'network'):
    """
        Usually all optimizers need at least two main arguments

        parameters of the network and a learning rate. More argumens
        should be declared in the 'optimizer_args' into config.policy

        parameters
        ----------
        - configDict: dict

        - network: nn.Module type object
        - deftLR: float
            A default learning rate if there's none declared in the config
            dict. A config dict by deault does not have this argument.
        - key: str
            Default 'network'. Depends on the key to get the configuration from
            the dict config. Eg, 'baseline' to generate an optimizer with those
            configs.
            
        returns
        --------
        optimizer for network
    """
    name = config['policy'][key].get('optimizer')
    lr = config['policy'][key].get('learning_rate', deftLR)

    if name == 'adam':
        FOpt = optim.Adam
    elif name == 'rmsprop':
        FOpt = optim.RMSprop
    elif name == 'sgd':
        FOpt = optim.SGD
    elif name == 'adagrad':
        FOpt = optim.Adagrad
    elif name == 'dummy':
        FOpt = dummyOptimizer
    else:
        print("Warning: {} is not a valid optimizer. {} was generated instead".format(name, OPTIMIZER_DEF))
        config['policy'][key]['optimizer'] = OPTIMIZER_DEF
        config['policy'][key]['learning_rate'] = OPTIMIZER_LR_DEF
        config['policy'][key]['optimizer_args'] = {}
        return getOptimizer(config, network)

    return FOpt(network.parameters(), lr = lr, **config['policy'][key].get("optimizer_args", {}))

class dummyOptimizer():
    egg = 'Do nothing, receive everything (?)'

    def __init__(self, parameters, lr = 0, **kwargs):
        self.parameters = [p for p in parameters]

    def __repr__(self) -> str:
        return self.egg

    def zero_grad(self):
        zeroGradParams(self.parameters)

    def step(self):
        pass
