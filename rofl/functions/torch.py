from .const import *
from .functions import Tmean, Tcat, multiplyIter, nn, optim

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

### Meant to be used to share information between BaseNets of the same type ###
# order, shapes and dtypes of parameters are NOT in check #
# while  using ray, ndarrays is the best way to share information
# between main and workers, as their serialization is done in shared memory
# from https://docs.ray.io/en/master/serialization.html

def getListNParams(net):
    '''
        Ordered list of ARRAYS for a network's parameters
    '''
    params = []
    for p in net.parameters():
        targetP = p.data.cpu().numpy()
        params.append(targetP)
    return params

def updateNet(net, targetLoad):
    if isinstance(targetLoad, dict):
        net.load_state_dict(targetLoad)

    elif isinstance(targetLoad, list):
        for p, pt in zip(targetLoad, net.parameters()):
            pt.requires_grad_(False) # This is a must to change the values properly
            #p = np.copy(p) # works better(in time) using a copy (weird!?!)
            pt.data = torch.from_numpy(p).to(pt.device)
            pt.requires_grad_(True)
    else:
        raise ValueError('Should be either a state_dict or a list of ndarrays')

def getParams(policy):
    """
        Returns lists of ARRAYS of the parameters for actor and baseline (if any 
        different from actor).
    """
    pi = policy.actor
    baseline = getattr(policy, 'baseline')
    isAC = getattr(policy, 'actorHasCritic', False)
    piParams = getListNParams(pi)
    blParams = [] if baseline is None or isAC else getListNParams(baseline)
    return piParams, blParams

def getDictState(net, cpu:bool = True):
    stateDict = net.state_dict()
    if cpu:
        for key in stateDict.keys():
            stateDict[key] = stateDict[key].to(DEVICE_DEFT)
    return stateDict

def getListTParams(net, device = None):
    '''
        Ordered list of TENSORS for a network's parameters.
        These are detached and can be directed to a device.
    '''
    params = []
    for p in net.parameters():
        p = p.detach()
        if device is not None:
            p.to(device)
        params.append(p)
    return params

def maxGrad(net):
    return max(p.grad.detach().abs().max() for p in net.parameters()).item()

def meanGrad(net):
    return Tmean(torch.tensor([Tmean(p.grad.detach()) for p in net.parameters()])).item()

def zeroGrad(obj, isPolicy: bool = False):
    if not isPolicy:
        zeroGradParams(obj.parameters())
    else:
        pi = obj.actor
        bl = getattr(obj, 'baseline')
        useBl = getattr(obj, 'doBaseline', False)
        zeroGradParams(pi.parameters())
        if useBl:
            zeroGradParams(bl.parameters())

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
        shapes.append(p.shape)
        flat.append(p.flatten())
    return Tcat(flat, dim=0), shapes

def convertFromFlat(x, shapes):
    newX, iL, iS = [], 0, 0
    for s in shapes:
        iS = iL + multiplyIter(s)
        newX.append(x[iL:iS].reshape(s))
        iL = iS
    return newX
    
def getNGradients(net):
    '''
        Returns a list of ARRAYS for the gradients
        in the networks parameters
    '''
    grads = []
    for p in net.parameters():
        grads.append(p.grad.cpu().numpy())
    return grads

def accumulateGrad(net, *grads):
    for grad in grads:
        for p, g in zip(net.parameters(), grad):
            p.grad.add_(torch.from_numpy(g).to(p.device))

def tryCopy(T: TENSOR):
    if isinstance(T, TENSOR): 
        return T.detach()
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
