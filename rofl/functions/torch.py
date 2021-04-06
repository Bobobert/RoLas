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
        newSD[i] = t.new_empty(t.shape, requires_grad=grad).copy_(t)
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

def getDictState(net, cpu:bool):
    stateDict = net.state_dict()
    if cpu:
        for key in stateDict.keys():
            stateDict[key] = stateDict[key].to(DEVICE_DEFT)
    return stateDict

def getListState(net, cpu:bool):
    params = []
    for p in net.parameters():
        params += [p if not cpu else p.clone().to(DEVICE_DEFT)]
    return params
    
def analysisGrad(net):
    """
        Should return 
    """
    max_grad, mean_grad = 0.0, 0.0
    max_grad = max(p.grad.detach().abs().max() for p in net.parameters()).item()
    mean_grad = torch.mean(torch.tensor([torch.mean(p.grad.detach()) for p in net.parameters()])).item()
    return max_grad, mean_grad