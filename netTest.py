from rofl.config import createConfig, getEnvMaker, createNetwork
from rofl.utils import pathManager
from rofl.functions import getDevice

import torch
import timeit

targetConfig = {
    'env' : {
        'name' : 'CartPole-v0',
        'envMaker' : 'gymEnvMaker',
    },
    'policy' : {
        'n_actions' : 4,
        'network' : {
            'networkClass' : 'gymActor',
            'linear_3' : 32,
            'linear_4' : 56,
            'linear_2' : 64,
            'linear_1' : 100,
        },
        'oldNetwork':{
            'networkClass' : 'gymActorOld'
        }
    }

}
device = getDevice(False)
config = createConfig(targetConfig, expName = 'pg')
maker = getEnvMaker(config)
actor = createNetwork(config).to(device)
actorOld = createNetwork(config, 'oldNetwork').to(device)

"""print(dir(actor),actor._layers_)
t = torch.zeros((1,4))
print(actor(t))
for p in actor.parameters():
    print(p.shape, p.names)"""

import torch.nn.functional as F
from rofl.functions.torch import zeroGrad


def testSpeed(net):
    t = torch.randn((10, 4), requires_grad=True, device=device)
    out = net(t)
    loss = torch.sum(out)
    loss.backward()
    zeroGrad(net)

def testNew():
    testSpeed(actor)

def testOld():
    testSpeed(actorOld)

if __name__ == '__main__':

    print('new actor:',timeit.timeit(testNew, number=10**4))
    print(actor._layers_)
    print('old actor:',timeit.timeit(testOld, number=10**4))
    print(dir(actorOld))


