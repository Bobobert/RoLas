from rofl.config import createConfig
from rofl.utils import pathManager
import torch

config = createConfig(expName = 'test')
config['env']['name'] = 'test0'

manager = pathManager(config)
obj2Save = [i for i in range(100)]

saver = manager.startSaver(10)
resTest = {'test':0}

saver.addObj(torch.tensor(obj2Save), 'testList', False)
saver.addObj(obj2Save, 'testListMax', key = 'test')
saver.addObj(obj2Save, 'testListMin', key = 'test', discardMin= False)

for i in range(30):
    saver.check(resTest)
    resTest['test'] += 1

saver.saveAll(resTest)
manager.close()
