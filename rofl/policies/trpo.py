from rofl.functions.const import TENSOR
from rofl.functions.functions import F, no_grad, np, Tdot, Tsqrt, torch, Texp, reduceBatch, Tmean, Tmul
from rofl.functions.torch import cloneNet, getListTParams, tensors2Flat, flat2Tensors, getGradients, noneGrad, updateNet
from rofl.policies.pg import pgPolicy
from rofl.policies.ppo import putVariables
from rofl.utils.policies import getBaselines, setEmptyOpt, calculateGAE, trainBaseline
from rofl.functions.distributions import kl_divergence

class trpoPolicy(pgPolicy):
    '''
        Basic TRPO policy

        Does the fisher information matrix with a double auto-grad technique.

        Expecting an Actor and, optionally, a baseline.
    '''
    name = 'trpo v0'

    def initPolicy(self, **kwargs):
        super().initPolicy(**kwargs)
        self.actorOther = cloneNet(self.actor)
        
        config = self.config
        self.cgIters = config['policy']['cg_iterations']
        self.lsIters = config['policy']['ls_iterations']
        putVariables(self)

    def update(self, *batchDict):
        oldDistributions, actor = [], self.actor

        for dict_ in batchDict:
            with no_grad():
                params = actor.onlyActor(dict_['observation'])
                dist = actor.getDist(params)
            oldDistributions.append(dist)
        
        for dict_, dist in zip(batchDict, oldDistributions):
            self.batchUpdate(dict_, dist)

        self.newEpoch = True
        self.epoch += 1

    def batchUpdate(self, batchDict, distFix):
        # unpack from batch
        observations, actions, returns = batchDict['observation'], batchDict['action'], batchDict['return']
        logProbOld = reduceBatch(batchDict['log_prob'])
        NInv = 1.0 / observations.shape[0]
        
        # calculate advantages
        actor = self.actor
        baselines = getBaselines(self, observations)

        if self.gae:
            advantages = calculateGAE(self, baselines, batchDict['next_observation'],\
                batchDict['done'], batchDict['reward'], self.gamma, self.lmbd)
        else:
            advantages = returns - baselines.detach()
        advantages.squeeze_()
        
        # define functions
        actor = self.actor
        _F, _FBl = Tmean, F.mse_loss

        def calculateSurrogate(stateDict=None):

            if stateDict is not None:
                pi = self.actorOther
                updateNet(pi, stateDict)
                with no_grad():
                    params = pi.onlyActor(observations)
            else:
                pi = actor
                params = pi.onlyActor(observations)

            logProbs, _ = pi.processDist(params, actions)
            logProbs = reduceBatch(logProbs)
            ratio = Texp(logProbs - logProbOld)
            surrogate = Tmul(ratio, advantages)

            return _F(surrogate)
        
        def grads4Loss(loss: TENSOR, retainGraph:bool = False):
            loss.backward(create_graph=retainGraph, retain_graph=retainGraph)
            return getGradients(self.actor, retainGraph)

        Grad = torch.autograd.grad
        def fisherVectorP(v, shapes):
            # hvp adhoc - based on the pytorch hvp function
            noneGrad(actor)
            params = actor.onlyActor(observations)
            dist = actor.getDist(params)
            klDiv = (kl_divergence(distFix, dist) * NInv,)
            
            params = tuple(getListTParams(actor, detach = False))
            gradOutputs = (None,) * len(klDiv)
            jacobian = Grad(klDiv, params, gradOutputs, create_graph = True)

            gradJac = tuple(torch.zeros_like(par, requires_grad = True) for par in params)
            doubleBack = Grad(jacobian, params, gradJac, create_graph = True)

            v = tuple(flat2Tensors(v, shapes))
            hvp = Grad(doubleBack, gradJac, v)
            hvp, _ = tensors2Flat(hvp)

            return hvp

        # Copy of actual params for later, perhaps not necesary
        actorParams = getListTParams(actor)
        # Calculate gradient respect to L(Theta)
        surrogate = calculateSurrogate()
        policyGradient = grads4Loss(surrogate)

        # solve search direction s~A^-1g with conjugate gradient
        stepDir, stepDirShapes = conjugateGrad(fisherVectorP, policyGradient, self.cgIters)
        fvp = fisherVectorP(stepDir, stepDirShapes)
        sHs = Tdot(stepDir, fvp)
        beta = Tsqrt(2 * self.maxKLDiff / sHs) # step length B
        fullStep = stepDir * beta # Bs, this will be added to the parameters through a linea search

        flatPG, _ = tensors2Flat(policyGradient)
        GStepDir = Tdot(flatPG, fullStep)
        
        success, theta = lineSearch(calculateSurrogate, actorParams, fullStep, GStepDir, self.lsIters)
        if success: updateNet(actor, theta)

        if self.doBaseline:
            lossBaseline = trainBaseline(self, baselines, returns, _FBl)
        else:
            lossBaseline = torch.zeros(())

        tbw, epoch = self.tbw, self.epoch
        if tbw != None and (epoch % self.tbwFreq == 0) and self.newEpoch:
            tbw.add_scalar('train/Actor loss', surrogate.cpu().item(), epoch)
            tbw.add_scalar('train/Baseline loss', lossBaseline.cpu().item(), epoch)
            tbw.add_scalar('train/LineSearch success', success, epoch)
            #self._evalTBWActor_()
        self.newEpoch = False

def conjugateGrad(mvp, b, iters: int = 10, epsilon: float = 1e-10):
    """
    Conjugated gradietns in pytorch

    based on https://web.stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf
    
    parameters
    ----------
    mvp: matrix-vector product function
        Any from the approximation of the fisher-vector
        product
    b: list of Tensors
        Such as the gradients to solve s = A⁻1 g
    iters: int
        number of iterations to run the algorithm
    epsilon: float
        The threshold to finish early the algorithm when
        sqrt(rho_{k-1}) \leq \epsilon * |b|
    doMax: bool
        Default True if the problem is to maximize on b.
    """
    # Init
    b, shapes = tensors2Flat(b)
    x = b.new_zeros(b.shape)
    r = b.detach()
    rho = Tdot(r, r) 

    # Iterations
    for k in range(iters):
        if rho < epsilon:
            break
        oldRho = rho
        
        if k == 0:
            p = r
        else:
            p = r + rhoRatio * p

        Ap = mvp(p, shapes)
        pAp = Tdot(p, Ap)
        alpha = rho / pAp
        x += alpha * p
        r -= alpha * Ap
        rho = Tdot(r, r)
        rhoRatio = rho / oldRho

    return x, shapes

def lineSearch(f, x, direction, 
                expectedImproveRate,
                maxBacktracks: int = 10,
                acceptRatio:float = 0.1):
    """
    Backtracking Line search algorithm with Armijo condition,

    http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/Descent-Line-Search.pdf 
    
    Based on https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
    
    params
    ------
    f: function 
        to evaluate the direction on
    x: tensor
        Start point for evaluation
    direction: tensor
        Direction to search on
    expectedImproveRate: tensor
        The expected amount of improvement, from TRPO this
        amount is the slope dy/dx at the input

    """
    fx = f(x)
    x, xShapes = tensors2Flat(x)
    for stepFrac in 0.5 ** np.arange(maxBacktracks):
        newX =  x + stepFrac * direction
        newfx = f(flat2Tensors(newX, xShapes))
        improvement = fx - newfx
        expectedImprovement = expectedImproveRate * stepFrac
        r = improvement / expectedImprovement
        if r > acceptRatio and improvement > 0:
            return True, flat2Tensors(newX, xShapes)
    return False, flat2Tensors(x, xShapes)

class trpoWorkerPolicy(trpoPolicy):
    name = 'ppo v0 - worker'

    def initPolicy(self, **kwargs):
        setEmptyOpt(self)
        super().initPolicy(**kwargs)
