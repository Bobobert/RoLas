from rofl.utils.utils import timeToStop, time

def initResultDict():
    return {"mean_return": [],
            "mean_steps": [],
            "std_return": [],
            "std_steps": [],
            "custom": [],
            "time_start": time.time(),
            "time_elapsed": 0.0,
            "time_execution": [],
            "max_return": [],
            "min_return": [],
            }

def testEvaluation(config, agent, trainResults = None):
    """
        From an agent evaluates the test method, saves
        the results into the results dict, and if it is
        enough to stop the train algorithm.

        returns
        -------
        results: dict
            From the agent.test()
        trainResults: dict
            From the training loop, it expands it
            with the latest results from the agent.
        stop: boolean
            From the config dict, if available evaluates
            if the train should stop or not.
    """
    results = agent.test(iters = config["train"]["iters_test"])
    if trainResults is None:
        trainResults = initResultDict()
    # Appending results
    for key in results.keys():
        trainResults[key] += [results[key]]
    # Evaluating time
    trainResults, stopTime = timeToStop(trainResults, config["train"].get("max_time", None))
    # Evaluating performance
    ## Expected
    stopPerE = False
    expectedPerformance = config["train"].get("expected_performance", None)
    if expectedPerformance is not None:
        c = -1.0 # Here if some confidence more than the one deviation
        expPer = results["mean_return"] + c * results["std_return"]
        stopPerE = True if expPer >= expectedPerformance else False
    ## Maxc
    stopPerM = False
    maxPerformance = config["train"].get("max_performance", None)
    if maxPerformance is not None:
        stopPerM = True if results["max_return"] >= maxPerformance else False
    return results, trainResults, (stopTime or stopPerE or stopPerM)