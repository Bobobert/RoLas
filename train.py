from rofl import setUpExperiment, retrieveConfigYaml
import argparse
import sys

def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse arguments for Train RoLas Experiments",
        epilog="python train.py -a dqn -c dqn_bulldozer")

    parser.add_argument(
        '--alg', '-a', type=str, required = True,
        help= 'Name of the algorithm to run')
    parser.add_argument(
        '--config', '-c', type=str, required= True,
        help = 'Name of the yaml config file on the scripts folder',
    )
    parser.add_argument(
        '--dm', action='store_false',
        help= 'Enable a data manager, otherwise will incur in not saving anyhing'
    )
    parser.add_argument(
        '--cuda', action='store_true',
        help= 'Enable trying to use any available CUDA devices'
    )

    return parser.parse_known_args(args)[0]

if __name__ == '__main__':
    args = argParser(sys.argv[:])
    configUser = retrieveConfigYaml(args.config)
    config, agent, policy, train, manager =\
        setUpExperiment(args.alg, configUser, dummyManager = args.dm, cuda = args.cuda)
    results = train(config, agent, policy, saver = manager.startSaver())
    agent.close()
    manager.close()
