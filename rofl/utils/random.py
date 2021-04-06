from rofl.functions.const import *

def seeder(seed:int):
    """
        Function to seed all the experiments.
        Some random generation such as the environment
        may work with their unique seeds.

        parameters
        ----------
        seed: int
            Positive integer
    """
    assert seed > 0, "seed must to be a positive number"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

