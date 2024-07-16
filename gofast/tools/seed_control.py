# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>
"""
The provided setup_module function sets up a fixed seed for random number 
generators (RNGs) in both numpy and the built-in random module based on an 
environment variable ``GOFAST_SEED``. If the environment variable is not set, 
it selects a random seed within the valid range for a 32-bit integer and uses 
it for seeding. This approach ensures reproducibility when the environment 
variable is set, but allows for variability otherwise.

However, if you want to ensure that a fixed seed is used every time the module 
is imported, without needing to manually call :func:`setup_module`, you can 
simply execute the seeding process at the module level, removing the need for 
the :func:`setup_module` function wrapper. Additionally, to ensure that 
the fixed seed is executed every time without depending on the environment 
variable, you can directly set the seed value in the code as::
    
    import numpy as np
    import random
    import os
    
    # Fixed seed value
    FIXED_SEED = 42  # Or any other preferred fixed seed value
    
    # Optionally, check for an environment variable to allow overriding
    _random_seed = os.environ.get("GOFAST_SEED", FIXED_SEED)
    
    print(f"I: Seeding RNGs with {_random_seed}")
    
    # Seed RNGs
    np.random.seed(int(_random_seed))
    random.seed(int(_random_seed))

"""
import os 

# Seed control function,
def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import numpy as np
    import random

    _random_seed = os.environ.get("GOFAST_SEED", np.random.randint(0, 2**32 - 1, dtype=np.int64))
    print(f"I: Seeding RNGs with {_random_seed}")
    np.random.seed(int(_random_seed))
    random.seed(int(_random_seed))

if __name__=='__main__': 
    setup_module()