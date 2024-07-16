# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>
"""
conftest.py 

By placing this fixture in a conftest.py file at the root of your tests 
directory, pytest will automatically apply this fixture to every test in 
the session, thanks to the autouse=True parameter. This way, you don't need 
to modify each test file to ensure consistent seeding. The scope='session' 
parameter ensures the seed is set once per test session, rather than before 
each individual test, which is usually sufficient unless your tests modify 
the global RNG state in a way that requires re-seeding before each test.

This pytest fixture approach is more aligned with typical testing practices 
and integrates seamlessly with the pytest test lifecycle.
"""

import pytest
import numpy as np
import random
import os

@pytest.fixture(scope='session', autouse=True)
def global_rng_seed():
    """Fixture to set a globally controllable seed for all tests in the session."""
    _random_seed = os.environ.get("GOFAST_SEED", np.random.randint(0, 2**32 - 1, dtype=np.int64))
    print(f"I: Seeding RNGs for all tests with {_random_seed}")
    np.random.seed(int(_random_seed))
    random.seed(int(_random_seed))

# No need to modify individual test files
