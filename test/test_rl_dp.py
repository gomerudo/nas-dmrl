"""Test Dynamic Programming approaches for Reinforcement Learning."""

import unittest
import numpy as np
import nas_rl2.rl2.rl.policy_value_iteration as pvi


class TestRLDP(unittest.TestCase):
    """Test Dynamic Programming approaches."""

    def test_policy_iteration(self):
        final_policy, final_v = pvi.policy_iteration()

        print("Final Policy ")
        print(final_policy)

        print("Final Policy grid : (0=up, 1=right, 2=down, 3=left)")
        print(np.reshape(np.argmax(final_policy, axis=1), pvi.GW_ENV.shape))

        print("Final Value Function grid")
        print(final_v.reshape(pvi.GW_ENV.shape))

    def test_value_iteration(self):
        final_policy, final_v = pvi.value_iteration()

        print("Final Policy ")
        print(final_policy)

        print("Final Policy grid : (0=up, 1=right, 2=down, 3=left)")
        print(np.reshape(np.argmax(final_policy, axis=1), pvi.GW_ENV.shape))

        print("Final Value Function grid")
        print(final_v.reshape(pvi.GW_ENV.shape))
