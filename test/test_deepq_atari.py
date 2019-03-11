"""Test Deep Q Learning on Atari game."""

import unittest
import nas_rl2.rl2.rl.deep_q_atari as dql_atari


class TestDQLAtari(unittest.TestCase):
    """Test Deep Q Learning for atari."""

    def test_atari_breakdown(self):
        # dql_atari.solve_atari()
