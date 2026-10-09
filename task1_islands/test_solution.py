import unittest
from task1_islands.solution import count_islands


class IslandsTests(unittest.TestCase):
    def test_examples_and_edges(self):
        for grid, expected in [([[0,1,0],[0,0,0],[0,1,1]],2),
                               ([[0,0,0,1],[0,0,1,0],[0,1,0,0]],3),
                               ([[0,0,0,1],[0,0,1,1],[0,1,0,1]],2),
                               ([],0), ([[]],0), ([[0]],0), ([[1]],1),
                               ([[1,0],[0,1]],2), ([[1]*2000],1)]:
            with self.subTest(grid=grid[:1]):
                original = [row[:] for row in grid]
                self.assertEqual(count_islands(grid), expected)
                self.assertEqual(grid, original)

    def test_invalid(self):
        for grid in ([[1],[1,0]], [[2]]):
            with self.assertRaises(ValueError):
                count_islands(grid)

if __name__ == "__main__":
    unittest.main()
