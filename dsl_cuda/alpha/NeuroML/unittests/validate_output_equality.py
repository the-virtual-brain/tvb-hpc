import unittest
import filecmp


class TestEqualFiles(unittest.TestCase):
    def test_something(self):
        fp_cuda = '../models/CUDA/rateB_kuramoto.c'
        fp_golden = '../models/CUDA/kuramoto_network.c'
        self.assertTrue(filecmp.cmp(fp_golden, fp_cuda, shallow=False))


if __name__ == '__main__':
    unittest.main()
