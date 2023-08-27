import unittest
import numpy as np
from phonnx.col_preprocessors import ensure_min_2d, ColumnTypes, MAP


class TestPreprocessors(unittest.TestCase):
    def test_ensure_min_2d(self):
        X_0d = np.array(1)
        X_1d = np.array([1, 2, 3])
        X_2d = np.array([[1, 2], [3, 4]])
        self.assertEqual(ensure_min_2d(X_0d).shape, (1, 1))
        self.assertEqual(ensure_min_2d(X_1d).shape, (1, 3))
        self.assertEqual(ensure_min_2d(X_2d).shape, (2, 2))

    def test_MAP(self):
        for _, func in MAP.items():
            self.assertEqual(func, ensure_min_2d)


if __name__ == "__main__":
    unittest.main()
