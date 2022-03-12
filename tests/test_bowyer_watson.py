import unittest
from algorithms.bowyer_watson import triangulate
from geometry import Point, Triangle

class TestDelaunayTriangulation(unittest.TestCase):
    def test_3_points(self):
        points = [
            Point(0, 1),
            Point(1, 1),
            Point(0, 0)
        ]
        triangle = Triangle(
            *points
        )
        d_tri = triangulate(points)
        self.assertEqual(d_tri[0], triangle)

    def test_square_points(self):
        points = [
            Point(0, 0),
            Point(0, 1),
            Point(1, 1),
            Point(1, 0),
        ]
        d_tri = triangulate(points)
        self.assertEqual(len(d_tri), 2)

if __name__ == '__main__':
    unittest.main()
