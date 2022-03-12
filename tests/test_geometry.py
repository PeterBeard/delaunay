"""Unit tests for the geometry module."""
from __future__ import division
import os, sys
import unittest
import math
from random import randrange, seed


lib_path = os.path.abspath('..')
sys.path.append(lib_path)
from geometry import *

class TestPointTuple(unittest.TestCase):
    # Test correct number of values
    def test_two_values(self):
        p = Point(0, 1)
        self.assertEqual(p.x, 0)
        self.assertEqual(p.y, 1)

    # Test one value
    def test_one_value(self):
        self.assertRaises(TypeError, Point, (0))

    # Test three values
    def test_three_values(self):
        self.assertRaises(TypeError, Point, (0, 0, 0))

class TestLineSegmentTuple(unittest.TestCase):
    # Test correct number of values
    def test_two_values(self):
        l = LineSegment(0, 1)
        self.assertEqual(l.start, 0)
        self.assertEqual(l.end, 1)

    # Test one value
    def test_one_value(self):
        self.assertRaises(TypeError, LineSegment, (0))

    # Test three values
    def test_three_values(self):
        self.assertRaises(TypeError, LineSegment, (0, 0, 0))

class TestLineTuple(unittest.TestCase):
    # Test correct number of values
    def test_two_values(self):
        l = Line(0, 1)
        self.assertEqual(l.slope, 0)
        self.assertEqual(l.yintercept, 1)

    # Test one value
    def test_one_value(self):
        self.assertRaises(TypeError, Line, (0))

    # Test three values
    def test_three_values(self):
        self.assertRaises(TypeError, Line, (0, 0, 0))

class TestTriangleTuple(unittest.TestCase):
    # Test correct number of values
    def test_three_values(self):
        t = Triangle(0, 1, 2)

        self.assertEqual(t.a, 0)
        self.assertEqual(t.b, 1)
        self.assertEqual(t.c, 2)

    # Test two values
    def test_two_values(self):
        self.assertRaises(TypeError, Triangle, (0, 0))

    # Test four values
    def test_four_values(self):
        self.assertRaises(TypeError, Triangle, (0, 0, 0, 0))

class TestCircleTuple(unittest.TestCase):
    # Test correct number of values
    def test_two_values(self):
        c = Circle(0, 1)

        self.assertEqual(c.center, 0)
        self.assertEqual(c.radius_squared, 1)

    # Test one value
    def test_one_value(self):
        self.assertRaises(TypeError, Circle, (0))

    # Test three values
    def test_three_values(self):
        self.assertRaises(TypeError, Circle, (0, 0, 0))

class TestMidpoint(unittest.TestCase):
    # Find the midpoint of a zero-length line segment
    def test_midpoint_single(self):
        self.assertEqual(midpoint(LineSegment(Point(0, 0), Point(0, 0))), (0, 0))

    # Find the midpoint of a horizontal line segment on the x-axis
    def test_midpoint_horizontal(self):
        self.assertEqual(midpoint(LineSegment(Point(-1, 0), Point(1, 0))), (0, 0))

    # Find the midpoint of a vertical line segment on the y-axis
    def test_midpoint_vertical(self):
        self.assertEqual(midpoint(LineSegment(Point(0, -1), Point(0, 1))), (0, 0))

    # Find midpoint of a segment in the first quadrant
    def test_midpoint_q1(self):
        self.assertEqual(midpoint(LineSegment(Point(1, 1), Point(3, 3))), (2, 2))

    # Find midpoint of a segment in the second quadrant
    def test_midpoint_q2(self):
        self.assertEqual(midpoint(LineSegment(Point(-3, 3), Point(-1, 1))), (-2, 2))

    # Find the midpoint of a segment in the third quadrant
    def test_midpoint_q3(self):
        self.assertEqual(midpoint(LineSegment(Point(-3, -3), Point(-1, -1))), (-2, -2))

    # Find the midpoint of a segment in the fourth quadrant
    def test_midpoint_q4(self):
        self.assertEqual(midpoint(LineSegment(Point(3, -3), Point(1, -1))), (2, -2))

class TestSlope(unittest.TestCase):
    # Find the slope of a horizontal line segment
    def test_slope_horizontal(self):
        self.assertEqual(slope(LineSegment(Point(-1, 0), Point(1, 0))), 0)

    # Find the slope of a line with positive slope
    def test_slope_positive(self):
        self.assertEqual(slope(LineSegment(Point(-1, -1), Point(1, 1))), 1)
        # Reverse the coordinates
        self.assertEqual(slope(LineSegment(Point(1, 1), Point(-1, -1))), 1)

    # Find the slope of a line with negative slope
    def test_slope_negative(self):
        self.assertEqual(slope(LineSegment(Point(-1, 1), Point(1, -1))), -1)
        # Reverse the coordinates
        self.assertEqual(slope(LineSegment(Point(1, -1), Point(-1, 1))), -1)

    # Find the slope of a vertical line segment
    def test_slope_vertical(self):
        self.assertIsNone(slope(LineSegment(Point(0, -1), Point(0, 1))))

    # Find the slope of a single point
    def test_slope_point(self):
        self.assertRaises(ValueError, slope, LineSegment(Point(0, 0), Point(0, 0)))


class TestPerpSlope(unittest.TestCase):
    # Find the perpendicular slope of a vertical line segment
    def test_ps_vertical(self):
        self.assertEqual(perp_slope(LineSegment(Point(0, -1), Point(0, 1))), 0)

    # Find the perpendicular slope of a line with positive slope
    def test_ps_positive(self):
        self.assertEqual(perp_slope(LineSegment(Point(-1, -2), Point(1, 2))), -0.5)

    # Find the perpendicular slope of a line with negative slope
    def test_ps_negative(self):
        self.assertEqual(perp_slope(LineSegment(Point(-1, 2), Point(1, -2))), 0.5)

    # Find the perpendicular slope of a horizontal line segment
    def test_ps_horizontal(self):
        self.assertIsNone(perp_slope(LineSegment(Point(-1, 0), Point(1, 0))))

    # Find the perpendicular slope of a single point
    def test_ps_point(self):
        self.assertRaises(ValueError, perp_slope, LineSegment(Point(0, 0), Point(0, 0)))


class TestPStoYIntercept(unittest.TestCase):
    # Test positive slope with y-intercept at origin
    def test_positive_origin(self):
        self.assertEqual(point_slope_to_y_intercept(1, Point(2, 2)), Line(1, 0))

    # Test negative slope w/y-int at origin
    def test_negative_origin(self):
        self.assertEqual(point_slope_to_y_intercept(-1, Point(2, -2)), Line(-1, 0))

    # Test positive slope w/y-int above 0
    def test_positive_top(self):
        self.assertEqual(point_slope_to_y_intercept(1, Point(2, 3)), Line(1, 1))

    # Test positive slope w/y-int below 0
    def test_positive_bottom(self):
        self.assertEqual(point_slope_to_y_intercept(1, Point(2, 1)), Line(1, -1))

    # Test negative slope w/y-int above 0
    def test_negative_top(self):
        self.assertEqual(point_slope_to_y_intercept(-1, Point(2, -1)), Line(-1, 1))

    # Test negative slope w/y-int below 0
    def test_negative_bottom(self):
        self.assertEqual(point_slope_to_y_intercept(-1, Point(2, -3)), Line(-1, -1))

    # Test horizontal segment w/y-int at origin
    def test_horizontal_origin(self):
        self.assertEqual(point_slope_to_y_intercept(0, Point(2, 0)), Line(0, 0))

    # Test horizontal segment w/y-int above 0
    def test_horizontal_top(self):
        self.assertEqual(point_slope_to_y_intercept(0, Point(2, 1)), Line(0, 1))

    # Test horizontal segment w/y-int below 0
    def test_horizontal_bottom(self):
        self.assertEqual(point_slope_to_y_intercept(0, Point(2, -1)), Line(0, -1))

class TestIsVertical(unittest.TestCase):
    # Test vertical line segment
    def test_vertical(self):
        self.assertTrue(is_vertical(LineSegment(Point(0, -1), Point(0, 1))))

    # Test horizontal line segment
    def test_horizontal(self):
        self.assertFalse(is_vertical(LineSegment(Point(-1, 0), Point(1, 0))))

class TestIsHorizontal(unittest.TestCase):
    # Test vertical line segment
    def test_vertical(self):
        self.assertFalse(is_horizontal(LineSegment(Point(0, -1), Point(0, 1))))

    # Test horizontal line segment
    def test_horizontal(self):
        self.assertTrue(is_horizontal(LineSegment(Point(-1, 0), Point(1, 0))))


class TestIsCollinear(unittest.TestCase):
    # Test non-collinear points
    def test_non_collinear(self):
        self.assertFalse(is_collinear(Point(0,0), Point(0,1), Point(1,1)))

    # Test collinear points
    def test_collinear(self):
        self.assertTrue(is_collinear(Point(0,0), Point(1,1), Point(2,2)))

class TestLinesIntersection(unittest.TestCase):
    # Test parallel horizontal lines
    def test_par_horizontal(self):
        self.assertIsNone(lines_intersection(Line(0, -1), Line(0, 1)))
    
    # Test parallel lines w/positive slope
    def test_par_positive(self):
        self.assertIsNone(lines_intersection(Line(1, -1), Line(1, 1)))

    # Test parallel lines w/negative slope
    def test_par_negative(self):
        self.assertIsNone(lines_intersection(Line(-1, -1), Line(-1, 1)))

    # Test perpendicular lines intersecting at origin
    def test_perp_origin(self):
        self.assertEqual(lines_intersection(Line(1, 0), Line(-1, 0)), Point(0, 0))

    # Test perp. lines intersecting in Q1
    def test_perp_q1(self):
        self.assertEqual(lines_intersection(Line(1, 0), Line(-1, 2)), Point(1, 1))

    # Test perp. lines intersecting in Q2
    def test_perp_q2(self):
        self.assertEqual(lines_intersection(Line(1, 2), Line(-1, 0)), Point(-1, 1))

    # Test perp. lines intersecting in Q3
    def test_perp_q3(self):
        self.assertEqual(lines_intersection(Line(1, 0), Line(-1, -2)), Point(-1, -1))

    # Test perp. lines intersecting in Q4
    def test_perp_q4(self):
        self.assertEqual(lines_intersection(Line(1, -2), Line(-1, 0)), Point(1, -1))

class TestLineIntersectVertical(unittest.TestCase):
    # Test intersection at origin
    def test_horizontal_origin_origin(self):
        self.assertEqual(line_intersect_vertical(Line(0, 0), Point(0, 1)), Point(0, 0))

    # Test intersection with horizontal line (y=0) at x>0
    def test_horizontal_origin_right(self):
        self.assertEqual(line_intersect_vertical(Line(0, 0), Point(1, 1)), Point(1, 0))

    # Test intersection with horizontal line (y=0) at x<0
    def test_horizontal_origin_left(self):
        self.assertEqual(line_intersect_vertical(Line(0, 0), Point(-1, -1)), Point(-1, 0))

    # Test intersection with horizontal line (y>0) at x=0
    def test_horizontal_top_origin(self):
        self.assertEqual(line_intersect_vertical(Line(0, 1), Point(0, 0)), Point(0, 1))

    # Test intersection with horizontal line (y>0) at x>0
    def test_horizontal_top_right(self):
        self.assertEqual(line_intersect_vertical(Line(0, 1), Point(1, 1)), Point(1, 1))

    # Test intersection with horizontal line (y>0) at x<0
    def test_horizontal_top_left(self):
        self.assertEqual(line_intersect_vertical(Line(0, 1), Point(-1, -1)), Point(-1, 1))

    # Test intersection with horizontal line (y<0) at x=0
    def test_horizontal_bottom_origin(self):
        self.assertEqual(line_intersect_vertical(Line(0, -1), Point(0, 0)), Point(0, -1))

    # Test intersection with horizontal line (y<0) at x>0
    def test_horizontal_bottom_right(self):
        self.assertEqual(line_intersect_vertical(Line(0, -1), Point(1, 0)), Point(1, -1))

    # Test intersection with horizontal line (y<0) at x<0
    def test_horizontal_bottom_left(self):
        self.assertEqual(line_intersect_vertical(Line(0, -1), Point(-1, 0)), Point(-1, -1))

    # Test intersection with positive slope at x=0
    def test_positive_origin(self):
        self.assertEqual(line_intersect_vertical(Line(1, 0), Point(0, 0)), Point(0, 0))

    # Test intersection with positive slope in Q1
    def test_positive_q1(self):
        self.assertEqual(line_intersect_vertical(Line(1, 0), Point(1, 0)), Point(1, 1))

    # Test intersection with positive slope in Q2
    def test_positive_q2(self):
        self.assertEqual(line_intersect_vertical(Line(1, 2), Point(-1, 0)), Point(-1, 1))

    # Test intersection with positive slope in Q3
    def test_positive_q3(self):
        self.assertEqual(line_intersect_vertical(Line(1, 0), Point(-1, 0)), Point(-1, -1))

    # Test intersection with positive slope in Q4
    def test_positive_q4(self):
        self.assertEqual(line_intersect_vertical(Line(1, -2), Point(1, 0)), Point(1, -1))

    # Test intersection with negative slope at origin
    def test_negative_origin(self):
        self.assertEqual(line_intersect_vertical(Line(-1, 0), Point(0, 0)), Point(0, 0))

    # Test intersection with negative slope in Q1
    def test_negative_q1(self):
        self.assertEqual(line_intersect_vertical(Line(-1, 2), Point(1, 0)), Point(1, 1))

    # Test intersection with negative slope in Q2
    def test_negative_q2(self):
        self.assertEqual(line_intersect_vertical(Line(-1, 0), Point(-1, 0)), Point(-1, 1))

    # Test intersection with negative slope in Q3
    def test_negative_q3(self):
        self.assertEqual(line_intersect_vertical(Line(-1, -2), Point(-1, 0)), Point(-1, -1))

    # Test intersection with negative slope in Q4
    def test_negative_q4(self):
        self.assertEqual(line_intersect_vertical(Line(-1, 0), Point(1, 0)), Point(1, -1))

class TestCompareTris(unittest.TestCase):
    # Compare two identical triangles
    def test_same_tri(self):
        linea = LineSegment(Point(0, 0), Point(0, 4))
        lineb = LineSegment(Point(0, 4), Point(3, 0))
        linec = LineSegment(Point(3, 0), Point(0, 0))

        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(linea, lineb, linec)))
        
    # Compare identical triangles with vertices rearranged
    def test_rearranged_tri(self):
        linea = LineSegment(Point(0, 0), Point(0, 4))
        lineb = LineSegment(Point(0, 4), Point(3, 0))
        linec = LineSegment(Point(3, 0), Point(0, 0))

        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(linea, linec, lineb)))
        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(lineb, linea, linec)))
        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(lineb, linec, linea)))
        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(linec, linea, lineb)))
        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(linec, lineb, linea)))

    def test_reversed_edges(self):
        vert1 = Point(0, 0)
        vert2 = Point(0, 1)
        vert3 = Point(1, 0)
        linea = LineSegment(vert1, vert2)
        lineb = LineSegment(vert2, vert3)
        linec = LineSegment(vert3, vert1)

        linea_rev = LineSegment(vert2, vert1)
        lineb_rev = LineSegment(vert3, vert2)
        linec_rev = LineSegment(vert1, vert3)

        self.assertTrue(compare_tris(Triangle(linea, lineb, linec), Triangle(linea_rev, lineb_rev, linec_rev)))

    def test_different_tris(self):
        tri_a = Triangle(Point(0, 1), Point(1, 2), Point(0, 2))
        tri_b = Triangle(Point(1, 3), Point(3, 1), Point(0, 0))

        self.assertFalse(compare_tris(tri_a, tri_b))

class TestCalculateTriVertices(unittest.TestCase):
    # Try to make a triangle with parallel sides
    def test_parallel_sides(self):
        linea = LineSegment(Point(0, 0), Point(0, 5))
        lineb = LineSegment(Point(1, 0), Point(1, 5))
        linec = LineSegment(Point(0, 0), Point(1, 0))

        self.assertIsNone(calculate_tri_vertices(linea, lineb, linec))
        self.assertIsNone(calculate_tri_vertices(linea, linec, lineb))
        self.assertIsNone(calculate_tri_vertices(lineb, linec, linea))
        self.assertIsNone(calculate_tri_vertices(lineb, linea, linec))
        self.assertIsNone(calculate_tri_vertices(linec, linea, lineb))
        self.assertIsNone(calculate_tri_vertices(linec, lineb, linea))

    # Try a 3-4-5 triangle in quadrant I
    def test_345_q1(self):
        linea = LineSegment(Point(0, 0), Point(0, 4))
        lineb = LineSegment(Point(0, 4), Point(3, 0))
        linec = LineSegment(Point(3, 0), Point(0, 0))

        ref_triangle = Triangle(Point(0, 4), Point(3, 0), Point(0, 0))

        self.assertTrue(compare_tris(calculate_tri_vertices(linea, lineb, linec), ref_triangle))
        self.assertTrue(compare_tris(calculate_tri_vertices(linea, linec, lineb), ref_triangle))
        self.assertTrue(compare_tris(calculate_tri_vertices(lineb, linea, linec), ref_triangle))
        self.assertTrue(compare_tris(calculate_tri_vertices(lineb, linec, linea), ref_triangle))
        self.assertTrue(compare_tris(calculate_tri_vertices(linec, linea, lineb), ref_triangle))
        self.assertTrue(compare_tris(calculate_tri_vertices(linec, lineb, linea), ref_triangle))

    # Try a 3-4-5 triangle in quadrant II
    def test_345_q2(self):
        linea = LineSegment(Point(0, 0), Point(0, 4))
        lineb = LineSegment(Point(0, 4), Point(-3, 0))
        linec = LineSegment(Point(-3, 0), Point(0, 0))

        ref_triangle = Triangle(Point(0, 4), Point(-3, 0), Point(0, 0))

        self.assertEqual(calculate_tri_vertices(linea, lineb, linec), ref_triangle)

    # Try a 3-4-5 triangle in quadrant III
    def test_345_q3(self):
        linea = LineSegment(Point(0, 0), Point(0, -4))
        lineb = LineSegment(Point(0, -4), Point(-3, 0))
        linec = LineSegment(Point(-3, 0), Point(0, 0))

        ref_triangle = Triangle(Point(0, -4), Point(-3, 0), Point(0, 0))

        self.assertEqual(calculate_tri_vertices(linea, lineb, linec), ref_triangle)

    # Try a 3-4-5 triangle in quadrant IV
    def test_345_q4(self):
        linea = LineSegment(Point(0, 0), Point(0, -4))
        lineb = LineSegment(Point(0, -4), Point(3, 0))
        linec = LineSegment(Point(3, 0), Point(0, 0))

        ref_triangle = Triangle(Point(0, -4), Point(3, 0), Point(0, 0))

        self.assertEqual(calculate_tri_vertices(linea, lineb, linec), ref_triangle)


    # Try to make a triangle out of points that lie on the same line
    def test_line(self):
        linea = LineSegment(Point(0, 0), Point(0, 1))
        lineb = LineSegment(Point(0, 1), Point(0, 2))
        linec = LineSegment(Point(0, 2), Point(0, 3))

        self.assertIsNone(calculate_tri_vertices(linea, lineb, linec))

class TestTriFromEdgePoint(unittest.TestCase):
    # Try a 3-4-5 triangle
    def test_345(self):
        line = LineSegment(Point(0, 0), Point(4, 0))
        point = Point(0, 3)

        ref_triangle = Triangle(Point(0, 0), Point(4, 0), Point(0, 3))
        self.assertEqual(triangle_from_edge_point(line, point), ref_triangle)

class TestVerticesToEdges(unittest.TestCase):
    # Try a 3-4-5 triangle
    def test_345(self):
        edge_triangle = Triangle(
            LineSegment(Point(0, 0), Point(0, 4)),
            LineSegment(Point(0, 4), Point(3, 0)),
            LineSegment(Point(3, 0), Point(0, 0))
        )
        vertex_triangle = Triangle(
            Point(0, 4),
            Point(3, 0),
            Point(0, 0)
        )
        self.assertEqual(vertices_to_edges(vertex_triangle), edge_triangle)

    # Try too few vertices
    def test_too_few_vertices(self):
        self.assertIsNone(vertices_to_edges(((0, 0), (0, 1))))

    # Try too many vertices
    def test_too_many_vertices(self):
        self.assertIsNone(vertices_to_edges(((0, 0), (0, 1), (0, 2), (0, 3))))

class TestEdgesToVertices(unittest.TestCase):
    # Try a 3-4-5 triangle
    def test_345(self):
        edge_triangle = Triangle(
            LineSegment(Point(0, 0), Point(0, 4)),
            LineSegment(Point(0, 4), Point(3, 0)),
            LineSegment(Point(3, 0), Point(0, 0))
        )
        vertex_triangle = Triangle(
            Point(0, 4),
            Point(3, 0),
            Point(0, 0)
        )
        self.assertEqual(edges_to_vertices(edge_triangle), vertex_triangle)

    # Try too few edges
    def test_too_few_edges(self):
        self.assertIsNone(edges_to_vertices((((0, 0), (0, 1)), ((0, 2), (2, 0)))))

    # Try too many edges
    def test_too_many_edges(self):
        self.assertIsNone(edges_to_vertices((((0, 0), (0, 1)), ((0, 2), (2, 0)), ((2, 0), (1, 1)), ((1, 1), (2, 1)))))

class TestEdgeEquivalence(unittest.TestCase):
    # Make sure we get the original triangle back after applying edges_to_vertices and vertices_to_edges
    def test_idempotence(self):
        tri = Triangle(
            LineSegment(Point(0, 0), Point(0, 4)),
            LineSegment(Point(0, 4), Point(3, 0)),
            LineSegment(Point(3, 0), Point(0, 0))
        )
        tri_verts = edges_to_vertices(tri)
        self.assertEqual(vertices_to_edges(tri_verts), tri)

class TestTriContainsPoint(unittest.TestCase):
    # Try a vertex
    def test_vertex(self):
        tri = Triangle(Point(0, 0), Point(0, 4), Point(3, 0))

        self.assertTrue(tri_contains_point(tri, Point(0, 0)))
        self.assertTrue(tri_contains_point(tri, Point(0, 4)))
        self.assertTrue(tri_contains_point(tri, Point(3, 0)))

    # Try a point in the middle of the triangle
    def test_point_center(self):
        tri = Triangle(Point(0, 0), Point(0, 4), Point(3, 0))
        point = Point(1, 1.33)

        self.assertTrue(tri_contains_point(tri, point))

    # Try a point on the edge of the triangle
    def test_point_edge(self):
        tri = Triangle(Point(0, 0), Point(0, 4), Point(3, 0))
        point = Point(0, 2)

        self.assertTrue(tri_contains_point(tri, point))

    # Try a point outside the triangle
    def test_point_outside(self):
        tri = Triangle(Point(0, 0), Point(0, 4), Point(3, 0))
        point = Point(-1, -1)

        self.assertFalse(tri_contains_point(tri, point))


class TestTriCircumcenter(unittest.TestCase):
    # Test circumcenter at origin
    def test_circumcenter_at_origin(self):
        root3 = sqrt(3)
        vert1 = Point(-1, -1*root3/3)
        vert2 = Point(1, -1*root3/3)
        vert3 = Point(0, 2*root3/3)

        center123 = tri_circumcenter(Triangle(vert1, vert2, vert3))
        self.assertAlmostEqual(center123.x, 0)
        self.assertAlmostEqual(center123.y, 0)

        center132 = tri_circumcenter(Triangle(vert1, vert3, vert2))
        self.assertAlmostEqual(center132.x, 0)
        self.assertAlmostEqual(center132.y, 0)

        center213 = tri_circumcenter(Triangle(vert2, vert3, vert1))
        self.assertAlmostEqual(center213.x, 0)
        self.assertAlmostEqual(center213.y, 0)

        center231 = tri_circumcenter(Triangle(vert2, vert1, vert3))
        self.assertAlmostEqual(center231.x, 0)
        self.assertAlmostEqual(center231.y, 0)

        center321 = tri_circumcenter(Triangle(vert3, vert2, vert1))
        self.assertAlmostEqual(center321.x, 0)
        self.assertAlmostEqual(center321.y, 0)

        center312 = tri_circumcenter(Triangle(vert3, vert1, vert2))
        self.assertAlmostEqual(center312.x, 0)
        self.assertAlmostEqual(center312.y, 0)

    def test_collinear_points(self):
        tri = Triangle(Point(0, 0), Point(1, 0), Point(2, 0))
        circle = tri_circumcircle(tri)
        self.assertEqual(circle.center, Point(1, 0))
        self.assertEqual(circle.radius_squared, 1.0)


class TestTriCentroid(unittest.TestCase):
    # Test centroid at origin
    def test_centroid_at_origin(self):
        self.centroid_at_origin = Triangle(
            Point(-1, -1),
            Point(1, -1),
            Point(0, 2)
        )
        self.assertEqual(tri_centroid(self.centroid_at_origin), Point(0, 0))

class TestTriCircumcircle(unittest.TestCase):
    # Test equilateral triangle with center at the origin and radius ~ 1
    def test_circumcircle_at_origin(self):
        root3 = sqrt(3)
        vert1 = Point(-1, -1*root3/3)
        vert2 = Point(1, -1*root3/3)
        vert3 = Point(0, 2*root3/3)

        ref_circle = Circle(Point(0, 0), 4.0/3.0)

        circle = tri_circumcircle(Triangle(vert1, vert2, vert3))

        self.assertAlmostEqual(ref_circle.center.x, circle.center.x)
        self.assertAlmostEqual(ref_circle.center.y, circle.center.y)
        self.assertAlmostEqual(ref_circle.radius_squared, circle.radius_squared)


class TestTriShareVertices(unittest.TestCase):
    # Set up test data
    def setUp(self):
        self.origin_345 = Triangle(
            Point(0, 0),
            Point(4, 0),
            Point(0, 3)
        )

        self.origin_345_alt = Triangle(
            Point(4, 0),
            Point(0, 0),
            Point(0, 3)
        )

        self.onevert_345 = Triangle(
            Point(0, 0),
            Point(-4, 0),
            Point(0, -3)
        )

        self.twovert_345 = Triangle(
            Point(0, 0),
            Point( -4, 0),
            Point(0, 3)
        )

        self.distant_345 = Triangle(
            Point(10, 10),
            Point(14, 10),
            Point(10, 13)
        )


    # Test triangles with no common vertices
    def test_no_common_verts(self):
        self.assertFalse(tri_share_vertices(self.origin_345, self.distant_345))


    # Test triangles with 1 common vertex
    def test_one_common_vert(self):
        self.assertTrue(tri_share_vertices(self.origin_345, self.onevert_345))


    # Test triangle with 2 common vertices
    def test_two_common_verts(self):
        self.assertTrue(tri_share_vertices(self.origin_345, self.twovert_345))


    # Test identical triangles
    def test_identical_tris(self):
        self.assertTrue(tri_share_vertices(self.origin_345, self.origin_345))


    # Test triangles with the same vertices but rearranged
    def test_same_verts(self):
        self.assertTrue(tri_share_vertices(self.origin_345, self.origin_345_alt))


class TestAngle(unittest.TestCase):
    # Vectors are collinear
    def test_collinear(self):
        self.assertEqual(angle(Point(2, 1), Point(3, 2)), math.pi/4)
        self.assertEqual(angle(Point(3, 2), Point(2, 1)), 5*math.pi/4)

    # Vectors are on a horizontal line
    def test_horizontal(self):
        self.assertEqual(angle(Point(1, 1), Point(2, 1)), 0)
        self.assertEqual(angle(Point(2, 1), Point(1, 1)), math.pi)

    # Vectors are perpendicular in q1
    def test_qI(self):
        self.assertEqual(angle(Point(0, 0), Point(1, 0)), 0)

    # Vectors are perpendicular in q2
    def test_qII(self):
        self.assertEqual(angle(Point(0, 0), Point(0, 1)), math.pi/2)

    # Vectors are perpendicular in q3
    def test_qIII(self):
        self.assertEqual(angle(Point(0, 0), Point(-1, 0)), math.pi)

    # Vectors are perpendicular in q4
    def test_qIV(self):
        self.assertEqual(angle(Point(0, 0), Point(0, -1)), 3*math.pi/2)


class TestTurnDirection(unittest.TestCase):
    # Test three of the same value
    def test_three_same(self):
        for v in range(-100,100):
            p = Point(v,v)
            self.assertEqual(turn_direction(p,p,p), 0)

    # Test collinear points
    def test_collinear(self):
        for v in range(-100,100):
            self.assertEqual(turn_direction(Point(v,v), Point(v+1,v+1), Point(2*v,2*v)), 0)

    # Test clockwise turn
    def test_clockwise(self):
        self.assertGreater(turn_direction(Point(-1, 0), Point(0, 0), Point(0, -1)), 0)

    # Test counter-clockwise turn
    def test_counterclockwise(self):
        self.assertLess(turn_direction(Point(-1, 0), Point(0, 0), Point(0, 1)), 0)


class TestTranslateTri(unittest.TestCase):
    # Create reference data
    def setUp(self):
        self.base_tri = Triangle(
            Point(0, 0),
            Point(0, 4),
            Point(3, 0)
        )

        self.left_1 = Triangle(
            Point(-1, 0),
            Point(-1, 4),
            Point(2, 0)
        )

        self.right_1 = Triangle(
            Point(1, 0),
            Point(1, 4),
            Point(4, 0)
        )

        self.up_1 = Triangle(
            Point(0, 1),
            Point(0, 5),
            Point(3, 1)
        )

        self.down_1 = Triangle(
            Point(0, -1),
            Point(0, 3),
            Point(3, -1)
        )


    # Test zero-translation
    def test_zero(self):
        self.assertEqual(translate_tri(self.base_tri, Vector(0, 0)), self.base_tri)

    # Test negative x
    def test_move_left(self):
        self.assertEqual(translate_tri(self.base_tri, Vector(-1, 0)), self.left_1)

    # Test positive x
    def test_move_right(self):
        self.assertEqual(translate_tri(self.base_tri, Vector(1, 0)), self.right_1)

    # Test negative y
    def test_move_down(self):
        self.assertEqual(translate_tri(self.base_tri, Vector(0, -1)), self.down_1)

    # Test positive y
    def test_move_up(self):
        self.assertEqual(translate_tri(self.base_tri, Vector(0, 1)), self.up_1)

class TestScaleTri(unittest.TestCase):
    def setUp(self):
        self.tri = Triangle(
            Point(-3/2, -1),
            Point(3/2, -1),
            Point(0, 2)
        )

        self.tri_3x = Triangle(
            Point(-9/2, -3),
            Point(9/2, -3),
            Point(0, 6)
        )

        self.tri_third = Triangle(
            Point(-1/2, -1/3),
            Point(1/2, -1/3),
            Point(0, 2/3)
        )

    def test_scale_3x(self):
        self.assertEqual(scale_tri(self.tri, 3), self.tri_3x)

    def test_scale_third(self):
        self.assertEqual(scale_tri(self.tri, 1/3), self.tri_third)

class TestConvexHull(unittest.TestCase):
    def setUp(self):
        # Generate some random points
        COUNT = 1000
        MIN = -500
        MAX = 500

        seed(0)

        self.points = list(map(lambda _: Point(randrange(MIN, MAX), randrange(MIN, MAX)), range(COUNT)))
        self.hull3 = sorted(self.points[0:3])

    def test_2_points(self):
        hull = convex_hull(self.points[0:2])
        self.assertIsNone(hull)

    def test_3_points(self):
        hull = sorted(convex_hull(self.points[0:3]))
        self.assertEqual(hull, self.hull3)

    def test_all_points(self):
        # Use the winding number test to determine whether all points lie
        # within the convex hull.
        # (Sunday, 2001: http://geomalgorithms.com/a03-_inclusion.html)
        def is_left(p0, p1, p2):
            return ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y))

        hull = convex_hull(self.points)

        all_points_contained = True
        for p in self.points:
            wn = 0
            if p not in hull:
                for i in range(-1,len(hull)-1):
                    v = hull[i]
                    vnext = hull[i+1]
                    if v.y <= p.y:
                        if vnext.y > p.y:
                            if is_left(v, vnext, p) > 0:
                                wn += 1
                    else:
                        if vnext.y <= p.y:
                            if is_left(v, vnext, p) < 0:
                                wn -= 1
                # Winding number != 0 means p is inside the hull
                if wn == 0:
                    all_points_contained = False
                self.assertNotEqual(wn, 0)
        self.assertTrue(all_points_contained)

class TestEnclosingTriangle(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
