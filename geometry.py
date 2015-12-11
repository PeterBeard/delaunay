"""
Geometry functions for finding the Delaunay triangulation of a set of points.

This module exports the following:
Types:
    Point: point in 2-space
    Vector: 2-d vector
    LineSegment: line segment by its start and end points
    Line: line defined by its slope and y-intercept
    Triangle: triangle defined by its three edges or vertices
    Circle: circle defined by its center point and radius

Functions:
    is_valid_segment(line): check a line segment for validity
    midpoint(line): find the midpoint of a line segment
    slope(line): find the slope of a line segment
    perp_slope(line): find the perpendicular slope of a line segment
    point_slope_to_y_intercept(m, p): convert a line segment from point-slope to y-intercept form
    is_vertical(line): determine whether a line segment is vertical
    is_horizontal(line): determine whether a line segment is horizontal
    lines_intersection(a, b): find the point where two lines intersect
    line_intersect_vertical(line, point): find the y-coordinate where a line intersects a vertical line
    compare_tris(a, b): determine whether two triangles are equivalent
    calculate_tri_vertices(a, b, c): calculate the vertices of a triangle defined by the given line segments
    triangle_from_edge_point(e, p): calculate a triangle from the given line segment and point
    vertices_to_edges(tri): convert a triangle from vertex form to edge form
    edges_to_vertices(tri): convert a triangle from edge form to vertex form
    tri_contains_point(tri, point): determine whether a triangle contains a point
    tri_circumcenter(tri): find the circumcenter of a triangle
    tri_centroid(tri): find the centroid of a triangle
    tri_circumcircle(tri): find the circumcircle of a triangle
    tri_share_vertices(a, b): determine whether two triangles share any vertices
    angle(a, b): find the angle between two points
    cross_product(a, b): find the cross product of three points
    translate_tri(tri, vector): translate a triangle by the given vector
    scale_tri(t, s): scale a triangle by the given scale factor
    convex_hull(p): find the convex hull of a set of points
    enclosing_triangle(p): find a triangle enclosing a set of points
    delaunay_triangulation(p): find the Delaunay triangulation of a set of points
"""
from __future__ import division
from collections import namedtuple
from math import sqrt, atan2
import numbers

# A point has an x and a y coordinate
Point = namedtuple('Point', 'x y')
# A vector also has an x and a y coordinate
Vector = namedtuple('Vector', 'x y')
# A line segment consists of two points, a and b
LineSegment = namedtuple('LineSegment', 'start end')
# A line is defined by its slope and its y-intercept
Line = namedtuple('Line', 'slope yintercept')
# A triangle can be defined by either three edges or points
Triangle = namedtuple('Triangle', 'a b c')
# A circle is defined by its radius and center
Circle = namedtuple('Circle', 'center radius')


def is_valid_segment(line):
    """
    Determine whether line is a valid line segment.

    Arguments:
    line is a 2-tuple of x,y coordinates, e.g. ((x1, y1), (x2, y2))

    Returns:
    True if the segment is valid and False otherwise.
    """
    return (
        isinstance(line, LineSegment) or
        isinstance(line, tuple) or
        isinstance(line, list)
    ) and\
        len(line) == 2 and\
        (
            isinstance(line[0], tuple) or
            isinstance(line[0], list) or
            isinstance(line[0], Point)
        ) and\
        len(line[0]) == 2 and\
        (
            isinstance(line[0], tuple) or
            isinstance(line[0], list) or
            isinstance(line[0], Point)
        ) and\
        len(line[1]) == 2


def midpoint(line):
    """
    Find the midpoint of a line segment.

    Arguments:
    line is a 2-tuple of x,y coordinates, e.g. ((x1,y1),(x2,y2))

    Returns:
    A Point object representing the midpoint of the line.
    """
    return Point((line.start.x + line.end.x)/2, (line.start.y + line.end.y)/2)


def slope(line):
    """
    Find the slope of a line segment. May raise ValueError.

    Arguments:
    line is a 2-tuple of x,y coordinates, e.g. ((x1,y1),(x2,y2))

    Returns:
    The slope of the line (float)
    """
    try:
        return (line.end.y - line.start.y)/(line.end.x - line.start.x)
    # Catch exceptions and raise more helpful ones if possible
    except ZeroDivisionError:
        # Raise an error if both points are the same
        if line.start == line.end:
            raise ValueError('Both points are the same')
        # Raise an error if dx = 0
        if line.start.x == line.end.x:
            raise ValueError('Line has infinite slope')
    except AttributeError:
        # Raise an error if the input isn't a line segment
        if not is_valid_segment(line):
            raise ValueError('Input is not a line segment ((x1, y1), (x2, y2))')


def perp_slope(line):
    """
    Find the slope of a line perpendicular to a line segment. May raise ValueError.
    
    Arguments:
    line is a 2-tuple of x,y coordinates, e.g. ((x1, y1), (x2, y2))

    Returns:
    The slope of the perpendicular line (float)
    """
    try:
        # Perpendicular slope is the negative reciprocal of the slope, i.e. -dx/dy
        return -1*(line.end.x - line.start.x)/(line.end.y - line.start.y)
    # Catch exceptions and raise more helpful ones if possible
    except ZeroDivisionError:
        # Raise an error if both points are the same
        if line.start == line.end:
            raise ValueError('Both points are the same')
        # Raise an error if dy = 0
        if line.start.y == line.end.y:
            raise ValueError('Line has zero slope')
    except AttributeError:
        # Raise an error if the input isn't a line segment
        if not is_valid_segment(line):
            raise ValueError('Input is not a line segment ((x1, y1), (x2, y2))')


def point_slope_to_y_intercept(m, p):
    """
    Convert a line from point-slope form to y-intercept form.
    
    The y-intercept is calculated as b = y - mx

    Arguments:
    m is the slope of the line (float)
    p is any point on the line (Point or 2-tuple)

    Returns:
    A Line object corresponding to the original point and slope (m).
    """
    return Line(m, p.y - m*p.x)


def is_vertical(l):
    """
    Determine whether a line is vertical (dx = 0).

    Arguments:
    l is a LineSegment object

    Returns:
    True if the line is vertical and False otherwise.
    """
    return l.start.x == l.end.x


def is_horizontal(l):
    """
    Determine whether a line is horizontal (dy = 0).

    Arguments:
    l is a LineSegment object

    Returns:
    True if the line is horizontal and False otherwise.
    """
    return l.start.y == l.end.y


def lines_intersection(a, b):
    """
    Find the intersection of two lines

    Arguments:
    a is a Line object
    b is a Line object

    Returns:
    A Point object representing the intersection of a and b or None if there is no intersection.
    """
    try:
        x = (b.yintercept - a.yintercept)/(a.slope - b.slope)
        y = a.slope * x + a.yintercept
        return Point(x, y)
    # Division by zero means the lines are parallel
    except ZeroDivisionError:
        return None


def line_intersect_vertical(a, p):
    """
    Find the intersection of a line with a vertical line.

    Arguments:
    a is a line defined by its slope and y-intercept, e.g. (m, b)
    p is an x,y coordinate pair that the vertical line passes through

    Returns:
    A Point where a crosses the line through p
    """
    x = p.x
    y = a.slope * x + a.yintercept
    return Point(x, y)


def compare_tris(a, b):
    """
    Determine whether two triangles are equal.

    Arguments:
    a is a Triangle or 3-tuple defining a triangle by its vertices or edges
    b is a Triangle or 3-tuple defining a triangle by its vertices or edges

    Returns:
    True if both triangles have the same vertices and False otherwise
    Also returns False for invalid input
    """
    # Only compare triangles
    if (not isinstance(a, Triangle) or not isinstance(b, Triangle)) and (len(a) != 3 or len(b) != 3):
        return False

    if a == b:
        return True

    # Order doesn't matter; triangles are the same if they have the same vertices
    if a[0] in b and a[1] in b and a[2] in b:
        return True

    return False

def calculate_tri_vertices(side_a, side_b, side_c):
    """
    Calculate the vertices of a triangle given three line segments along its sides.

    Arguments:
    a is a line segment represented by a 2-tuple of x,y coordinates, i.e. ((x1, y1), (x2, y2))
    b and c are represented the same way

    Returns:
    A vertex-defined Triangle or None.
    """
    # Make sure the inputs are line segments
    if not is_valid_segment(side_a) or not is_valid_segment(side_b) or not is_valid_segment(side_c):
        return None

    # Calculate slopes and y-intercepts of the sides
    if is_vertical(side_a):
        A = Line(None, None)
    else:
        m = slope(side_a)
        A = Line(m, side_a.end.y - m * side_a.end.x)

    if is_vertical(side_b):
        B = Line(None, None)
    else:
        m = slope(side_b)
        B = Line(m, side_b.end.y - m * side_b.end.x)

    if is_vertical(side_c):
        C = Line(None, None)
    else:
        m = slope(side_c)
        C = Line(m, side_c.end.y - m * side_c.end.x)

    # If any two sides are parallel then this is not a valid triangle
    if A.slope == B.slope or A.slope == C.slope or B.slope == C.slope:
        return None

    # Calculate the vertices
    # If one of the sides is vertical, we have a special case
    # Vertical A
    if A.slope is None:
        a_x = side_a.start.x
        a_y = B.slope*a_x + B.yintercept

        b = lines_intersection(B, C)

        c_x = side_a.start.x
        c_y = C.slope*c_x + C.yintercept
        return Triangle(Point(a_x, a_y), b, Point(c_x, c_y))
    # Vertical B
    elif B.slope is None:
        a_x = side_b.start.x
        a_y = C.slope*a_x + C.yintercept

        b_x = side_b.start.x
        b_y = C.slope*b_x + C.yintercept

        c = lines_intersection(C, A)
        return Triangle(Point(a_x, a_y), Point(b_x, b_y), c)
    # Vertical C
    elif C.slope is None:
        a = lines_intersection(A, B)

        b_x = side_c.start.x
        b_y = B.slope*b_x + B.yintercept

        c_x = side_c.start.x
        c_y = A.slope*c_x + A.yintercept
        return Triangle(a, Point(b_x, b_y), Point(c_x, c_y))

    # We may encounter a division by zero error if the slopes are too close
    try:
        a = lines_intersection(A, B)
        b = lines_intersection(B, C)
        c = lines_intersection(C, A)
        return Triangle(a, b, c)

    except ZeroDivisionError:
        return None


def triangle_from_edge_point(edge, point):
    """
    Calculate the vertices of a triangle defined by the given edge and point

    Arguments:
    edge is a LineSegment object
    point is a Point object

    Returns:
    A vertex-defined Triangle object
    """
    return Triangle(edge.start, edge.end, point)


def vertices_to_edges(t):
    """
    Convert a vertex-defined Triangle to an edge-defined one.

    Arguments:
    t is a vertex-defined Triangle object

    Returns
    An edge-defined Triangle object or None
    """
    if len(t) != 3:
        return None
    return Triangle(
        LineSegment(t.c, t.a),
        LineSegment(t.a, t.b),
        LineSegment(t.b, t.c)
    )


def edges_to_vertices(t):
    """
    Convert an edge definition of a triangle to a vertex definition

    Arguments:
    t is an edge-defined Triangle object

    Returns:
    A vertex-defined Triangle object or None
    """
    if len(t) != 3:
        return None
    return Triangle(
        t.a.end,
        t.b.end,
        t.c.end
    )


def tri_contains_point(t, p):
    """
    Determine whether the given triangle contains the given point. May raise ValueError.

    Arguments:
    t is a Triangle object
    p is a Point object

    Returns:
    True if t contains p and False otherwise.
    """
    # Validate inputs
    # Invalid triangle or invalid point
    if len(t) != 3 or len(p) != 2:
        raise ValueError('Triangle does not have three vertices')
    # Triangle defined by edges:
    for v in t:
        if len(v) != 2:
            raise ValueError('Triangle vertices are not 2-tuples')
        if not isinstance(v[0], numbers.Number) or not isinstance(v[1], numbers.Number):
            raise ValueError('Triangle vertices are not numeric')
    # Error within 1ppm is acceptable
    epsilon = 1e-6
    # Calcula1te the barycentric coordinates of p
    p1 = t.a
    p2 = t.b
    p3 = t.c
    # Make sure the point isn't a vertex
    if p == p1 or p == p2 or p == p3:
        return True
    denom = (p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y)
    if denom != 0:
        alpha = ((p2.y - p3.y)*(p.x - p3.x) + (p3.x - p2.x)*(p.y - p3.y))/denom
        beta = ((p3.y - p1.y)*(p.x - p3.x) + (p1.x - p3.x)*(p.y - p3.y))/denom
        gamma = 1.0 - alpha - beta
        # If all three coordinates are positive, p lies within t
        return alpha+epsilon >= 0 and beta+epsilon >= 0 and gamma+epsilon >= 0
    # Invalid triangle
    else:
        return False


def tri_circumcenter(t):
    """
    Calculate the circumcenter of a triangle.

    The circumcenter is the point where the perpendicular bisectors of the edges intersect.

    Arguments:
    t is a vertex-defined Triangle object

    Returns:
    A Point object representing the circumcenter of t
    """
    # The circumcenter of the triangle is the point where the perpendicular bisectors of the sides intersect
    # Define the sides we care about
    A = LineSegment(t.a, t.b)
    B = LineSegment(t.b, t.c)

    # Calculate the midpoints
    mp_a = midpoint(A)
    mp_b = midpoint(B)
    # Calculate the perpendicular slopes -- we only care about two PBs since the third will intersect them at the same point
    # If one of the lines is horizontal then its PB has infinite slope. We need to handle this case separately
    # We also need to handle the case where one of the sides is vertical, i.e. its PB is horizontal
    # This assumes that the triangle is valid; i.e. no sides are parallel
    if is_vertical(A):
        ma = 0
    elif is_horizontal(A):
        # Find where B intersects a vertical line through mp_a
        mb = perp_slope(B)
        center = line_intersect_vertical(point_slope_to_y_intercept(mb, mp_b), mp_a)
        return center
    else:
        ma = perp_slope(A)

    # Handle horizontal or vertical B
    if is_vertical(B):
        mb = 0
    elif is_horizontal(B):
        # Find where B intersects a vertical line through mp_a
        ma = perp_slope(A)
        center = line_intersect_vertical(point_slope_to_y_intercept(ma, mp_a), mp_b)
        return center
    else:
        mb = perp_slope(B)

    # Find the intersection of these lines
    center = lines_intersection(point_slope_to_y_intercept(ma, mp_a), point_slope_to_y_intercept(mb, mp_b))
    return center


def tri_centroid(t):
    """
    Calculate the centroid of a triangle.

    The centroid of a triangle is the mean of its vertices.

    Arguments:
    t is a vertex-defined Triangle object

    Returns:
    A Point object representing the centroid of the triangle.
    """
    return Point(
        ((t.a.x + t.b.x + t.c.x)/3),
        ((t.a.y + t.b.y + t.c.y)/3)
    )


def tri_circumcircle(t):
    """
    Calculate the circumcircle of a triangle.

    The radius of the circumcircle is the distance from the circumcenter to any vertex.

    Arguments:
    t is a vertex-defined Triangle object

    Returns:
    A Circle object representing the circumcircle of t
    """
    if not isinstance(t, Triangle):
        t = Triangle(
            Point(t[0][0], t[0][1]),
            Point(t[1][0], t[1][1]),
            Point(t[2][0], t[2][1])
        )

    center = tri_circumcenter(t)
    # Get the distance from the center to a vertex
    radius = sqrt((center.x - t.a.x)**2 + (center.y - t.a.y)**2)

    return Circle(center, radius)


def tri_share_vertices(t1, t2):
    """
    Determine whether two triangles have any vertices in common.

    Arguments:
    t1 and t2 are two vertex-defined Triangle objects

    Returns:
    True if any vertices are shared by the triangles and False otherwise
    """
    # Identical triangles are easy to detect
    if t1 == t2:
        return True

    # Iterate over the vertices in t1 and compare each one to every vertex in t2
    for vertex1 in t1:
        for vertex2 in t2:
            if vertex1 == vertex2:
                return True
    # No vertices matched
    return False


def angle(a, b):
    """
    Calculate the angle between two points.

    Arguments:
    a and b are Point objects

    Returns:
    The angle between a and b in radians (float)
    """
    if not isinstance(a, Point) or not isinstance(b, Point):
        a = Point(a[0], a[1])
        b = Point(b[0], b[1])
    return atan2(b.y, b.x) - atan2(a.y, a.x)


def cross_product(a, b, c):
    """
    Calculates the cross product of three points.

    Note that this isn't actually a cross product, but I couldn't think of a better name after I wrote it.

    Arguments:
    a, b, and c are all Point objects

    Returns:
    The cross product of a, b, and c (float)
    """
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)


def translate_tri(t, d):
    """
    Translate a triangle in 2-d.

    Arguments:
    t is a vertex-defined Triangle object
    d is a Vector object describing the desired translation

    Returns:
    A new Triangle object translated by d
    """
    return Triangle(
        Point(t.a.x + d.x, t.a.y + d.y),
        Point(t.b.x + d.x, t.b.y + d.y),
        Point(t.c.x + d.x, t.c.y + d.y)
    )


def scale_tri(t, s):
    """
    Scale a triangle from its center point by the given scale factor.

    Arguments:
    t is a vertex-defined Triangle object
    s is the scale factor (float)

    Returns:
    A new Triangle object scaled by s
    """
    centroid = tri_centroid(t)
    # Translate the vertices of the triangle as if the centroid were the origin
    trans_t = translate_tri(t, Vector(centroid.x * -1, centroid.y * -1))
    # Multiply all of the vertices by the scale factor
    scaled_t = Triangle(
        Point(trans_t.a.x * s, trans_t.a.y * s),
        Point(trans_t.b.x * s, trans_t.b.y * s),
        Point(trans_t.c.x * s, trans_t.c.y * s)
    )
    # Translate the triangle back
    scaled_t = translate_tri(scaled_t, centroid)
    return scaled_t


def convex_hull(points):
    """
    Calculate the convex hull of a set of points.

    Arguments:
    points is a list of 2-tuples of x,y coordinates

    Returns:
    The convex hull as a list of points represented by 2-tuples of x,y coordinate pairs, i.e. h = [(x1,y1), (x2,y2), etc] .
    """
    # Find the point with the lowest y-coordinate
    # If there's a tie, the one with the lowest x-coordinate is chosen
    min_point = None
    min_i = None
    i = -1
    for p in points:
        i += 1
        if min_point is None or p.y <= min_point.y:
            if min_point and p.y == min_point.y:
                if p.x < min_point.x:
                    min_point = p
                    min_i = i
            else:
                min_point = p
                min_i = i
    points_copy = points[::1]
    del points_copy[min_i]
    # Next, sort the points by angle (asc) relative to the minimum point
    spoints = [min_point] + sorted(points_copy, key=lambda x: angle(min_point, x))
    # Now we start iterating over the points, considering them three at a time
    hull = spoints[0:3]
    for p in spoints[3:]:
        # Find the next point
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        # Add the point to the hull
        if not hull or p != hull[-1]:
            hull.append(p)
    return hull


def enclosing_triangle(points):
    """
    Calculate a triangle that encloses a set of points -- note that the triangle may not contain any points in the given set.

    Arguments:
    points is a list of Point objects

    Returns:
    A vertex-defined Triangle object that encloses all of the points.
    """
    # Find the convex hull that encloses the points
    hull = convex_hull(points)
    # If the hull only has three points, this is our triangle
    if len(hull) == 3:
        return Triangle(
            hull[0],
            hull[1],
            hull[2]
        )
    # Convert the hull from a list of points to a list of edges
    edges = []
    for p in range(0, len(hull)):
        edges.append(LineSegment(hull[p-1], hull[p]))
    triangle = None
    # This is not a fast way to do it, but it works and is way easier to implement than the O(n) algorithm
    for i in edges:
        for j in edges:
            for k in edges:
                # Make sure we've picked three different edges
                if i != j and i != k and j != k:
                    # Create a triangle with three edges flush with the current edges of the bounding polygon
                    # Calculate the vertices of this triangle
                    triangle = calculate_tri_vertices(i, j, k)
                    # Make sure it contains all the points
                    if triangle:
                        contains_all = True
                        for p in hull:
                            if not tri_contains_point(triangle, p):
                                contains_all = False
                                break
                        if contains_all:
                            return triangle
    # We couldn't find a bounding triangle, but we can still find a bounding rectangle and convert it to a triangle
    xmin = min([p.x for p in points])
    xmax = max([p.x for p in points])
    ymin = min([p.y for p in points])
    ymax = max([p.y for p in points])

    # Use the bottom side as the base of the triangle and construct edges that pass through the two top points
    top_left = Point(xmin-1, ymax+1)
    top_right = Point(xmax+1, ymax+1)
    # Calculate equations for the edges
    left_edge = point_slope_to_y_intercept(1, top_left)
    right_edge = point_slope_to_y_intercept(-1, top_right)
    base = point_slope_to_y_intercept(0, Point(xmin-1, ymin-1))
    # The vertices of the triangle are wherever the edges intersect
    a = lines_intersection(left_edge, right_edge)
    b = lines_intersection(base, left_edge)
    c = lines_intersection(base, right_edge)

    return Triangle(a, b, c)


def delaunay_triangulation(points):
    """
    Calculate the Delaunay triangulation of a set of points. May raise ValueError.

    Currently using the Bowyer-Watson algorithm but might switch to something faster in the future. For the current algorithm, we pre-calculate all of the circumcircles of the triangles to speed things up.

    Arguments:
    points is a list of 2-tuples of x,y coordinates

    Returns:
    A list of vertex-defined Triangle objects (e.g. [t1, t2, t3, ...]) or None
    """
    # Less than three points is impossible to triangulate
    if len(points) < 3:
        raise ValueError('Need at least three points to triangulate')
    # Three points is the simplest case
    if len(points) == 3:
        return [tuple(points)]
    # Start with a triangle large enough to contain all of the points
    scale_factor = 1.5
    supertriangle = scale_tri(enclosing_triangle(points), scale_factor)
    if not supertriangle:
        return None

    # The graph is a list of 2-tuples; the first element of each 2-tuple is a
    # triangle and the second element is its circumcircle.
    # This saves us considerable time in recalculating the circles
    graph = [(supertriangle, tri_circumcircle(supertriangle))]
    # Add points to the graph one at a time
    for p in points:
        # Find the triangles that contain the point
        invalid_triangles_vertices = []
        invalid_triangles_edges = []
        for (t, circumcircle) in graph:
            if sqrt((circumcircle.center.x-p.x)**2+(circumcircle.center.y-p.y)**2) <= circumcircle.radius:
                # Add the triangle to the list
                invalid_triangles_edges.append(vertices_to_edges(t))
                invalid_triangles_vertices.append(t)
        # There is a polygonal hole around the new point. Find its edges
        hole = []
        for t in invalid_triangles_edges:
            for e in t:
                # Make sure the edge is unique
                unique = True
                for u in invalid_triangles_edges:
                    # Tried using reversed(e), but it returns an iterable and not a tuple
                    if (e in u or e[::-1] in u) and u != t:
                        unique = False
                        break
                # If the edge is unique, add it to the hole
                if unique:
                    hole.append(e)
        # Delete the invalid triangles from the graph
        for t in invalid_triangles_vertices:
            i = 0
            while i < len(graph):
                if t == graph[i][0]:
                    del graph[i]
                    break
                i += 1
        # Re-triangulate the hole made by the new point
        for e in hole:
            if e[0] != p and e[1] != p:
                t = triangle_from_edge_point(e, p)
                graph.append((t, tri_circumcircle(t)))
    # Delete the supertriangle from the graph
    i = 0
    while i < len(graph):
        if tri_share_vertices(graph[i][0], supertriangle):
            del graph[i]
            i -= 1
        i += 1

    # Clean up the graph data structure, removing all of the circumcircles
    clean_graph = [t[0] for t in graph]

    # Return the graph
    return clean_graph
