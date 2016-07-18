"""
Functions for generating points.

All functions return a list of Point objects.
"""
from __future__ import division
from random import randrange
from math import ceil, sqrt
from fractions import gcd
from geometry import Point


def generate_random_points(count, area, scale=1, decluster=True):
    """
    Generate points at random locations using python's random module.

    Arguments:
    count is the number of points to generate (int)
    area is a 2-tuple of the maximum x and y values of the field
    scale is a value that describes how much of the area to fill with points
      -- values greater than 1 will result in points outside the area
      -- values less than 1 result in points bounded by scale*area
      -- default is 1
    decluster is a boolean flag; points will be de-clustered if True (default)

    Returns:
    A list of Point objects.
    """
    cluster_fraction = 1
    if decluster:
        # Generate extra points for declustering later
        cluster_fraction = 2

    # Scale the area
    bound_x, bound_y = int(area[0]*scale), int(area[1]*scale)

    n_extra_points = int(cluster_fraction*count)
    points = [
        Point(int(randrange(bound_x)), int(randrange(bound_y)))
        for __ in range(0, n_extra_points)
    ]

    # Translate the points so that some are in negative x and y for nicer edges
    dx = (bound_x - area[0]) / 2
    dy = (bound_y - area[1]) / 2

    points = [Point(p.x - dx, p.y - dy) for p in points]

    return points


def generate_rectangular_points(count, area):
    """
    Generate a rectangular grid of points.

    Arguments:
    count is the approximate number of points to generate (int)
    area is a 2-tuple of the maximum x and y values of the field

    Returns:
    A list of Point objects
    """
    # Reduce the area to lowest terms and calculate grid spacing
    k = gcd(area[0], area[1])
    reduced_x, reduced_y = area[0]/k, area[1]/k
    aspect = reduced_x / reduced_y

    count_x = ceil(sqrt(count)*aspect)
    count_y = ceil(sqrt(count)/aspect)

    x_spacing = int(max(ceil(area[0]/count_x), 1))
    y_spacing = int(max(ceil(area[1]/count_y), 1))

    return [
        Point(x, y)
        for y in range(0, area[1], y_spacing)
        for x in range(0, area[0], x_spacing)
    ]


def generate_equilateral_points(count, area):
    """
    Generate a set of points that will triangulate to equilateral triangles.

    To get equilateral triangles, the grid has to be offset so that every
    other line is advanced by half the x-spacing. For example, a grid like

    *   *   *   *
      *   *   *
    *   *   *   *

    Will result in equilateral triangles.

    Arguments:
    n_points is the number of points to generate
    area is a 2-tuple describing the maximum x and y values of the field

    Returns:
    A list of Point objects
    """
    points = []

    # Figure out roughly how many points we need in x
    count_x = ceil(sqrt(count))

    # Calculate the spacing
    x_spacing = max(ceil(area[0]/count_x), 1)
    x_offset = x_spacing/2
    y_spacing = sqrt(x_spacing**2 - (x_spacing/2)**2)

    # Generate the points
    xmax = area[0]+x_spacing+x_offset
    ymax = area[1]+y_spacing
    y = 0
    odd_row = False
    while y <= ymax:
        # Offset every other row for equilateral triangles
        if odd_row:
            x = -x_offset
        else:
            x = 0

        while x <= xmax:
            points.append(Point(int(x), int(y)))
            x += x_spacing
        odd_row = not odd_row
        y += y_spacing

    return points


def generate_halton_points(count, area, p=2, q=3):
    """
    Generate points using the p,q Halton sequence.

    https://en.wikipedia.org/wiki/Halton_sequence
    A Halton sequence is a quasi-random sequence that uses one prime number for
    each dimension as a base. For example, the 2,3 Halton sequence
    (the default) looks like this:

    (1/2, 1/3), (1/4, 2/3), (3/4, 1/9), (1/8, 4/9), ...

    These values are then scaled to fill the given area.

    Arguments:
    count is the number of points to generate
    area is a 2-tuple of the maximum x and y values of the field
    p is the first prime (default 2)
    q is the second prime (default 3)

    Returns:
    A list of Point objects
    """
    points = []
    for i in xrange(1, count+1):
        fx = 1
        fy = 1
        ix = i
        iy = i
        rx = 0
        ry = 0

        # Calculate the ith p- and q-values
        while ix > 0:
            fx /= p
            rx += fx * (ix % p)
            ix = int(ix/p)

        while iy > 0:
            fy /= q
            ry += fy * (iy % q)
            iy = int(iy/q)

        # Scale the point and add to the list
        points.append(Point(rx*area[0], ry*area[1]))

    # Add four anchor points at the corners
    points.append(Point(0, 0))
    points.append(Point(0, area[1]))
    points.append(Point(area[0], area[1]))
    points.append(Point(area[0], 0))

    return points
