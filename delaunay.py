#!/usr/bin/env python3
"""
Use Delaunay triangulations to make interesting images.

usage: delaunay.py [-h] [-o OUTPUT_FILENAME] [-n N_POINTS] [-x WIDTH]
                   [-y HEIGHT] [-g GRADIENT] [-i INPUT_FILENAME]
                   [-k DARKEN_AMOUNT] [-a] [-l] [-d] [-r] [-e]

Try delaunay.py --help for details.
"""
from __future__ import print_function
import sys
import argparse
from PIL import Image, ImageDraw
from random import randrange
from collections import namedtuple
from math import sqrt
from geometry import delaunay_triangulation, tri_centroid, Point, Triangle
from distributions import *

# Some types to make things a little easier
Color = namedtuple('Color', 'r g b')
Gradient = namedtuple('Gradient', 'start end')


def hex_to_color(hex_value):
    """
    Convert a hexadecimal representation of a color to an RGB triplet.

    For example, the hex value FFFFFF corresponds to (255, 255, 255).

    Arguments:
    hex_value is a string containing a 6-digit hexadecimal color

    Returns:
    A Color object equivalent to the given hex value or None for invalid input
    """
    if hex_value is None:
        return None

    if hex_value[0] == '#':
        hex_value = hex_value[1:]

    hex_value = hex_value.lower()

    red = hex_value[:2]
    green = hex_value[2:4]
    blue = hex_value[4:]

    try:
        return Color(int(red, 16), int(green, 16), int(blue, 16))
    except ValueError:
        return None


def cart_to_screen(points, size):
    """
    Convert Cartesian coordinates to screen coordinates.

    Arguments:
    points is a list of Point objects or a vertex-defined Triangle object
    size is a 2-tuple of the screen dimensions (width, height)

    Returns:
    A list of Point objects or a Triangle object, depending on the type of the input
    """
    if isinstance(points, Triangle):
        return Triangle(
            Point(points.a.x, size[1] - points.a.y),
            Point(points.b.x, size[1] - points.b.y),
            Point(points.c.x, size[1] - points.c.y)
        )
    else:
        trans_points = [Point(p.x, size[1] - p.y) for p in points]
        return trans_points


def calculate_color(grad, val):
    """
    Calculate a point on a color gradient. Color values are in [0, 255].

    Arguments:
    grad is a Gradient object
    val is a value in [0, 1] indicating where the color is on the gradient

    Returns:
    A Color object
    """
    slope_r = grad.end.r - grad.start.r
    slope_g = grad.end.g - grad.start.g
    slope_b = grad.end.b - grad.start.b

    r = int(grad.start.r + slope_r*val)
    g = int(grad.start.g + slope_g*val)
    b = int(grad.start.b + slope_b*val)

    # Perform thresholding
    r = min(max(r, 0), 255)
    g = min(max(g, 0), 255)
    b = min(max(b, 0), 255)

    return Color(r, g, b)


def draw_polys(draw, colors, polys):
    """
    Draw a set of polygons to the screen using the given colors.

    Arguments:
    colors is a list of Color objects, one per polygon
    polys is a list of polygons defined by their vertices as x, y coordinates
    """
    for i in range(0, len(polys)):
        draw.polygon(polys[i], fill=colors[i])


def draw_lines(draw, color, polys, line_thickness=1):
    """
    Draw the edges of the given polygons to the screen in the given color.

    Arguments:
    draw is an ImageDraw object
    color is a Color tuple
    polys is a list of vertices
    line_thickness is the thickness of each line in px (default 1)
    """
    if line_thickness is None:
        line_thickness = 1

    for p in polys:
        draw.line(p, color, line_thickness)


def draw_points(draw, color, polys, vert_radius=16):
    """
    Draw the vertices of the given polygons to the screen in the given color.

    Arguments:
    draw is an ImageDraw object
    color is a Color tuple
    polys is a list of vertices
    vert_radius is the radius of each vertex in px (default 16)
    """
    if vert_radius is None:
        vert_radius = 16

    for p in polys:
        v1 = [p[0].x - vert_radius/2, p[0].y - vert_radius/2, p[0].x + vert_radius/2, p[0].y + vert_radius/2]
        v2 = [p[1].x - vert_radius/2, p[1].y - vert_radius/2, p[1].x + vert_radius/2, p[1].y + vert_radius/2]
        v3 = [p[2].x - vert_radius/2, p[2].y - vert_radius/2, p[2].x + vert_radius/2, p[2].y + vert_radius/2]
        draw.ellipse(v1, color)
        draw.ellipse(v2, color)
        draw.ellipse(v3, color)


def color_from_image(background_image, triangles):
    """
    Color a graph of triangles using the colors from an image.

    The color of each triangle is determined by the color of the image pixel at
    its centroid.

    Arguments:
    background_image is a PIL Image object
    triangles is a list of vertex-defined Triangle objects

    Returns:
    A list of Color objects, one per triangle such that colors[i] is the color
    of triangle[i]
    """
    colors = []
    pixels = background_image.load()
    size = background_image.size
    for t in triangles:
        centroid = tri_centroid(t)
        # Truncate the coordinates to fit within the boundaries of the image
        int_centroid = (
            int(min(max(centroid[0], 0), size[0]-1)),
            int(min(max(centroid[1], 0), size[1]-1))
        )
        # Get the color of the image at the centroid
        p = pixels[int_centroid[0], int_centroid[1]]
        colors.append(Color(p[0], p[1], p[2]))
    return colors


def color_from_gradient(gradient, image_size, triangles):
    """
    Color a graph of triangles using a gradient.

    Arguments:
    gradient is a Gradient object
    image_size is a tuple of the output dimensions, i.e. (width, height)
    triangles is a list of vertex-defined Triangle objects

    Returns:
    A list of Color objects, one per triangle such that colors[i] is the color
    of triangle[i]
    """
    colors = []
    # The size of the screen
    s = sqrt(image_size[0]**2+image_size[1]**2)
    for t in triangles:
        # The color is determined by the location of the centroid
        tc = tri_centroid(t)
        # Bound centroid to boundaries of the image
        c = (min(max(0, tc[0]), image_size[0]),
             min(max(0, tc[1]), image_size[1]))
        frac = sqrt(c[0]**2+c[1]**2)/s
        colors.append(calculate_color(gradient, frac))
    return colors


def main():
    """Calculate Delaunay triangulation and output an image"""
    # Anti-aliasing amount -- multiply screen dimensions by this when supersampling
    aa_amount = 4
    # Some gradients
    gradient = {
        'sunshine': Gradient(
            Color(255, 248, 9),
            Color(255, 65, 9)
        ),
        'purples': Gradient(
            Color(255, 9, 204),
            Color(4, 137, 232)
        ),
        'grass': Gradient(
            Color(255, 232, 38),
            Color(88, 255, 38)
        ),
        'valentine': Gradient(
            Color(102, 0, 85),
            Color(255, 25, 216)
        ),
        'sky': Gradient(
            Color(0, 177, 255),
            Color(9, 74, 102)
        ),
        'ubuntu': Gradient(
            Color(119, 41, 83),
            Color(221, 72, 20)
        ),
        'fedora': Gradient(
            Color(41, 65, 114),
            Color(60, 110, 180)
        ),
        'debian': Gradient(
            Color(215, 10, 83),
            Color(10, 10, 10)
        ),
        'opensuse': Gradient(
            Color(151, 208, 5),
            Color(34, 120, 8)
        )
    }

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.set_defaults(output_filename='triangles.png')
    parser.set_defaults(n_points=100)
    parser.set_defaults(distribution='uniform')

    # Value options
    parser.add_argument('-o', '--output', dest='output_filename', help='The filename to write the image to. Supported filetypes are BMP, TGA, PNG, and JPEG')
    parser.add_argument('-n', '--npoints', dest='n_points', type=int, help='The number of points to use when generating the triangulation.')
    parser.add_argument('-x', '--width', dest='width', type=int, help='The width of the image.')
    parser.add_argument('-y', '--height', dest='height', type=int, help='The height of the image.')
    parser.add_argument('-g', '--gradient', dest='gradient', help='The name of the gradient to use.')
    parser.add_argument('-i', '--image-file', dest='input_filename', help='An image file to use when calculating triangle colors. Image dimensions will override dimensions set by -x and -y.')
    parser.add_argument('-k', '--darken', dest='darken_amount', type=int, help='Darken random triangles my the given amount to make the pattern stand out more')

    # Flags
    parser.add_argument('-a', '--antialias', dest='antialias', action='store_true', help='If enabled, draw the image at 4x resolution and downsample to reduce aliasing.')
    parser.add_argument('-l', '--lines', dest='lines', action='store_true', help='If enabled, draw lines along the triangle edges.')
    parser.add_argument('--linethickness', dest='line_thickness', type=int, help='The thickness (in px) of edges drawn on the graph. Implies -l.')
    parser.add_argument('--linecolor', dest='line_color', type=str, help='The color of edges drawn on the graph in hex (e.g. ffffff for white). Implies -l.')
    parser.add_argument('-p', '--points', dest='points', action='store_true', help='If enabled, draw a circle for each vertex on the graph.')
    parser.add_argument('--vertexradius', dest='vert_radius', type=int, help='The radius (in px) of the vertices drawn on the graph. Implies -p.')
    parser.add_argument('--vertexcolor', dest='vert_color', type=str, help='The color of vertices drawn on the graph in hex (e.g. ffffff for white). Implies -p.')
    parser.add_argument('--distribution', dest='distribution', type=str, help='The desired distribution of the random points. Options are uniform (default) or Halton.')
    parser.add_argument('-d', '--decluster', dest='decluster', action='store_true', help='If enabled, try to avoid generating clusters of points in the triangulation. This will significantly slow down point generation.')
    parser.add_argument('-r', '--right', dest='right_tris', action='store_true', help='If enabled, generate right triangles rather than random ones.')
    parser.add_argument('-e', '--equilateral', dest='equilateral_tris', action='store_true', help='If enabled, generate equilateral triangles rather than random ones.')

    # Parse the arguments
    options = parser.parse_args()

    # Set the number of points to use
    npoints = options.n_points

    # Make sure the gradient name exists (if applicable)
    gname = options.gradient
    if not gname and not options.input_filename:
        print('Require either gradient (-g) or input image (-i). Try --help for details.')
        sys.exit(64)
    elif gname not in gradient and not options.input_filename:
        print('Invalid gradient name')
        sys.exit(64)
    elif options.input_filename:
        # Warn if a gradient was selected as well as an image
        if options.gradient:
            print('Image supercedes gradient; gradient selection ignored')
        background_image = Image.open(options.input_filename)

    # Input and output files can't be the same
    if options.input_filename == options.output_filename:
        print('Input and output files must be different.')
        sys.exit(64)

    # If an image is being used as the background, set the canvas size to match it
    if options.input_filename:
        # Warn if overriding user-defined width and height
        if options.width or options.height:
            print('Image dimensions supercede specified width and height')
        size = background_image.size
    else:
        # Make sure width and height are positive
        if options.width <= 0 or options.height <= 0:
            print('Width and height must be greater than zero.')
            sys.exit(64)

        size = (options.width, options.height)


    # Generate points on this portion of the canvas
    scale = 1.25
    if options.equilateral_tris:
        points = generate_equilateral_points(npoints, size)
    elif options.right_tris:
        points = generate_rectangular_points(npoints, size)
    else:
        if options.distribution == 'uniform':
            points = generate_random_points(npoints, size, scale, options.decluster)
        elif options.distribution == 'halton':
            points = generate_halton_points(npoints, size)
        else:
            print('Unrecognized distribution type.')
            sys.exit(64)

    # Dedup the points
    points = list(set(points))

    # Calculate the triangulation
    triangulation = delaunay_triangulation(points)

    # Failed to find a triangulation
    if not triangulation:
        print('Failed to find a triangulation.')
        sys.exit(1)

    # Translate the points to screen coordinates
    trans_triangulation = list(map(lambda x: cart_to_screen(x, size), triangulation))

    # Assign colors to the triangles
    if options.input_filename:
        colors = color_from_image(background_image, trans_triangulation)
    else:
        colors = color_from_gradient(gradient[gname], size, trans_triangulation)

    # Darken random triangles
    if options.darken_amount:
        for i in range(0, len(colors)):
            c = colors[i]
            d = randrange(options.darken_amount)
            darkened = Color(max(c.r-d, 0), max(c.g-d, 0), max(c.b-d, 0))
            colors[i] = darkened

    # Set up for anti-aliasing
    if options.antialias:
        # Scale the image dimensions
        size = (size[0] * aa_amount, size[1] * aa_amount)
        # Scale the graph
        trans_triangulation = [
            Triangle(
                Point(t.a.x * aa_amount, t.a.y * aa_amount),
                Point(t.b.x * aa_amount, t.b.y * aa_amount),
                Point(t.c.x * aa_amount, t.c.y * aa_amount)
            )
            for t in trans_triangulation
        ]

    # Create image object
    image = Image.new('RGB', size, 'white')
    # Get a draw object
    draw = ImageDraw.Draw(image)
    # Draw the triangulation
    draw_polys(draw, colors, trans_triangulation)

    if options.lines or options.line_thickness or options.line_color:
        if options.line_color is None:
            line_color = Color(255, 255, 255)
        else:
            line_color = hex_to_color(options.line_color)

        draw_lines(draw, line_color, trans_triangulation, options.line_thickness)

    if options.points or options.vert_radius or options.vert_color:
        if options.vert_color is None:
            vertex_color = Color(255, 255, 255)
        else:
            vertex_color = hex_to_color(options.vert_color)

        draw_points(draw, vertex_color, trans_triangulation, options.vert_radius)

    # Resample the image using the built-in Lanczos filter
    if options.antialias:
        size = (int(size[0]/aa_amount), int(size[1]/aa_amount))
        image = image.resize(size, Image.ANTIALIAS)

    # Write the image to a file
    image.save(options.output_filename)
    print('Image saved to %s' % options.output_filename)
    sys.exit(0)

# Run the main function
if __name__ == '__main__':
    main()
