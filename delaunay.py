#!/usr/bin/python

import Image, ImageDraw
import random
import sys
from optparse import OptionParser
from math import sqrt, ceil
from fractions import gcd
from geometry import delaunay_triangulation, tri_centroid

# Convert Cartesian coordinates to screen coordinates
#	points is a list of points of the form (x,y)
#	size is a 2-tuple of the screen dimensions (width, height)
#	Returns a list of points that have been transformed to screen coords
def cart_to_screen(points, size):
	trans_points = []
	if points:
		for p in points:
			trans_points.append((p[0], size[1]-p[1]))
	return trans_points

# Calculate a point on a color gradient
#	grad is a gradient with two endpoints, both 3-tuples of RGB coordinates
#	val is a value between 0 and 1 indicating the point on the gradient where we want the color
#	Returns a set of RGB coordinates representing the color at that point on the gradient
def calculate_color(grad, val):
	# Do gradient calculations
	slope_r = grad[1][0] - grad[0][0]
	slope_g = grad[1][1] - grad[0][1]
	slope_b = grad[1][2] - grad[0][2]

	r = int(grad[0][0] + slope_r*val)
	g = int(grad[0][1] + slope_g*val)
	b = int(grad[0][2] + slope_b*val)

	# Perform thresholding
	r = min(max(r,0),255)
	g = min(max(g,0),255)
	b = min(max(b,0),255)

	return (r, g, b)

# Generate a random set of points to triangulate
#	n_points is the number of points to generate
#	area is a 2-tuple describing the boundaries of the field in x and y
#	scale is a value that describes how much of the area to fill with points
#	 -- values greater than 1 will result in points outside the area, values less than 1 result in a set of points bounded by a rectangle that's scale times the size of the given area
#	decluster is a boolean flag - if True, extra points will be generated, the points will be sorted by distance to the nearest neighbor, and the extra points with the closest neighbors will be removed - this avoids clusters of points
def generate_random_points(n_points, area, scale=1, decluster=True):
	# Generate random points -- note that we generate extra_points so we can de-cluster later
	if decluster:
		cluster_fraction = 2
	else:
		cluster_fraction = 1

	points = []
	n_extra_points = int(cluster_fraction*n_points)
	for i in range(0,n_extra_points):
		points.append((int((1-scale)/2*area[0])+random.randrange(int(area[0]*scale)), int((1-scale)/2*area[1])+random.randrange(int(area[1]*scale))))

	# De-cluster the points
	if decluster:
		# Sort the points by distance to the nearest point
		sorted_points = []
		for p in points:
			d = None
			# Find the minimum distance to another point
			for q in points:
				if q == p:
					break
				q_d = sqrt((p[0]-q[0])**2+(p[1]-q[1])**2)
				if not d or q_d < d:
					d = q_d
			if d:
				# Insert the distance-point pair into the array at the correct position
				#if len(sorted_points) == 0:
				if not sorted_points:
					sorted_points.append((d, p))
				# Does it go at the end of the list?
				elif d > sorted_points[-1][0]:
					sorted_points.append((d, p))
				else:
					i = 0
					while i < len(sorted_points):
						if sorted_points[i][0] < d:
							i += 1
						else:
							sorted_points.insert(i, (d, p))
							break
		# Remove the most clustered points
		for i in range(0,(n_extra_points - n_points)):
			del sorted_points[0]
		# Put the remaining points back into a flat list
		points = [p[1] for p in sorted_points]
	
	# Always include four points beyond the corners of the screen to make sure the triangulation covers the edges
	points.append((-300,-10))
	points.append((area[0]+10, -300))
	points.append((area[0]+300, area[1]+10))
	points.append((-100, area[1]+300))
	return points

# Generate a rectangular grid of points
#	n_points is the number of points to generate
#	area is a 2-tuple describing the boundaries of the field in x and y
def generate_grid_points(n_points, area):
	points = []

	# Find the GCD of x and y and factor it out
	k = gcd(area[0],area[1])
	reduced_x = area[0]/k
	reduced_y = area[1]/k

	# Find a number of points that will make a nice grid
	n_points_x = ceil(sqrt(n_points*reduced_x))
	n_points_y = ceil(sqrt(n_points*reduced_y))

	x_spacing = max(ceil(area[0]/n_points_x), 1)
	y_spacing = max(ceil(area[1]/n_points_y), 1)

	y = 0
	while y <= area[1]+y_spacing:
		x = 0
		while x <= area[0]+x_spacing:
			points.append((x, y))
			x += x_spacing
		y += y_spacing

	return points

# Generate a set of points that will triangulate to equilateral triangles
#	n_points is the number of points to generate
#	area is a 2-tuple describing the boundaries of the field in x and y
def generate_equilateral_points(n_points, area):
	points = []

	# Figure out roughly how many points we need in x
	n_points_x = ceil(sqrt(n_points))

	# Calculate the spacing
	x_spacing = max(ceil(area[0]/n_points_x), 1)
	x_offset = x_spacing/2
	y_spacing = sqrt(x_spacing**2 - (x_spacing/2)**2)
	
	# Generate the points
	xmax = area[0]+x_spacing+x_offset
	ymax = area[1]+y_spacing
	y = 0
	odd_row = False
	while y <= ymax:
		# Offset adjacent rows to get equilateral triangles instead of right triangles
		if odd_row:
			x = -x_offset
		else:
			x = 0

		while x <= xmax:
			points.append((int(x),int(y)))
			x += x_spacing
		odd_row = not odd_row
		y += y_spacing
	
	return points

# Draw a set of polygons to the screen using the given colors
#	colors is a list of RGB coordinates, one per polygon
#	polys is a list of polygons defined by their vertices as x,y coordinates
#	draw_outlines is a boolean -- if true, draws outlines around the triangles
#	outline_color is an RGB triplet describing the color of the outlines of the triangles (if drawn)
def draw_polys(draw, colors, polys, draw_outlines=False, outline_color=(255,255,255)):
	if draw_outlines:
		for i in range(0, len(polys)):
			draw.polygon(polys[i], outline=outline_color, fill=colors[i])
	else:
		for i in range(0, len(polys)):
			draw.polygon(polys[i], fill=colors[i])

# Color a graph of triangles using the colors from an image -- uses the value of the image at the centroid of the triangle to determine the color of the triangle
#	background_image is a PIL Image object
#	triangles is a Delaunay triangulation like those generated by geometry.delaunay_triangulation()
#	Returns a list of RGB coordinates describing the color of each triangle such that colors[i] gives the color of triangle[i] as (R, G, B)
def color_from_image(background_image, triangles):
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
		colors.append(pixels[int_centroid[0], int_centroid[1]])
	return colors

# Color a graph of triangles using a gradient
#	gradient is a tuple of RGB triplets describing the start and end points of a color gradient -- value is linearly interpolated between the endpoints
#	image_size is a tuple describing the dimensions of the output image, i.e. (width, height)
#	triangles is a Delaunay triangulation like those generated by geometry.delaunay_triangulation()
#	Returns a list of RGB coordinates describing the color of each triangle such that colors[i] gives the color of triangle[i] as (R, G, B)
def color_from_gradient(gradient, image_size, triangles):
	colors = []
	# The size of the screen
	s = sqrt(image_size[0]**2+image_size[1]**2)
	for t in triangles:
		# The color is determined by the location of the centroid
		c = tri_centroid(t)
		frac = sqrt(c[0]**2+c[1]**2)/s
		colors.append(calculate_color(gradient, frac))
	return colors

# Anti-aliasing amount -- multiplies the screen dimensions by this value when supersampling
aa_amount = 4
# Some gradients
gradient = {
'sunshine':(
	(255,248,9),
	(255,65,9)
),
'purples':(
	(255,9,204),
	(4,137,232)
),
'grass':(
	(255,232,38),
	(88,255,38)
),
'valentine':(
	(102,0,85),
	(255,25,216)
),
'sky':(
	(0,177,255),
	(9,74,102)
),
'ubuntu':(
	(119,41,83),
	(221,72,20)
),
'fedora':(
	(41,65,114),
	(60,110,180)
),
'debian':(
	(215,10,83),
	(10,10,10)
),
'opensuse':(
	(151,208,5),
	(34,120,8)
)
}

# Get command line arguments
parser = OptionParser()
parser.set_defaults(filename='triangles.png')
parser.set_defaults(n_points=100)

# Value options
parser.add_option('-o', '--output', dest='filename', type='string', help='The filename to write the image to. Supported filetyles are BMP, TGA, PNG, and JPEG')
parser.add_option('-n', '--npoints', dest='n_points', type='int', help='The number of points to use when generating the triangulation.')
parser.add_option('-x', '--width', dest='width', type='int', help='The width of the image.')
parser.add_option('-y', '--height', dest='height', type='int', help='The height of the image.')
parser.add_option('-g', '--gradient', dest='gradient', type='string', help='The name of the gradient to use.')
parser.add_option('-i', '--image-file', dest='image', type='string', help='An image file to use when calculating triangle colors. Image dimensions will override dimensions set by -x and -y.')
parser.add_option('-k', '--darken', dest='darken_amount', type='int', help='Darken random triangles my the given amount to make the pattern stand out more')

# Flags
parser.add_option('-a', '--antialias', dest='antialias', action='store_true', help='If enabled, draw the image at 4x resolution and downsample to reduce aliasing.')
parser.add_option('-l', '--lines', dest='lines', action='store_true', help='If enabled, draw lines along the triangle edges.')
parser.add_option('-d', '--decluster', dest='decluster', action='store_true', help='If enabled, try to avoid generating clusters of points in the triangulation. This will significantly slow down point generation.')
parser.add_option('-r', '--right', dest='right_tris', action='store_true', help='If enabled, generate right triangles rather than random ones.')
parser.add_option('-e', '--equilateral', dest='equilateral_tris', action='store_true', help='If enabled, generate equilateral triangles rather than random ones.')

# Parse the arguments
(options, args) = parser.parse_args()

# Set the number of points to use
npoints = options.n_points

# Make sure the gradient name exists (if applicable)
gname = options.gradient
if not gname and not options.image:
	print 'Must select either a gradient (-g) or input image (-i). See help for details.'
	sys.exit(64)
elif gname not in gradient and not options.image:
	print 'Invalid gradient name'
	sys.exit(64)
elif options.image:
	# Warn if a gradient was selected as well as an image
	if options.gradient:
		print 'Image supercedes gradient; gradient selection ignored'
	background_image = Image.open(options.image)


# If an image is being used as the background, set the canvas size to match it
if options.image:
	# Warn if overriding user-defined width and height
	if options.width or options.height:
		print 'Image dimensions supercede specified width and height'
	size = background_image.size
else:
	# Make sure width and height are positive
	if options.width <= 0 or options.height <= 0:
		print 'Width and height must be greater than zero.'
		sys.exit(64)

	size = (options.width, options.height)


# Generate points on this portion of the canvas
scale = 1.25
if options.equilateral_tris:
	points = generate_equilateral_points(npoints, size)
elif options.right_tris:
	points = generate_grid_points(npoints, size)
else:
	points = generate_random_points(npoints, size, scale, options.decluster)

# Calculate the triangulation
triangulation = delaunay_triangulation(points)

# Failed to find a triangulation
if not triangulation:
	print 'Failed to find a triangulation.'
	sys.exit(1)

# Translate the points to screen coordinates
trans_triangulation = map(lambda x:cart_to_screen(x, size), triangulation)

# Assign colors to the triangles
if options.image:
	colors = color_from_image(background_image, trans_triangulation)
else:
	colors = color_from_gradient(gradient[gname], size, trans_triangulation)

# Darken random triangles
if options.darken_amount:
	for i in range(0,len(colors)):
		c = colors[i]
		d = random.randrange(options.darken_amount)
		darkened = (max(c[0]-d,0), max(c[1]-d,0), max(c[2]-d,0))
		colors[i] = darkened

# Set up for anti-aliasing
if options.antialias:
	# Scale the image dimensions
	size = (size[0] * aa_amount, size[1] * aa_amount)
	# Scale the graph
	trans_triangulation = [[tuple(map(lambda x:x*aa_amount, v)) for v in p] for p in trans_triangulation]

# Create image object
image = Image.new('RGB', size, 'white')
# Get a draw object
draw = ImageDraw.Draw(image)
# Draw the triangulation
draw_polys(draw, colors, trans_triangulation, options.lines)
# Resample the image using the built-in Lanczos filter
if options.antialias:
	image = image.resize((size[0]/aa_amount, size[1]/aa_amount), Image.ANTIALIAS)

# Write the image to a file
image.save(options.filename)
print 'Image saved to %s' % options.filename
sys.exit(0)

