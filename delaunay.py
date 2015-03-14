#!/bin/python

import pygame
import random
import geometry
import sys
from optparse import OptionParser

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
def generate_points(n_points, area, scale=1, decluster=True):
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
				q_d = geometry.distance(p, q)
				if not d or q_d < d:
					d = q_d
			# Insert the distance-point pair into the array at the correct position
			if len(sorted_points) == 0:
				sorted_points.append((d, p))
			else:
				# Does it go at the end of the list?
				if d > sorted_points[-1][0]:
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


# Draw a set of polygons to the screen using the given colors
#	polys is a list of polygons defined by their vertices as x,y coordinates
#	colors is a list of RGB coordinates, one per polygon
def draw_polys(colors, polys, surface, draw_outlines=False, outline_color=(255,255,255)):
	for i in range(0, len(polys)):
		pygame.draw.polygon(surface, colors[i], polys[i])
	if draw_outlines:
		for p in polys:
			pygame.draw.aalines(screen, outline_color, True, p)
		

pygame.init()

point_radius = 5
poly_thickness = 2

bg_color = (255,255,255)
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
parser.set_defaults(width=1920)
parser.set_defaults(height=1080)
parser.set_defaults(gradient='sunshine')

parser.add_option('-f', '--file', dest='filename', type='string', help='The filename to write the image to. Supported filetyles are BMP, TGA, PNG, and JPEG')
parser.add_option('-n', '--npoints', dest='n_points', type='int', help='The number of points to use when generating the triangulation.')
parser.add_option('-x', '--width', dest='width', type='int', help='The width of the image.')
parser.add_option('-y', '--height', dest='height', type='int', help='The height of the image.')
parser.add_option('-g', '--gradient', dest='gradient', type='string', help='The name of the gradient to use.')
parser.add_option('-i', '--image-file', dest='image', type='string', help='An image file to use when calculating triangle colors. Image dimensions will override dimensions set by -x and -y.')
parser.add_option('-l', '--lines', dest='lines', action='store_true', help='If enabled, draw lines along the triangle edges.')
parser.add_option('-d', '--decluster', dest='decluster', action='store_true', help='If enabled, try to avoid generating clusters of points in the triangulation. This will significantly slow down point generation.')

# Parse the arguments
(options, args) = parser.parse_args()

# Set the size of the image
npoints = options.n_points
# Make sure the gradient name exists (if applicable)
gname = options.gradient
if gname not in gradient and not options.image:
	print 'Invalid gradient name'
	sys.exit(64)
if options.image:
	# Warn if a gradient was selected as well as an image
	if options.gradient:
		print 'Image supercedes gradient; gradient selection ignored'
	image = pygame.image.load(options.image)

# Make sure width and height are positive
if options.width <= 0 or options.height <= 0:
	print 'Width and height must be greater than zero.'
	sys.exit(64)

# If an image is being used as the background, set the canvas size to match it
if image:
	size = image.get_size()
else:
	size = (options.width, options.height)

# Set up the screen
screen = pygame.display.set_mode(size)

# Generate points on this portion of the canvas
scale = 1.25
points = generate_points(npoints, size, scale, options.decluster)

# Calculate the triangulation
triangulation = geometry.calculate_triangles(points)

# Failed to find a triangulation
if not triangulation:
	print 'Failed to find a triangulation.'
	sys.exit(1)

# Translate the points to screen coordinates
trans_triangulation = []
for t in triangulation:
	trans_triangulation.append(cart_to_screen(t, size))

# Assign colors to the triangles
colors = []
# If an image was selected, assign colors to the triangles based on the color of the image at the centroid of each triangle
# Note that we translate the centroid to screen coordinates before sampling the image
if image:
	for t in trans_triangulation:
		centroid = geometry.tri_centroid(t)
		# Truncate the coordinates to fit within the boundaries of the image
		int_centroid = [centroid[0], centroid[1]]
		if centroid[0] < 0:
			int_centroid[0] = 0
		elif centroid[0] >= size[0]:
			int_centroid[0] = size[0] - 1
		else:
			int_centroid[0] = int(centroid[0])
		if centroid[1] < 0:
			int_centroid[1] = 0
		elif centroid[1] >= size[1]:
			int_centroid[1] = size[1] - 1
		else:
			int_centroid[1] = int(centroid[1])

		colors.append(image.get_at(int_centroid))
else:
	# If a gradient was selected, use that to assign colors to the triangles
	# The size of the screen
	s = geometry.distance((0,0), size)
	for t in triangulation:
		# The color is determined by the location of the centroid
		c = geometry.tri_centroid(t)
		frac = geometry.distance((0,0), c)/s
		#frac = c[0]/size[0]
		colors.append(calculate_color(gradient[gname], frac))

# Draw
# Blank the screen
screen.fill(bg_color)
# Draw the triangulation
draw_polys(colors, trans_triangulation, screen, options.lines)
# Refresh the display
pygame.display.flip()
# Write the image to a file
pygame.image.save(screen, options.filename)
print 'Image saved to %s' % options.filename
sys.exit(0)
