# Calculate the Delaunay triangulation of a set of points
from __future__ import division
from math import sqrt, atan2

# Find the midpoint of a line segment
#	line is a 2-tuple of x,y coordinates, e.g. ((x1,y1),(x2,y2))
#	Returns an x,y coordinate pair
def midpoint(line):
	return ((line[1][0]+line[0][0])/2, (line[1][1]+line[0][1])/2)

# Find the slope of a line segment
#	line is a 2-tuple of x,y coordinates, e.g. ((x1,y1),(x2,y2))
#	Returns the slope of the line
def slope(line):
	try:
		return (line[1][1] - line[0][1])/(line[1][0] - line[0][0])
	except:
		# Raise an error if the input isn't a line segment
		if len(line) != 2 or len(line[0]) != 2 or len(line[1]) != 2:
			raise ValueError('Input is not a line segment ((x1, y1), (x2, y2))')
		# Raise an error if both points are the same
		if line[0] == line[1]:
			raise ValueError('Both points are the same')
		# Raise an error if dx = 0
		if line[0][0] == line[1][0]:
			raise ValueError('Line has infinite slope')

# Find the slope of a line perpendicular to a line segment
#	line is a 2-tuple of x,y coordinates, e.g. ((x1,y1),(x2,y2))
#	Returns the slope of the perpendicular line
def perp_slope(line):
	try:
		# Perpendicular slppe is the negative reciprocal of the slope, i.e. -dx/dy
		return -1*(line[1][0] - line[0][0])/(line[1][1] - line[0][1])
	except:
		# Raise an error if the input isn't a line segment
		if len(line) != 2 or len(line[0]) != 2 or len(line[1]) != 2:
			raise ValueError('Input is not a line segment ((x1, y1), (x2, y2))')
		# Raise an error if both points are the same
		if line[0] == line[1]:
			raise ValueError('Both points are the same')
		# Raise an error if dy = 0
		if line[0][1] == line[1][1]:
			raise ValueError('Line has zero slope')

# Convert a line from point-slope form to y-intercept form
#	m is the slope of the line
#	p is an x,y coordinate on the line
#	Returns a 2-tuple containing the slope of the line and the y-intercept, e.g. (m, b)
def point_slope_to_y_intercept(m, p):
	# b = y - mx
	return (m, p[1] - m*p[0])

# Determine whether a line is vertical
#	l is a line represented by two x,y coordinates
def is_vertical(l):
	# The line is vertical if dx = 0
	return l[0][0] == l[1][0]

# Determine whether a line is horizontal
#	l is a line represented by two x,y coordinates
def is_horizontal(l):
	# The line is horizontal if dy = 0
	return l[0][1] == l[1][1]

# Find the intersection of two lines
#	a is a line defined by its slope and y-intercept, e.g. (m, b)
#	b is a line defined the same way as a
#	Returns an x,y coordinate pair
def lines_intersection(a, b):
	try:
		x = (b[1] - a[1])/(a[0] - b[0])
		y = a[0] * x + a[1]
		return (x,y)
	except:
		# Lines are either parallel or invalid. Either way, the intersection doesn't exist
		return None

# Find the intersection of a line with a vertical line
#	a is a line defined by its slope and y-intercept, e.g. (m, b)
#	p is an x,y coordinate pair that the vertical line passes through
#	Returns the point where a crosses the line through p
def line_intersect_vertical(a, p):
	x = p[0]
	y = a[0] * x + a[1]
	return (x,y)

# Calculate the vertices of a triangle given three line segments along its sides
#	a is a line segment represented by a 2-tuple of x,y coordinates, i.e. ((x1,y1), (x2,y2))
#	b and c are represented the same way
#	Returns a tuple of three x,y coordinate pairs; the vertices of the triangle
def calculate_tri_vertices(side_a, side_b, side_c):
	# Calculate slopes and y-intercepts of the sides
	if is_vertical(side_a):
		m_a = None
	else:
		m_a = slope(side_a)
		b_a = side_a[1][1] - m_a * side_a[1][0]
	
	if is_vertical(side_b):
		m_b = None
	else:
		m_b = slope(side_b)
		b_b = side_b[1][1] - m_b * side_b[1][0]

	if is_vertical(side_c):
		m_c = None
	else:
		m_c = slope(side_c)
		b_c = side_c[1][1] - m_c * side_c[1][0]

	# If any two sides are parallel then this is not a valid triangle
	if m_a == m_b or m_a == m_c or m_b == m_c:
		return None
	# Calculate the vertices
	# If one of the sides is vertical, we have a special case
	# Vertical A
	if not m_a:
		a_x = side_a[0][0]
		a_y = m_b*a_x + b_b

		b = lines_intersection((m_b, b_b), (m_c, b_c))

		c_x = side_a[0][0]
		c_y = m_c*c_x + b_c
		return ((a_x, a_y), b, (c_x, c_y))
	# Vertical B
	elif not m_b:
		a_x = side_b[0][0]
		a_y = m_b*a_x + b_b

		b_x = side_b[0][0]
		b_y = m_c*b_x + b_c

		c = lines_intersection((m_c, b_c), (m_a, b_a))
		return ((a_x, a_y), (b_x, b_y), c)
	# Vertical C
	elif not m_c:
		a = lines_intersection((m_a, b_a), (m_b, b_b))

		b_x = side_c[0][0]
		b_y = m_b*b_x + b_b

		c_x = side_c[0][0]
		c_y = m_a*c_x + b_a
		return (a, (b_x, b_y), (c_x, c_y))		
	
	# We may encounter a division by zero error if the slopes are too close
	try:
		a = lines_intersection((m_a, b_a), (m_b, b_b))
		b = lines_intersection((m_b, b_b), (m_c, b_c))
		c = lines_intersection((m_c, b_c), (m_a, b_a))
		return (a, b, c)

	except ZeroDivisionError:
		return None

# Calculate the vertices of a triangle defined by the given edge and point
#	edge is a pair of x,y coordinates, e.g. ((x1,y1),(x2,y2))
#	point is an x,y coordinate pair, e.g. (x,y)
#	Returns a vertex-defined triangle where each vertex is an x,y coordinate
def triangle_from_edge_point(edge, point):
	return (edge[0], edge[1], point)

# Convert a vertex definition of a triangle to an edge definition
#	t is a triangle defined by three vertices, each of which is an x,y coordinate pair, e.g. ((x1,y1),(x2,y2),(x3,y3))
#	Returns a triangle defined by three edges consisting of a pair of x,y coordinates, e.g. (((x1,y2),(x2,y2)), ((x2,y2),(x3,y3)), etc)
def vertices_to_edges(t):
	return (
		(t[-1], t[0]),
		(t[0], t[1]),
		(t[1], t[2])
	)

# Convert an edge definition of a triangle to a vertex definition
#	t is a triangle defined by three edges, each of which is a pair of x,y coordinates
#	Returns a triangle defined by three x,y coordinate vertices
def edges_to_vertices(t):
	return (t[0][1], t[1][1], t[2][1])

# Determine whether a circle contains a given point
#	circle is a 2-tuple containing the center of the circle (x,y) and its radius
#	p is an x,y coordinate pair
#	Returns True if p lies within the circle and false otherwise
def circle_contains_point(circle, p):
	#return distance(circle[0], p) <= circle[1]
	return sqrt((circle[0][0]-p[0])**2 + (circle[0][1]-p[1])**2) <= circle[1]

# Determine whether the given triangle contains the given point
#	t is a triangle defined by three pairs of x,y coordinates
#	p is a point represented by a pair of x,y coordinates
def tri_contains_point(t, p):
	# Error within 1ppm is acceptable
	epsilon = 1e-6
	# Calcula1te the barycentric coordinates of p
	p1 = t[0]
	p2 = t[1]
	p3 = t[2]
	# Make sure the point isn't a vertex
	if p == p1 or p == p2 or p == p3:
		return True
	denom = (p2[1] - p3[1])*(p1[0] - p3[0]) + (p3[0] - p2[0])*(p1[1] - p3[1])
	if denom != 0:
		alpha = ((p2[1] - p3[1])*(p[0] - p3[0]) + (p3[0] - p2[0])*(p[1] - p3[1]))/denom
		beta = ((p3[1] - p1[1])*(p[0] - p3[0]) + (p1[0] -p3[0])*(p[1] - p3[1]))/denom
		gamma = 1.0 - alpha - beta
		# If all three coordinates are positive, p lies within t
		return alpha+epsilon >= 0 and beta+epsilon >= 0 and gamma+epsilon >= 0
	# Invalid triangle
	else:
		return False

# Calculate the circumcenter of a triangle
#	t is a triangle represented by a 3-tuple of x,y coordinates
#	Returns an x,y coordiante pair describing the circumcenter of the triangle
def tri_circumcenter(t):
	# The circumcenter of the triangle is the point where the perpendicular bisectors of the sides intersect
	# Define the sides we care about
	A = (t[0],t[1])
	B = (t[1],t[2])

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

# Calculate the centroid of a triangle
#	t is a triangle represented by a 3-tuple of x,y coordinates
#	Returns an x,y coordiante pair describing the centroid of the triangle
def tri_centroid(t):
	# Centroid is just the mean of the vertices
	return ( ((t[0][0] + t[1][0] + t[2][0])/3), ((t[0][1] + t[1][1] + t[2][1])/3) )

# Calculate the circumcircle of a triangle
#	t is a triangle represented by a 3-tuple of x,y coordinates
#	Returns the circumcircle of tri represented by a 2-tuple consisting of the center and radius ((x,y), r)
def tri_circumcircle(t):
	# Get the circumcenter of the triangle
	center = tri_circumcenter(t)
	# Define the sides
	A = (t[0],t[1])
	B = (t[1],t[2])
	C = (t[2],t[0])
	# The radius of the circumcircle is given by this formula of the lengths of the sides
	'''
	a = distance(A[0], A[1])
	b = distance(B[0], B[1])
	c = distance(C[0], C[1])
	'''
	a = sqrt((A[0][0]-A[1][0])**2 + (A[0][1]-A[1][1])**2)
	b = sqrt((B[0][0]-B[1][0])**2 + (B[0][1]-B[1][1])**2)
	c = sqrt((C[0][0]-C[1][0])**2 + (C[0][1]-C[1][1])**2)

	radius = (a*b*c)/sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))
	return (center, radius)

# Determine whether the circumcircle of the given triangle contains the given point
#	t is a triangle defined by three pairs of x,y coordinates
#	p is a point represented by a pair of x,y coordinates
def tri_circle_contains_point(t, p):
	# Get the circumcircle of the triangle
	circle = tri_circumcircle(t)
	# Determine whether the point is within the circle
	return circle_contains_point(circle, p)

# Determine whether two triangles have any vertices in common
#	t1 and t2 are two vertex-defined triangles
#	Returns true if any vertices are shared and false otherwise
def tri_share_vertices(t1, t2):
	# Iterate over the vertices in t1 and compare them to every vertex in t2
	for vertex1 in t1:
		for vertex2 in t2:
			if vertex1 == vertex2:
				return True
	# No vertices matched
	return False

# Calculate the angle between two points
#	a is a point represented by a pair of x,y coordinates
#	b is also a point
#	Returns the angle between a and b in radians
def angle(a, b):
	return atan2(b[1]-a[1], b[0]-a[0])

# Calculates the cross product of three points
#	a is a point represented by a pair of x,y coordinates
#	b is also a point
#	c is also a point
#	Returns the cross product of a, b, and c
def cross_product(a, b, c):
	return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])

# Translate a triangle in x and y
#	t is a triangle represented by a 3-tuple of x,y coordinates
#	d is a 2-tuple describing the translations in x and y (x, y)
#	Returns a new triangle translated by d
def translate_tri(t, d):
	trans_t = []
	for point in t:
		trans_t.append((point[0]+d[0], point[1]+d[1]))
	return tuple(trans_t)

# Scale a triangle from its center point by the given scale factor
#	t is a triangle represented by a 3-tuple of x,y coordinates
#	s is the scale factor
#	Returns a new triangle scaled by s
def scale_tri(t, s):
	centroid = tri_centroid(t)
	# Translate the vertices of the triangle as if the centroid were the origin
	trans_t = translate_tri(t, (centroid[0]*-1, centroid[1]*-1))
	# Multiply all of the vertices by the scale factor
	scaled_t = []
	for vertex in trans_t:
		scaled_t.append((vertex[0]*s, vertex[1]*s))
	# Translate the triangle back
	scaled_t = translate_tri(scaled_t, centroid)
	return tuple(scaled_t)


# Calculate the convex hull of a set of points
#	points is a list of 2-tuples of x,y coordinates
#	Returns the convex hull as a list of points presented by 2-tuples of x,y coordinate pairs
#		i.e. h = [(x1,y1), (x2,y2), etc] .
def convex_hull(points):
	# Find the point with the lowest y-coordinate
	# If there's a tie, the one with the lowest x-coordinate is chosen
	min_point = None
	min_i = None
	i = -1
	for p in points:
		i += 1
		if not min_point or p[1] <= min_point[1]:
			if min_point and p[1] == min_point[1]:
				if p[0] < min_point[0]:
					min_point = p
					min_i = i
			else:
				min_point = p
				min_i = i
	points_copy = points[::1]
	del points_copy[min_i]
	# Next, sort the points by angle (asc) relative to the minimum point
	spoints = [min_point] + sorted(points_copy, lambda x,y: cmp(angle(min_point, x), angle(min_point, y)))
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

# Calculate a triangle that encloses a set of points -- note that the triangle may not contain any points in the given set
#	points is a list of 2-tuples of x,y coordinates
#	Returns a triangle represented as a triplet of points, which are pairs of x,y coordinates
#		i.e. t = ((x1,y1), (x2,y2), (x3,y3))
def enclosing_triangle(points):
	# Find the convex hull that encloses the points
	hull = convex_hull(points)
	# If the hull only has three points, this is our triangle
	if len(hull) == 3:
		return tuple(hull)
	# Convert the hull from a list of points to a list of edges
	edges = []
	for p in range(0,len(hull)):
		edges.append((hull[p-1], hull[p]))
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
	print 'Failed to find enclosing triangle for edges', edges
	return None

# Calculate the Delaunay triangulation of a set of points using the Bowyer-Watson algorithm
#	points is a list of 2-tuples of x,y coordinates
#	Returns a list of triangles represented as triplets of x,y coordinates (i.e. t = ( ((x11,y11), (x12,y12), (x13,y13)), ((x21,y21), ...) ))
def calculate_triangles(points):
	# Less than three points is impossible to triangulate
	if len(points) < 3:
		raise ValueError('Need at least three points to triangulate')
	# Three points is the simplest case
	if len(points) == 3:
		return tuple(points)
	# Start with a triangle large enough to contain all of the points
	scale_factor = 1.5
	supertriangle = scale_tri(enclosing_triangle(points), scale_factor)
	if not supertriangle:
		return None
	# The graph is a list of 2-tuples; the first element of each 2-tuple is a triangle and the second element is its circumcircle.
	# This saves us considerable time in recalculating the circles
	graph = [(supertriangle, tri_circumcircle(supertriangle))]
	# Add points to the graph one at a time
	for p in points:
		# Find the triangles that contain the point
		invalid_triangles_vertices = []
		invalid_triangles_edges = []
		for pair in graph:
			t = pair[0]
			c = pair[1]
			#if circle_contains_point(pair[1], p):
			if sqrt((c[0][0]-p[0])**2+(c[0][1]-p[1])**2) <= c[1]:
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

