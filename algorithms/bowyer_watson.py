"""
Functions related to calculating the Delaunay triangulation of a set of points using the Bowyer-Watson algorithm

This module exports the following:
Functions:
    triangulate(p): find the Delaunay triangulation of the points p
"""
from geometry import (
    Triangle,
    enclosing_triangle,
    scale_tri,
    tri_circumcircle,
    triangle_from_edge_point,
    vertices_to_edges,
)


def triangulate(points):
    """
    Calculate the Delaunay triangulation of a set of points. May raise ValueError.

    Currently using the Bowyer-Watson algorithm but might switch to something
    faster in the future. For the current algorithm, we pre-calculate all of
    the circumcircles of the triangles to speed things up.

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
        return [Triangle(points[0], points[1], points[2])]

    # Find a "supertriangle" that encloses all of the points
    supertriangle = enclosing_triangle(points)
    if supertriangle is None:
        return None
    supertriangle = scale_tri(supertriangle, 2)
    # The graph is a list of 2-tuples of the form (t, c), where t is a Triangle
    # object and c is its circumcircle. Precalculating the circumcircles saves
    # considerable time later on.
    graph = {(supertriangle, tri_circumcircle(supertriangle))}

    # Add points to the graph one at a time
    for p in points:
        # Find the triangles that contain the point
        invalid_triangles_vertices = []
        invalid_triangles_edges = []
        for t, circumcircle in graph:
            if (circumcircle.center.x-p.x)**2+(circumcircle.center.y-p.y)**2 <= circumcircle.radius_squared:
                invalid_triangles_edges.append(vertices_to_edges(t))
                invalid_triangles_vertices.append((t, circumcircle))  # Keep the circumcircle so we can remove the item later
        # There is a polygonal hole around the new point. Find its edges
        hole = []
        for t in invalid_triangles_edges:
            for e in t:
                # Make sure the edge is unique and add it to the hole if it is
                unique = True
                for u in invalid_triangles_edges:
                    if u != t and (e in u or tuple(reversed(e)) in u):
                        unique = False
                        break

                if unique:
                    hole.append(e)

        # Delete the invalid triangles from the graph
        for pair in invalid_triangles_vertices:
            graph.remove(pair)

        # Re-triangulate the hole made by the new point
        for e in hole:
            if p not in e:
                t = triangle_from_edge_point(e, p)
                graph.add((t, tri_circumcircle(t)))

    # Prune out any triangles that have a vertex in common with the supertriangle and remove the extra circumcircle info
    pruned_graph = filter(
        lambda t: not (t.a in supertriangle or t.b in supertriangle or t.c in supertriangle),
        (t for t, _ in graph)
    )
    return list(pruned_graph)

