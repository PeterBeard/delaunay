# delaunay
Python script for calculating and drawing triangular meshes. The mesh is the Delaunay triangulation of a random assortment of points. For more details on how this works, see https://www.peterbeard.co/blog/post/delaunay-triangulation-makes-pretty-pictures/

Dependencies
------------
* Uses the Python Imaging Library (PIL) to read/write image files

Usage
-----
```
Usage: delaunay.py [options]

Options:
  -h, --help            show this help message and exit
  -o FILENAME, --output=FILENAME
                        The filename to write the image to. Supported
                        filetypes are BMP, TGA, PNG, and JPEG
  -n N_POINTS, --npoints=N_POINTS
                        The number of points to use when generating the
                        triangulation.
  -x WIDTH, --width=WIDTH
                        The width of the image.
  -y HEIGHT, --height=HEIGHT
                        The height of the image.
  -g GRADIENT, --gradient=GRADIENT
                        The name of the gradient to use.
  -i IMAGE, --image-file=IMAGE
                        An image file to use when calculating triangle colors.
                        Image dimensions will override dimensions set by -x
                        and -y.
  -k DARKEN_AMOUNT, --darken=DARKEN_AMOUNT
                        Darken random triangles my the given amount to make
                        the pattern stand out more
  -a, --antialias       If enabled, draw the image at 4x resolution and
                        downsample to reduce aliasing.
  -l, --lines           If enabled, draw lines along the triangle edges.
  -d, --decluster       If enabled, try to avoid generating clusters of points
                        in the triangulation. This will significantly slow
                        down point generation.
  -r, --right           If enabled, generate right triangles rather than
                        random ones.
  -e, --equilateral     If enabled, generate equilateral triangles rather than
                        random ones.
```
