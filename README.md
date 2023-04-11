# tcore
Library of my helper functions used as a submodule in various projects. Includes generators for numerical quadratures, generators for basis functions for finite element methods, time integration methods, and some OpenGL helpers for rendering and GPGPU.

Some plots created using Python bindings of this library follow.

Lagrange bases can be used to interpolate points, i.e. each contains a point equal to 1.0 where all other bases equal to 0.0 for a given degree containing degree + 1 basis functions.
![plot](./Lagrange7.png)

Legendre bases are a set of independent basis functions containing perpendicular polynomials of increasing degree.
![plot](./Legendre7.png)

Bernstein bases are used in Bézier curves, where the curve contains exactly the first and last control point.
![plot](./Bernstein7.png)
