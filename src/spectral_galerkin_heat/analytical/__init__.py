"""Closed-form reference solutions for a laser moving on a cuboid.

Both solutions assume constant properties and a laser moving at constant
velocity along x on the top face ``z = Lz``, and add image sources so the field
satisfies Neumann boundary conditions.

* :func:`eagar_tsai_field`: transient Gaussian source (Eagar and Tsai, 1983).
* :func:`rosenthal_field`: quasi-steady point source (Rosenthal, 1946).

Both return a ``(z, y, x)`` array evaluated on the 1-D coordinate arrays given.
"""

from .eagar_tsai import eagar_tsai_field
from .rosenthal import rosenthal_field

__all__ = ["eagar_tsai_field", "rosenthal_field"]
