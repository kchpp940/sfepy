from sfepy.base.deps import dep_manager

# Import (or verify) the compiled C helpers through the central
# registry, so that the error message always points the user to the
# right install instructions when the extension modules have not been
# built yet.
try:
    from .extmods import *  # noqa: F401,F403
    from .mesh import Mesh
    from .domain import FEDomain
    from .fields_base import Field
    from sfepy.discrete.fem.meshio import MeshIO
    from .utils import extend_cell_data
except (ImportError, AttributeError) as exc:
    dep_manager.require(
        'c-ext-fem',
        context='sfepy.discrete.fem import failed: %s' % exc,
    )
