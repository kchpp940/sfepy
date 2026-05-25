from sfepy._extmods import ExtensionImportError


try:
    from .extmods import *
except (ImportError, AttributeError) as exc:
    # Translate the low-level error produced by our own ``ExtensionImportError`` into a
    # clearer, single message that points the top-level scripts can surface without
    # misleading the user into thinking the pure-Python source tree itself is broken.
    if not isinstance(exc, ExtensionImportError):
        # Preserve the original exception but add a short hint that the compiled
        # extensions are the likely culprit.
        print('sfepy extension modules may not be compiled!\n'
              'Try typing "make".')
    raise

from .mesh import Mesh
from .domain import FEDomain
from .fields_base import Field
from sfepy.discrete.fem.meshio import MeshIO
from .utils import extend_cell_data
