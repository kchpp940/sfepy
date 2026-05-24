from sfepy.base.deps import dep_manager

# Probe the compiled C helpers through the central registry BEFORE
# importing any module that depends on them.  This avoids a partially
# initialised ``sfepy.discrete`` being left in ``sys.modules`` when
# one of the downstream imports fails.
try:
    from sfepy.discrete.common import extmods as _common_extmods  # noqa: F401
    from sfepy.discrete.fem import extmods as _fem_extmods  # noqa: F401
    from sfepy.discrete.iga import extmods as _iga_extmods  # noqa: F401
    from sfepy.terms import extmods as _terms_extmods  # noqa: F401
except (ImportError, AttributeError) as exc:
    # Raise a descriptive DependencyMissingError through the central
    # registry - whichever extension backend is reported missing by
    # the first probe is the one surfaced to the user.
    for _name in ('c-ext-common', 'c-ext-fem', 'c-ext-iga',
                  'c-ext-terms'):
        if not dep_manager.available(_name):
            dep_manager.require(
                _name,
                context='sfepy.discrete import failed: %s' % exc,
            )
    dep_manager.require(
        'c-ext-fem',
        context='sfepy.discrete import failed: %s' % exc,
    )

from sfepy.discrete.common.domain import Domain
from sfepy.discrete.common.region import Region
from sfepy.discrete.common.fields import Field
from sfepy.discrete.common.poly_spaces import PolySpace

from .functions import Functions, Function
from .conditions import Conditions
from .variables import (Variables, Variable, FieldVariable, DGFieldVariable,
                        create_adof_conns)
from .materials import Materials, Material
from .equations import Equations, Equation
from .integrals import Integrals, Integral
from .problem import Problem
from .evaluate import assemble_by_blocks
