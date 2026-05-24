import numpy as nm

from sfepy.discrete.fem.fields_base import FEField, H1Mixin
from sfepy.discrete.fem.dof_builder import (
    HierarchicOrientationStrategy, NoSubstitutionStrategy,
)

class H1HierarchicVolumeField(H1Mixin, FEField):
    """
    Hierarchical basis approximation with Lobatto polynomials.
    """
    family_name = 'volume_H1_lobatto'

    _orientation_strategy_cls = HierarchicOrientationStrategy
    _substitution_strategy_cls = NoSubstitutionStrategy

    def _init_econn(self):
        """
        Initialize the extended DOF connectivity and facet orientation array.

        Kept for backward compatibility with external subclasses that may
        override this method.  The default implementation simply delegates
        to :class:`DofConnectivityBuilder`.
        """
        FEField._init_econn(self)
        self.ori = nm.zeros_like(self.econn)

    def _setup_facet_orientations(self):
        self.dof_builder.setup_facet_orientations()

    def _setup_edge_dofs(self):
        """
        Setup edge DOF connectivity.
        """
        if self.node_desc.edge is None:
            return 0, None, None

        return self.dof_builder.setup_facet_dofs(1,
                                                 self.node_desc.edge,
                                                 self.n_vertex_dof)

    def _setup_face_dofs(self):
        """
        Setup face DOF connectivity.
        """
        if self.node_desc.face is None:
            return 0, None, None

        return self.dof_builder.setup_facet_dofs(self.domain.shape.tdim - 1,
                                                 self.node_desc.face,
                                                 self.n_vertex_dof + self.n_edge_dof)

    def _setup_facet_dofs(self, dim, facet_desc, offset):
        """
        Helper function to setup facet DOF connectivity, works for both
        edges and faces.

        Kept for backward compatibility with external subclasses that may
        override this method.  The default implementation simply delegates
        to :class:`DofConnectivityBuilder`.
        """
        return self.dof_builder.setup_facet_dofs(dim, facet_desc, offset)

    def _setup_bubble_dofs(self):
        """
        Setup bubble DOF connectivity.
        """
        return self.dof_builder.setup_bubble_dofs()

    def set_dofs(self, fun=0.0, region=None, dpn=None, warn=None):
        """
        Set the values of DOFs in a given `region` using a function of space
        coordinates or value `fun`.
        """
        if region is None:
            region = self.region

        if dpn is None:
            dpn = self.n_components

        # Hack - use only vertex DOFs.
        gnods = self.get_dofs_in_region(region, merge=False)
        nods = nm.concatenate(gnods)
        n_dof = dpn * nods.shape[0]

        if nm.isscalar(fun):
            vals = nm.zeros(n_dof, dtype=nm.dtype(type(fun)))
            vals[:gnods[0].shape[0] * dpn] = fun

        elif callable(fun):
            coors = self.get_coor(gnods[0])
            vv = nm.asarray(fun(coors))
            if (vv.ndim > 1) and (vv.shape != (len(coors), dpn)):
                raise ValueError('The projected function return value should be'
                                 ' (n_point, dpn) == %s, instead of %s!'
                                 % ((len(coors), dpn), vv.shape))

            vals = nm.zeros(n_dof, dtype=vv.dtype)
            vals[:gnods[0].shape[0] * dpn] = vv.ravel()

        else:
            raise ValueError('unknown function/value type! (%s)' % type(fun))

        nods, indx = nm.unique(nods, return_index=True)
        ii = (nm.tile(dpn * indx, dpn)
              + nm.tile(nm.arange(dpn, dtype=nm.int32), indx.shape[0]))
        vals = vals[ii]

        vals.shape = (len(nods), -1)

        if not nm.isfinite(vals).all():
            raise ValueError(f'infs or nans in DOF values set with {fun}!')

        return nods, vals

    def create_basis_context(self):
        """
        Create the context required for evaluating the field basis.
        """
        # Hack for tests to pass - the reference coordinates are determined
        # from vertices only - we can use the Lagrange basis context for the
        # moment. The true context for Field.evaluate_at() is not implemented.
        gps = self.gel.poly_space
        mesh = self.create_mesh(extra_nodes=False)

        ctx = geo_ctx = gps.create_context(self.cmesh, 0, 1e-15, 100, 1e-8)
        ctx.geo_ctx = geo_ctx

        return ctx
