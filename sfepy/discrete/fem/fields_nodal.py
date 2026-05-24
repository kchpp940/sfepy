"""
Notes
-----

Important attributes of continuous (order > 0) :class:`Field` and
:class:`SurfaceField` instances:

- `vertex_remap` : `econn[:, :n_vertex] = vertex_remap[conn]`
- `vertex_remap_i` : `conn = vertex_remap_i[econn[:, :n_vertex]]`

where `conn` is the mesh vertex connectivity, `econn` is the
region-local field connectivity.
"""
import numpy as nm

from sfepy.base.base import assert_, Struct
from sfepy.discrete.integrals import Integral
from sfepy.discrete.fem.utils import prepare_remap
from sfepy.discrete.common.dof_info import expand_nodes_to_dofs
from sfepy.discrete.common.mappings import get_physical_qps
from sfepy.discrete.fem.facets import get_facet_dof_permutations
from sfepy.discrete.fem.fields_base import FEField, H1Mixin
from sfepy.discrete.fem.dof_builder import (
    NodalOrientationStrategy, NodalSubstitutionStrategy,
)

class GlobalNodalLikeBasis(Struct):

    _orientation_strategy_cls = NodalOrientationStrategy
    _substitution_strategy_cls = NodalSubstitutionStrategy

    def _setup_facet_orientations(self):
        self.dof_builder.setup_facet_orientations()

    def _setup_edge_dofs(self):
        """
        Setup edge DOF connectivity.
        """
        if self.node_desc.edge is None:
            return 0, None, None

        return self.dof_builder.setup_facet_dofs(1, self.node_desc.edge,
                                                 self.n_vertex_dof)

    def _setup_face_dofs(self):
        """
        Setup face DOF connectivity.
        """
        if self.node_desc.face is None:
            return 0, None, None

        return self.dof_builder.setup_facet_dofs(
            self.domain.shape.tdim - 1,
            self.node_desc.face,
            self.n_vertex_dof + self.n_edge_dof)

    def _setup_facet_dofs(self, dim, facet_desc, facet_perms, offset):
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

    def get_surface_basis(self, region):
        """
        Get basis for projections to region's facets.

        Notes
        -----
        Cannot be uses for all fields because IGA does not support surface
        mappings.
        """
        order = self.approx_order

        integral = Integral('i', order=2*order)
        geo, mapping = self.get_mapping(region, integral, 'surface')
        pqps = get_physical_qps(region, integral)
        qps = pqps.values.reshape(pqps.shape)

        bfs = nm.broadcast_to(
            geo.bf[..., 0, :],
            (qps.shape[0], qps.shape[1], geo.bf.shape[3]),
        )

        return qps, bfs, geo.det[..., 0]

class H1NodalMixin(H1Mixin, GlobalNodalLikeBasis):

    def _substitute_dofs(self, subs):
        """
        Perform facet DOF substitutions according to `subs`.

        Modifies `self.econn` in-place.

        Delegates to :class:`DofConnectivityBuilder` via the field's
        substitution strategy.
        """
        self.dof_builder.substitute_dofs(subs)

    def _eval_basis_transform(self, subs):
        """
        Evaluate the basis transformation matrix for substituted DOFs.

        Delegates to :class:`DofConnectivityBuilder` via the field's
        substitution strategy.
        """
        return self.dof_builder.eval_basis_transform(subs)

    def set_dofs(self, fun=0.0, region=None, dpn=None, warn=None):
        """
        Set the values of DOFs in a given `region` using a function of space
        coordinates or value `fun`.
        """
        if region is None:
            region = self.region

        if dpn is None:
            dpn = self.n_components

        aux = self.get_dofs_in_region(region)
        nods = nm.unique(aux)

        if callable(fun):
            coors = self.get_coor(nods)
            vals = nm.asarray(fun(coors))
            if (vals.ndim > 1) and (vals.shape != (len(coors), dpn)):
                raise ValueError('The projected function return value should be'
                                 ' (n_point, dpn) == %s, instead of %s!'
                                 % ((len(coors), dpn), vals.shape))

        elif nm.isscalar(fun):
            vals = nm.full(nods.shape[0] * dpn, fun, dtype=nm.dtype(type(fun)))

        elif isinstance(fun, nm.ndarray):
            try:
                assert_(len(fun) == dpn)

            except (TypeError, ValueError):
                msg = ('wrong array value shape for setting'
                       ' DOFs of "%s" field!'
                       ' (shape %s should be %s)'
                       % (self.name, fun.shape, (dpn,)))
                raise ValueError(msg)

            vals = nm.tile(fun, nods.shape[0])

        else:
            raise ValueError('unknown function/value type! (%s)' % type(fun))

        vals.shape = (len(nods), -1)

        if not nm.isfinite(vals).all():
            raise ValueError(f'infs or nans in DOF values set with {fun}!')

        return nods, vals

    def create_basis_context(self):
        """
        Create the context required for evaluating the field basis.
        """
        ps = self.poly_space
        gps = self.gel.poly_space

        mesh = self.create_mesh(extra_nodes=False)

        ctx = ps.create_context(None, 0, 1e-15, 100, 1e-8,
                                tdim=mesh.cmesh.tdim)
        geo_ctx = gps.create_context(mesh.cmesh, 0, 1e-15, 100, 1e-8)

        ctx.geo_ctx = geo_ctx

        return ctx

class H1NodalVolumeField(H1NodalMixin, FEField):
    """
    Lagrange basis nodal approximation.
    """
    family_name = 'volume_H1_lagrange'

    def interp_v_vals_to_n_vals(self, vec):
        """
        Interpolate a function defined by vertex DOF values using the FE
        geometry basis (P1 or Q1) into the extra nodes, i.e. define the
        extra DOF values.
        """
        if not self.node_desc.has_extra_nodes():
            enod_vol_val = vec.copy()

        else:
            dim = vec.shape[1]
            enod_vol_val = nm.zeros((self.n_nod, dim), dtype=nm.float64)

            coors = self.poly_space.node_coors

            bf = self.gel.poly_space.eval_basis(coors)
            bf = bf[:,0,:].copy()

            conn = self.econn[:, :self.gel.n_vertex]

            evec = nm.dot(bf, vec[conn])
            enod_vol_val[self.econn] = nm.swapaxes(evec, 0, 1)

        return enod_vol_val

class H1SNodalVolumeField(H1NodalVolumeField):
    """
    Lagrange basis nodal serendipity approximation with order <= 3.
    """
    family_name = 'volume_H1_serendipity'

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

class H1SEMVolumeField(H1NodalVolumeField):
    """
    Spectral element method approximation.

    Uses the Lagrange basis with Legendre-Gauss-Lobatto nodes and quadrature.
    """
    family_name = 'volume_H1_sem'

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

class H1DiscontinuousField(H1NodalMixin, FEField):
    """
    The C0 constant-per-cell approximation.
    """
    family_name = 'volume_H1_lagrange_discontinuous'

    def _setup_global_basis(self):
        """
        Setup global DOF/basis function indices and connectivity of the field.
        """
        self._setup_facet_orientations()

        self._init_econn()

        ii = self.region.get_cells()
        self.bubble_remap = prepare_remap(ii, self.cmesh.n_el)

        n_dof = int(nm.prod(self.econn.shape))
        all_dofs = nm.arange(n_dof, dtype=nm.int32)
        all_dofs.shape = self.econn.shape

        self.econn[:] = all_dofs

        self.n_nod = n_dof

        self.n_bubble_dof = n_dof
        self.bubble_dofs = all_dofs

        self.n_vertex_dof = self.n_edge_dof = self.n_face_dof = 0

        self._setup_esurface()

    def extend_dofs(self, dofs, fill_value=None):
        """
        Extend DOFs to the whole domain using the `fill_value`, or the
        smallest value in `dofs` if `fill_value` is None.
        """
        if self.approx_order != 0:
            dofs = self.average_to_vertices(dofs)

        new_dofs = FEField.extend_dofs(self, dofs)

        return new_dofs

    def remove_extra_dofs(self, dofs):
        """
        Remove DOFs defined in higher order nodes (order > 1).
        """
        if self.approx_order != 0:
            dofs = self.average_to_vertices(dofs)

        new_dofs = FEField.remove_extra_dofs(self, dofs)

        return new_dofs

    def average_to_vertices(self, dofs):
        """
        Average DOFs of the discontinuous field into the field region
        vertices.
        """
        data_qp, integral = self.interp_to_qp(dofs)
        vertex_dofs = self.average_qp_to_vertices(data_qp, integral)

        return vertex_dofs

class H1NodalSurfaceField(H1NodalMixin, FEField):
    """
    A field defined on a surface region.
    """
    family_name = 'surface_H1_lagrange'

    def interp_v_vals_to_n_vals(self, vec):
        """
        Interpolate a function defined by vertex DOF values using the FE
        surface geometry basis (P1 or Q1) into the extra nodes, i.e. define the
        extra DOF values.
        """
        if not self.node_desc.has_extra_nodes():
            enod_vol_val = vec.copy()

        else:
            msg = 'surface nodal fields do not support higher order nodes yet!'
            raise NotImplementedError(msg)

        return enod_vol_val

class H1SNodalSurfaceField(H1NodalSurfaceField):
    family_name = 'surface_H1_serendipity'

class H1SEMSurfaceField(H1NodalSurfaceField):
    family_name = 'surface_H1_sem'
