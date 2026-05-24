"""
DOF connectivity builder with pluggable orientation and substitution strategies.

Centralizes the construction of DOF connectivity for finite element fields,
separating the common bookkeeping (facet traversal, remap computation) from
field-type-specific orientation handling and hanging-node substitution rules.

Orientation strategies
----------------------
- :class:`NodalOrientationStrategy` — for Lagrange (nodal) fields, uses
  precomputed DOF permutation tables.
- :class:`HierarchicOrientationStrategy` — for Lobatto (hierarchic) fields,
  encodes orientation as sign bits in ``field.ori``.

Substitution strategies
----------------------
- :class:`NodalSubstitutionStrategy` — hanging-node DOF substitution for
  2_4 (quad) and 3_8 (hex) reference elements.
- :class:`NoSubstitutionStrategy` — no-op, raises ``NotImplementedError``
  on attempted substitution.

Surface econn helpers
---------------------
- :func:`compute_surface_econn` — extract face connectivity from a volume
  field's ``econn``.
- :func:`compute_phantom_econn` — use full cell connectivity as a phantom
  surface.
- :func:`finish_surface` — common post-processing (``leconn``, ``bkey``,
  ``face_type``, orientation map).

Main builder
------------
- :class:`DofConnectivityBuilder` — composes all of the above and is the
  single entry point called from :class:`FEField`.
"""

import numpy as nm

from sfepy.base.base import Struct, assert_
from sfepy.discrete.fem.utils import prepare_remap, prepare_translate
from sfepy.discrete.common.dof_info import expand_nodes_to_dofs
from sfepy.discrete.fem.facets import get_facet_dof_permutations

# =========================================================================
# Orientation strategies
# =========================================================================

class NodalOrientationStrategy:
    """Facet orientation handling for Lagrange (nodal) fields.

    Uses precomputed DOF permutation tables to reorder facet DOFs according
    to cmesh orientation flags.  The permutations are stored as
    ``field.edge_dof_perms`` and ``field.face_dof_perms``.
    """

    def init_econn(self, field):
        """No extra state required for nodal orientation."""

    def setup(self, field):
        """Build DOF permutation tables from the poly_space description."""
        order = field.approx_order
        node_desc = field.node_desc

        if node_desc.edge_nodes is not None:
            n_fp = field.gel.edges.shape[1]
            field.edge_dof_perms = get_facet_dof_permutations(n_fp, order)

        if node_desc.face_nodes is not None:
            n_fp = field.gel.faces.shape[1]
            field.face_dof_perms = get_facet_dof_permutations(n_fp, order)

    def apply(self, field, bk, dim):
        """Apply orientation-dependent DOF permutation."""
        facet_perms = (field.edge_dof_perms if dim == 1
                       else field.face_dof_perms)
        perms = facet_perms[bk.ori]
        iaux = nm.arange(bk.gdofs.shape[0], dtype=nm.int32)
        field.econn[bk.iel[:, None], bk.iep] = bk.gdofs[iaux[:, None], perms]


class HierarchicOrientationStrategy:
    """Facet orientation handling for Lobatto (hierarchic) fields.

    Encodes orientation as sign bits in ``field.ori``, with special
    handling for tensor-product edges (2_4, 3_8) and quadrilateral faces.
    """

    def init_econn(self, field):
        """Allocate the sign-encoding orientation array alongside *econn*.

        Note that ``H1HierarchicVolumeField._init_econn`` already sets
        ``field.ori`` for backward compatibility; this method simply
        ensures the array is present in case the builder is used from
        a different code path.
        """
        if not hasattr(field, 'ori') or field.ori is None:
            field.ori = nm.zeros_like(field.econn)

    def setup(self, field):
        """No extra setup needed beyond *node_desc*."""

    def apply(self, field, bk, dim):
        """Assign facet DOFs in order and compute sign-encoded orientation."""
        field.econn[bk.iel[:, None], bk.iep] = bk.gdofs

        n_fp = 2 if dim == 1 else field.gel.surface_facet.n_vertex
        ori = bk.ori

        if (n_fp == 2) and (field.gel.name in ['2_4', '3_8']):
            tp_edges = field.gel.edges
            ecs = field.gel.coors[tp_edges]
            tp_edge_ori = (nm.diff(ecs, axis=1).sum(axis=2) > 0).squeeze()
            n_el = field.region.get_cells().shape[0]
            aux = nm.tile(tp_edge_ori, n_el)
            ori = nm.where(aux, ori, 1 - ori)

        if n_fp == 2:  # Edges.
            ps = field.poly_space
            orders = ps.node_orders
            eori = nm.repeat(ori[:, None], bk.n_dof_per_facet, 1)
            eoo = orders[bk.iep] % 2
            field.ori[bk.iel[:, None], bk.iep] = eori * eoo

        elif n_fp == 3:  # Triangular faces.
            raise NotImplementedError

        else:  # Quadrilateral faces.
            new = nm.repeat(nm.arange(8, dtype=nm.int32), 3)
            translate = prepare_translate(
                [31, 59, 63,
                 0, 1, 4,
                 22, 30, 62,
                 32, 33, 41,
                 11, 15, 43,
                 3, 6, 7,
                 20, 52, 60,
                 48, 56, 57], new)
            ori = translate[ori]
            eori = nm.repeat(ori[:, None], bk.n_dof_per_facet, 1)

            ps = field.poly_space
            orders = ps.face_axes_nodes[bk.iep - ps.face_indx[0]]
            eoo = orders % 2
            eoo0, eoo1 = eoo[..., 0], eoo[..., 1]

            i0 = nm.where(eori < 4)
            i1 = nm.where(eori >= 4)

            eori[i0] = nm.bitwise_and(eori[i0], 2 * eoo0[i0] + 5)
            eori[i0] = nm.bitwise_and(eori[i0], eoo1[i0] + 6)

            eori[i1] = nm.bitwise_and(eori[i1], eoo0[i1] + 6)
            eori[i1] = nm.bitwise_and(eori[i1], 2 * eoo1[i1] + 5)

            field.ori[bk.iel[:, None], bk.iep] = eori


# =========================================================================
# Substitution strategies
# =========================================================================

class NodalSubstitutionStrategy:
    """Hanging-node DOF substitution rules for Lagrange (nodal) fields.

    Supports 2_4 (quadrilateral) and 3_8 (hexahedral) reference elements.
    """

    def substitute_dofs(self, field, subs):
        if field.gel.name == '2_4':
            self._substitute_2_4(field, subs)
        elif field.gel.name == '3_8':
            self._substitute_3_8(field, subs)
        else:
            raise ValueError('unsupported reference element type! (%s)'
                             % field.gel.name)

    @staticmethod
    def _substitute_2_4(field, subs):
        ef = field.efaces
        for ii, sub in enumerate(subs):
            ee = ef[sub[1]].copy()
            ee[0], ee[1] = ee[1], ee[0]
            ee[2:] = ee[-1:1:-1]

            master = field.econn[sub[0], ee]
            field.econn[sub[2], ef[sub[3]]] = master
            field.econn[sub[4], ef[sub[5]]] = master

    @staticmethod
    def _substitute_3_8(field, subs):
        def _sort4(p):
            key = 0
            if (p[0] < p[1]): key += 1
            if (p[0] < p[2]): key += 2
            if (p[1] < p[2]): key += 4
            if (p[0] < p[3]): key += 8
            if (p[1] < p[3]): key += 16
            if (p[2] < p[3]): key += 32
            return key

        if subs[0] is not None:
            ef = field.efaces
            epf = field.gel.get_edges_per_face()
            nde = field.node_desc.edge
            ndf = field.node_desc.face
            gedges = field.gel.edges
            gfaces = field.gel.faces

            for ii, sub in enumerate(subs[0]):
                master = field.econn[sub[0]]
                fmaster = master[ef[sub[1]]]
                lmaster = fmaster.tolist()

                for ic in range(4):
                    ia, ib = 2 + 2 * ic, 2 + 2 * ic + 1
                    cell = field.econn[sub[ia]]

                    iv = cell[ef[sub[ib]][0]]
                    i0 = lmaster.index(iv)
                    for ik in range(4):
                        cell[ef[sub[ib]][ik]] = fmaster[:4][i0 - ik]

                    if nde is not None:
                        sedges = epf[sub[ib]]
                        medges = epf[sub[1]]
                        for ie, sedge in enumerate(sedges):
                            iim = i0 - 1 - ie
                            ies = nde[sedge]
                            medge = medges[iim]
                            iem = nde[medge]

                            vm = master[gedges[medge]][0]
                            vs = cell[gedges[sedge]][0]
                            if vm == vs:
                                cell[ies] = field.econn[sub[0], iem]
                            else:
                                cell[ies] = field.econn[sub[0], iem[::-1]]

                    if ndf is not None:
                        new_ori = _sort4(cell[gfaces[sub[ib]]])
                        smaster = nm.sort(master[ndf[sub[1]]])
                        aux = field.face_dof_perms[new_ori]
                        cell[ndf[sub[ib]]] = smaster[aux]

        if subs[1] is not None:
            ef = field.eedges
            for ii, sub in enumerate(subs[1]):
                master = field.econn[sub[0]]
                me = master[gedges[sub[1]]]
                for ic in range(2):
                    ia, ib = 2 + 2 * ic, 2 + 2 * ic + 1
                    cell = field.econn[sub[ia]]
                    ce = cell[gedges[sub[ib]]]

                    if (me[0] == ce[0]) or (me[1] == ce[1]):
                        cell[ef[sub[ib]]] = master[ef[sub[1]]]
                    else:
                        ee = ef[sub[1]].copy()
                        ee[0], ee[1] = ee[1], ee[0]
                        ee[2:] = ee[-1:1:-1]
                        cell[ef[sub[ib]]] = master[ee]

    def eval_basis_transform(self, field, subs):
        from sfepy.discrete import Integral
        from sfepy.discrete.fem import Mesh, FEDomain, Field

        transform = nm.tile(nm.eye(field.econn.shape[1]),
                            (field.econn.shape[0], 1, 1))
        if subs is None:
            return transform

        gel = field.gel
        ao = field.approx_order

        conn = [gel.conn]
        mesh = Mesh.from_data('a', gel.coors, None, [conn], [nm.array([0])],
                              [gel.name])
        cdomain = FEDomain('d', mesh)
        comega = cdomain.create_region('Omega', 'all')
        rcfield = Field.from_args('rc', field.dtype, 1, comega,
                                  approx_order=ao)

        fdomain = cdomain.refine()
        fomega = fdomain.create_region('Omega', 'all')
        rffield = Field.from_args('rf', field.dtype, 1, fomega,
                                  approx_order=ao)

        def assign_transform(transform, bf, subs, ef):
            if not len(subs):
                return

            n_sub = (subs.shape[1] - 2) // 2

            for ii, sub in enumerate(subs):
                for ij in range(n_sub):
                    ik = 2 * (ij + 1)
                    fface = ef[sub[ik + 1]]
                    mtx = transform[sub[ik]]
                    ix, iy = nm.meshgrid(fface, fface)
                    cbf = bf[iy, 0, ix]
                    mtx[ix, iy] = cbf

        fcoors = rffield.get_coor()
        coors = fcoors[rffield.econn[0]]
        integral = Integral('i', coors=coors,
                            weights=nm.ones_like(coors[:, 0]))

        rcfield.clear_qp_basis()
        bf = rcfield.eval_basis('v', False, integral)

        if gel.name == '2_4':
            fsubs = subs
            esubs = None
            assign_transform(transform, bf, fsubs, rffield.efaces)
        else:
            fsubs = subs[0]
            esubs = subs[1]
            assign_transform(transform, bf, fsubs, rffield.efaces)
            if esubs is not None:
                assign_transform(transform, bf, esubs, rffield.eedges)

        assert_((nm.abs(transform.sum(1) - 1.0) < 1e-15).all())
        return transform


class NoSubstitutionStrategy:
    """No-op substitution strategy for fields that do not support hanging
    nodes.  Raises ``NotImplementedError`` on attempted substitution and
    returns an identity transform."""

    def substitute_dofs(self, field, subs):
        raise NotImplementedError(
            'DOF substitution not supported for field type %s'
            % field.__class__.__name__)

    def eval_basis_transform(self, field, subs):
        return nm.tile(nm.eye(field.econn.shape[1]),
                       (field.econn.shape[0], 1, 1))


# =========================================================================
# Surface econn helpers
# =========================================================================

def compute_surface_econn(region, efaces, volume_econn, volume_region=None):
    """Build *econn* for a regular surface from face indices.

    Parameters
    ----------
    region : Region
        The surface region.
    efaces : ndarray
        Reference element face definitions (vertex indices per face).
    volume_econn : ndarray
        Volume element DOF connectivity.
    volume_region : Region or slice, optional
        Restricts the volume cells to consider.

    Returns
    -------
    econn : ndarray of int32, shape (n_faces, n_fp)
        Surface DOF connectivity.
    fis : ndarray of int32, shape (n_faces, 2)
        Facet indices (cell_id, local_face_id).
    """
    face_indices = region.get_facet_indices()
    faces = efaces[face_indices[:, 1]]
    if faces.size == 0 and not region.is_empty:
        raise ValueError('region with no faces! (%s)' % region.name)

    if volume_region is None:
        ii = face_indices[:, 0]
    elif hasattr(volume_region, 'get_cell_indices'):
        ii = volume_region.get_cell_indices(face_indices[:, 0])
    else:
        ii = volume_region

    try:
        ee = volume_econn[ii]
    except Exception:
        raise ValueError('missing region face indices! (%s)'
                         % region.name)

    econn = nm.empty(faces.shape, dtype=nm.int32)
    for ir, face in enumerate(faces):
        econn[ir] = ee[ir, face]

    return econn, face_indices


def compute_phantom_econn(region, volume_econn):
    """Build *econn* for a phantom surface (full cell connectivity).

    Parameters
    ----------
    region : Region
        The region whose cells define the phantom surface.
    volume_econn : ndarray
        Volume element DOF connectivity.

    Returns
    -------
    econn : ndarray of int32, shape (n_cells, n_ep)
        Phantom surface DOF connectivity (same as cell connectivity).
    fis : ndarray of int32, shape (n_cells, 2)
        Facet indices with cell_id in column 0, -1 in column 1.
    """
    ii = region.get_cells()
    econn = volume_econn[ii]
    fis = -nm.ones((econn.shape[0], 2), dtype=nm.int32)
    fis[:, 0] = ii
    return econn, fis


def finish_surface(name, region, econn, fis, set_orientation_map_fn):
    """Common post-processing for surface data objects.

    Computes ``nodes``, ``leconn``, ``face_type``, ``bkey``, empty
    ``meconn`` / ``mleconn`` dicts, and calls *set_orientation_map_fn*
    before returning a flat attribute dict.

    Parameters
    ----------
    name : str
        Surface data object name.
    region : Region
        The surface region.
    econn : ndarray
        Surface DOF connectivity.
    fis : ndarray
        Facet indices.
    set_orientation_map_fn : callable
        Function ``(self) -> None`` that sets ``self.ori_map``.

    Returns
    -------
    attrs : dict
        All attributes to assign to the surface object.
    """
    from sfepy.base.base import get_default

    name = get_default(name, 'surface_data_%s' % region.name)
    fis = nm.ascontiguousarray(nm.asarray(fis, dtype=nm.int32))
    econn = nm.asarray(econn, dtype=nm.int32)

    nodes = nm.unique(econn)
    if len(nodes):
        remap = prepare_remap(nodes, nodes.max() + 1)
        leconn = remap[econn].copy()
    else:
        leconn = econn.copy()

    n_fa, n_fp = econn.shape
    face_type = 's%d' % n_fp
    bkey = 'b%s' % face_type[1:]

    attrs = dict(
        name=name,
        fis=fis,
        econn=econn,
        n_fa=n_fa,
        n_fp=n_fp,
        nodes=nodes,
        leconn=leconn,
        face_type=face_type,
        bkey=bkey,
        meconn={},
        mleconn={},
    )
    # Let the caller apply orientation_map via its own method (which may
    # depend on self.n_fp set above).
    return attrs


# =========================================================================
# Main builder
# =========================================================================

class DofConnectivityBuilder:
    """Central DOF connectivity builder with pluggable strategies.

    Encapsulates all DOF connectivity construction logic, separating the
    common bookkeeping (facet traversal, remap computation) from
    field-type-specific orientation handling and hanging-node substitution
    rules.

    A field declares its strategies via class-level attributes::

        class MyField(FEField):
            _orientation_strategy_cls = NodalOrientationStrategy
            _substitution_strategy_cls = NodalSubstitutionStrategy

    The builder is instantiated automatically during
    :meth:`FEField._setup_global_basis`.
    """

    def __init__(self, field, orientation_strategy, substitution_strategy=None):
        self.field = field
        self.orientation = orientation_strategy
        self.substitution = substitution_strategy

    # -- Public entry points called from FEField._setup_global_basis ----

    def init_econn(self):
        """Initialize extended connectivity and any extra state."""
        field = self.field
        field._init_econn()
        self.orientation.init_econn(field)

    def setup_facet_orientations(self):
        """Setup *node_desc* and orientation-specific state."""
        field = self.field
        field.node_desc = field.poly_space.describe_nodes()
        self.orientation.setup(field)

    def setup_facet_dofs(self, dim, facet_desc, offset):
        """Build DOF connectivity for edges (*dim=1*) or faces (*dim=2*)."""
        bk = self._facet_bookkeeping(dim, facet_desc, offset)
        self.orientation.apply(self.field, bk, dim)
        return bk.n_dof, bk.all_dofs, bk.remap

    def setup_bubble_dofs(self):
        """Setup bubble (cell-interior) DOF connectivity."""
        field = self.field
        if field.is_surface or field.node_desc.bubble is None:
            return 0, None, None

        offset = field.n_vertex_dof + field.n_edge_dof + field.n_face_dof
        n_dof_per_cell = field.node_desc.bubble.shape[0]

        ii = field.region.get_cells()
        remap = prepare_remap(ii, field.cmesh.n_el)

        n_cell = ii.shape[0]
        n_dof = n_dof_per_cell * n_cell

        all_dofs = nm.arange(offset, offset + n_dof, dtype=nm.int32)
        all_dofs.shape = (n_cell, n_dof_per_cell)
        iep = field.node_desc.bubble[0]
        field.econn[:, iep:] = all_dofs

        return n_dof, all_dofs, remap

    def substitute_dofs(self, subs):
        """Perform DOF substitution (hanging nodes)."""
        if self.substitution:
            self.substitution.substitute_dofs(self.field, subs)
        else:
            raise NotImplementedError(
                'DOF substitution not supported for field type %s'
                % self.field.__class__.__name__)

    def eval_basis_transform(self, subs):
        """Evaluate the basis transformation matrix for substituted DOFs."""
        if self.substitution:
            return self.substitution.eval_basis_transform(self.field, subs)
        else:
            field = self.field
            return nm.tile(nm.eye(field.econn.shape[1]),
                           (field.econn.shape[0], 1, 1))

    # -- Shared bookkeeping ----------------------------------------------

    def _facet_bookkeeping(self, dim, facet_desc, offset):
        """
        Common bookkeeping for facet DOF connectivity setup.

        Computes the arrays that all field types need before applying
        their orientation-specific adjustments.
        """
        field = self.field
        facet_desc = nm.array(facet_desc)
        n_dof_per_facet = facet_desc.shape[1]

        cmesh = field.cmesh

        facets = field.region.entities[dim]
        ii = nm.arange(facets.shape[0], dtype=nm.int32)
        all_dofs = offset + expand_nodes_to_dofs(ii, n_dof_per_facet)

        remap = prepare_remap(facets, cmesh.num[dim])

        cconn = cmesh.get_conn(field.region.tdim, dim)
        offs = cconn.offsets

        n_f = (field.gel.edges.shape[0] if dim == 1
               else field.gel.faces.shape[0])

        oris = cmesh.get_orientations(dim)

        gcells = field.region.get_cells()
        n_el = gcells.shape[0]

        iel = nm.arange(n_el, dtype=nm.int32).repeat(n_f)
        ies = nm.tile(nm.arange(n_f, dtype=nm.int32), n_el)

        aux = offs[gcells][:, None] + ies.reshape((n_el, n_f))

        indices = cconn.indices[aux]
        facets_of_cells = remap[indices].ravel()

        ori = oris[aux].ravel()

        gdofs = offset + expand_nodes_to_dofs(facets_of_cells,
                                              n_dof_per_facet)

        iep = facet_desc[ies]

        n_dof = n_dof_per_facet * facets.shape[0]
        assert_(n_dof == nm.prod(all_dofs.shape))

        return Struct(n_dof_per_facet=n_dof_per_facet,
                      all_dofs=all_dofs,
                      remap=remap,
                      n_f=n_f,
                      facets_of_cells=facets_of_cells,
                      gdofs=gdofs,
                      iel=iel,
                      ies=ies,
                      iep=iep,
                      ori=ori,
                      n_dof=n_dof)
