import numpy as nm

from sfepy.base.base import get_default, Struct
from sfepy.discrete.fem.facets import (build_orientation_map,
                                       get_facet_dof_permutations,
                                       get_facet_orientation_array,
                                       get_facet_orientations)
from sfepy.discrete.fem.utils import prepare_remap

class FESurface(Struct):
    """Description of a surface of a finite element domain."""

    def __init__(self, name, region, efaces, volume_econn, volume_region=None,
                 approx_order=1):
        """nodes[leconn] == econn"""
        """nodes are sorted by node number -> same order as region.vertices"""
        self.name = get_default(name, 'surface_data_%s' % region.name)

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

        except:
            raise ValueError('missing region face indices! (%s)'
                             % region.name)

        econn = nm.empty(faces.shape, dtype=nm.int32)
        for ir, face in enumerate(faces):
            econn[ir] = ee[ir, face]

        nodes = nm.unique(econn)
        if len(nodes):
            remap = prepare_remap(nodes, nodes.max() + 1)
            leconn = remap[econn].copy()

        else:
            leconn = econn.copy()

        n_fa, n_fp = econn.shape
        face_type = 's%d' % n_fp

        # Store bkey in SurfaceData, so that base function can be
        # queried later.
        bkey = 'b%s' % face_type[1:]

        self.econn = econn
        self.fis = nm.ascontiguousarray(face_indices.astype(nm.int32))
        self.n_fa, self.n_fp = n_fa, n_fp
        self.nodes = nodes
        self.leconn = leconn
        self.face_type = face_type
        self.bkey = bkey
        self.approx_order = approx_order
        self.meconn = {}
        self.mleconn = {}
        self.set_orientation_map()

    @staticmethod
    def from_region(name, region, ret_gel=False):
        face_indices = region.get_facet_indices()
        cells = face_indices[:, 0]

        econn, gel = region.domain.get_conn(ret_gel=True,
                                            tdim=region.tdim, cells=cells)

        surface = FESurface(name, region, gel.get_surface_entities(), econn,
                            slice(0, econn.shape[0]))

        if ret_gel:
            return surface, gel.surface_facet
        else:
            return surface

    def set_orientation_map(self):
        n_fp = self.n_fp
        if n_fp <= 4:
            oo, _, _ = build_orientation_map(n_fp)
            self.ori_map = get_facet_orientation_array(oo, n_fp)
            if self.approx_order is not None and self.approx_order > 0:
                self.dof_perms = get_facet_dof_permutations(
                    n_fp, self.approx_order)
            else:
                self.dof_perms = None
        else:
            self.ori_map = None
            self.dof_perms = None

    def setup_mirror_connectivity(self, region, mirror_name):
        """
        Setup mirror surface connectivity required to integrate over a
        mirror region.

        1. Get orientation of the faces:
           a) for own elements -> ooris
           b) for mirror elements -> moris

        2. orientation -> permutation.
        """
        def get_omap(reg):
            if reg.tdim == (reg.dim - 1):
                cells = reg.get_cells()
                conn = reg.domain.get_conn(tdim=reg.tdim)[cells, :]
                return nm.argsort(conn)
            else:
                conn = reg.cmesh.get_conn_as_graph(reg.dim, reg.dim - 1)
                oris = get_facet_orientations(reg.cmesh, reg.dim - 1)
                fis = reg.get_facet_indices()
                if self.dof_perms is not None:
                    perms = self.dof_perms[
                        oris[conn.indptr[fis[:, 0]] + fis[:, 1]]]
                    return perms[:, :self.n_fp]
                else:
                    return self.ori_map[
                        oris[conn.indptr[fis[:, 0]] + fis[:, 1]]]

        if mirror_name in self.meconn:
            return

        mregion = region.get_mirror_region(mirror_name)
        mmap = get_omap(mregion)
        omap = get_omap(region)

        econn = self.econn
        if mirror_name in region.mirror_maps\
            and region.mirror_maps[mirror_name] is not None:
            mirror_map = region.mirror_maps[mirror_name]
            omap = omap[mirror_map]
            econn = econn[mirror_map]

        n_el, n_ep = econn.shape
        ii = nm.repeat(nm.arange(n_el)[:, None], n_ep, 1)
        meconn = nm.empty_like(econn)
        meconn[ii, mmap] = econn[ii, omap]

        nodes = nm.unique(meconn)
        remap = prepare_remap(nodes, nodes.max() + 1)
        self.meconn[mirror_name] = meconn
        self.mleconn[mirror_name] = remap[meconn].copy()

    def get_connectivity(self, local=False, trace_region=None):
        """
        Return the surface element connectivity.

        Parameters
        ----------
        local : bool
            If True, return local connectivity w.r.t. surface nodes,
            otherwise return global connectivity w.r.t. all mesh nodes.
        trace_trace : None or str
            If not None, return mirror connectivity according to `local`.
        """
        if trace_region is None:
            if local:
                return self.leconn

            else:
                return self.econn

        else:
            if local:
                return self.mleconn[trace_region]

            else:
                return self.meconn[trace_region]


class FEPhantomSurface(FESurface):
    """A phantom surface of the region with tdim=2."""

    def __init__(self, name, region, volume_econn, approx_order=1):
        self.name = get_default(name, 'surface_data_%s' % region.name)

        ii = region.get_cells()
        econn = volume_econn[ii]
        nodes = nm.unique(econn)

        if len(nodes):
            remap = prepare_remap(nodes, nodes.max() + 1)
            leconn = remap[econn].copy()

        else:
            leconn = econn.copy()

        n_fa, n_fp = econn.shape
        face_type = 's%d' % n_fp

        # Store bkey in SurfaceData, so that base function can be
        # queried later.
        bkey = 'b%s' % face_type[1:]

        self.econn = econn
        self.fis = -nm.ones((n_fa, 2), dtype=nm.int32)
        self.fis[:, 0] = ii
        self.n_fa, self.n_fp = n_fa, n_fp
        self.nodes = nodes
        self.leconn = leconn
        self.face_type = face_type
        self.bkey = bkey
        self.approx_order = approx_order
        self.meconn = {}
        self.mleconn = {}
        self.set_orientation_map()
