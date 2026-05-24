
import numpy as nm

from sfepy.base.base import output, assert_, OneTypeList, Struct
from sfepy.discrete.common.region import (Region, RegionDependencyAnalyzer,
                                          get_dependency_graph,
                                          sort_by_dependency, get_parents)
from sfepy.discrete.parse_regions import (create_bnf, visit_stack,
                                           ParseException, RegionParser)
from sfepy.discrete.common.region_builder import (RegionBuilder,
                                                   region_leaf, region_op)
from sfepy.discrete.common.region_coordinator import RegionCoordinator

class Domain(Struct):

    def __init__(self, name, mesh=None, nurbs=None, bmesh=None, regions=None,
                 verbose=False):
        Struct.__init__(self, name=name, mesh=mesh, nurbs=nurbs, bmesh=bmesh,
                        regions=regions, verbose=verbose)
        self._region_coordinator = None

    def get_centroids(self, dim):
        """
        Return the coordinates of centroids of mesh entities with dimension
        `dim`.
        """
        return self.cmesh.get_centroids(dim)

    def has_faces(self):
        return self.shape.tdim == 3

    def reset_regions(self):
        """
        Reset the list of regions associated with the domain.
        """
        self.regions = OneTypeList(Region)
        self._region_coordinator = RegionCoordinator(self)

    def create_extra_tdim_region(self, region, functions, tdim):
        from sfepy.discrete.fem.geometry_element import (GeometryElement,
            create_geometry_elements)
        from sfepy.discrete import PolySpace
        """
        Create a new region which has its own cmesh with
        topological dimension tdim.
        """
        mesh = self.mesh
        if mesh.cmesh_tdim[tdim] is not None:
            raise ValueError(f'cmesh of dimension {tdim} already exists!')

        aux = mesh.from_region(region, mesh, tdim=tdim)
        cmesh = aux.cmesh
        new_mat_id = nm.max([nm.max(k.cell_groups) for k in mesh.cmesh_tdim
                             if k is not None]) + 1
        cmesh.cell_groups[:] = new_mat_id
        mesh.cmesh_tdim[tdim] = cmesh
        mesh.descs += aux.descs
        mesh.dims += aux.dims
        mesh.n_el += aux.n_el
        cmesh.set_local_entities(create_geometry_elements())
        cmesh.setup_entities()

        gel = GeometryElement(aux.descs[0])
        if gel.dim > 0:
            gel.create_surface_facet()

        new_gel_entry = {aux.descs[0]: gel}
        self.geom_els.update(new_gel_entry)

        self.fix_element_orientation(geom_els=new_gel_entry, force_check=True)

        key = gel.get_interpolation_name()

        gel.poly_space = PolySpace.any_from_args(key, gel, 1)
        gel = gel.surface_facet
        if gel is not None:
            key = gel.get_interpolation_name()
            gel.poly_space = PolySpace.any_from_args(key, gel, 1)

        select = f'cells of group {new_mat_id}'
        region = self._region_coordinator.parse_and_build(
            select, self.regions, functions, tdim)
        region.field_dim = tdim

        return region

    def create_region(self, name, select, kind='cell', parent=None,
                      check_parents=True, extra_options=None, functions=None,
                      add_to_regions=True, allow_empty=False):
        """
        Region factory constructor. Append the new region to
        self.regions list.
        """
        if self._region_coordinator is None:
            self.reset_regions()
        return self._region_coordinator.create_region(
            name, select, kind=kind, parent=parent,
            check_parents=check_parents, extra_options=extra_options,
            functions=functions, add_to_regions=add_to_regions,
            allow_empty=allow_empty)

    def create_regions(self, region_defs, functions=None, allow_empty=False):
        if self._region_coordinator is None:
            self.reset_regions()
        return self._region_coordinator.create_regions(
            region_defs, functions=functions, allow_empty=allow_empty)

    def save_regions(self, filename, region_names=None):
        """
        Save regions as individual meshes.

        Parameters
        ----------
        filename : str
            The output filename.
        region_names : list, optional
            If given, only the listed regions are saved.
        """
        import os

        if region_names is None:
            region_names = self.regions.get_names()

        trunk, suffix = os.path.splitext(filename)

        output('saving regions...')
        for name in region_names:
            region = self.regions[name]
            output(name)
            dim = region.tdim
            is_surf = not region.can[dim] and region.can[dim - 1]
            aux = self.mesh.from_region(region, self.mesh, is_surface=is_surf)
            aux.write('%s_%s%s' % (trunk, region.name, suffix),
                      io='auto')
        output('...done')

    def save_regions_as_groups(self, filename, region_names=None):
        """
        Save regions in a single mesh but mark them by using different
        element/node group numbers.

        If regions overlap, the result is undetermined, with exception of the
        whole domain region, which is marked by group id 0.

        Region masks are also saved as scalar point data for output formats
        that support this.

        Parameters
        ----------
        filename : str
            The output filename.
        region_names : list, optional
            If given, only the listed regions are saved.
        """

        output('saving regions as groups...')
        aux = self.mesh.copy()
        n_ig = c_ig = 0
        n_nod = self.shape.n_nod

        # The whole domain region should go first.
        names = (region_names if region_names is not None
                 else self.regions.get_names())
        for name in names:
            region = self.regions[name]
            if region.vertices.shape[0] == n_nod:
                names.remove(region.name)
                names = [region.name] + names
                break

        out = {}
        for name in names:
            region = self.regions[name]
            output(region.name)

            aux.cmesh.vertex_groups[region.vertices] = n_ig
            n_ig += 1

            mask = nm.zeros((n_nod, 1), dtype=nm.float64)
            mask[region.vertices] = 1.0
            out[name] = Struct(name='region', mode='vertex', data=mask,
                               var_name=name, dofs=None)

            if region.has_cells():
                ii = region.get_cells()
                aux.cmesh.cell_groups[ii] = c_ig
                c_ig += 1

        aux.write(filename, io='auto', out=out)
        output('...done')
