import numpy as nm
import pytest

from sfepy.base.conf import transform_functions
import sfepy.base.testing as tst

def get_vertices(coors, domain=None):
    x, z = coors[:,0], coors[:,2]

    return nm.where((z < 0.1) & (x < 0.1))[0]

def get_cells(coors, domain=None):
    return nm.where(coors[:, 0] < 0)[0]

@pytest.fixture(scope='module')
def data():
    from sfepy import data_dir
    from sfepy.base.base import Struct
    from sfepy.discrete.fem import Mesh, FEDomain
    from sfepy.discrete import Functions

    mesh = Mesh.from_file(data_dir
                          + '/meshes/various_formats/abaqus_tet.inp')
    mesh.nodal_bcs['set0'] = [0, 7]
    domain = FEDomain('test domain', mesh)

    conf_functions = {
        'get_vertices' : (get_vertices,),
        'get_cells' : (get_cells,),
    }
    functions = Functions.from_conf(transform_functions(conf_functions))

    return Struct(domain=domain, functions=functions)

def test_selectors(data):
    """
    Test basic region selectors.
    """
    selectors = [
        ['all', 'cell'],
        ['vertices of surface', 'facet'],
        ['vertices of group %d' % data.domain.cmesh.vertex_groups[0],
         'facet'],
        ['vertices of set set0', 'vertex'],
        ['vertices in (z < 0.1) & (x < 0.1)', 'facet'],
        ['vertices by get_vertices', 'cell'],
        ['vertex 0, 1, 2', 'vertex'],
        ['vertex in r.r6', 'vertex'],
        ['cells of group %d' % data.domain.cmesh.cell_groups[0], 'cell'],
        # ['cells of set 0', 'cell'], not implemented...
        ['cells by get_cells', 'cell'],
        ['cell 1, 4, 5', 'cell'],
        ['copy r.r5', 'cell'],
        ['r.r5', 'cell'],
    ]

    vertices = [
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
        [0,  1,  3,  7],
        [0,  7],
        [1,  2,  3,  4,  5,  9, 11],
        [1,  2,  3,  4,  5,  9, 11],
        [0,  1,  2],
        [0],
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
        [0,  1,  2,  3,  4,  5,  6,  9, 10, 11],
        [0,  1,  2,  3,  4,  5,  6,  8],
        [1,  2,  3,  4,  5,  9, 11],
        [1,  2,  3,  4,  5,  9, 11],
    ]

    data.domain.reset_regions()

    ok = True
    for ii, sel in enumerate(selectors):
        tst.report('select:', sel)
        reg = data.domain.create_region('r%d' % ii, sel[0], kind=sel[1],
                                        functions=data.functions)
        _ok = ((len(reg.vertices) == len(vertices[ii]))
               and (reg.vertices == vertices[ii]).all())
        tst.report('  vertices:', _ok)

        ok = ok and _ok

    assert ok

def test_operators(data):
    """
    Test operators in region selectors.
    """
    ok = True

    data.domain.reset_regions()

    r1 = data.domain.create_region('r1', 'all')

    sel = 'r.r1 -v vertices of group 0'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel, kind='vertex')
    av = [2,  4,  5,  6,  8,  9, 10, 11, 12]
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'vertex 0, 1, 2 +v vertices of group 0'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel, kind='vertex')
    av = [0,  1,  2,  3,  7]
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'vertex 0, 1, 2 *v vertices of group 0'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel, kind='vertex')
    av = [0,  1]
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'vertex 20 *v vertices of group 0'
    tst.report('select:', sel, ', allow_empty == False')
    _ok = False
    try:
        reg = data.domain.create_region('reg', sel, kind='vertex',
                                        allow_empty=False)
    except ValueError:
        _ok = True
    tst.report('  exception raised:', _ok)
    ok = ok and _ok

    sel = 'vertex 20 *v vertices of group 0'
    tst.report('select:', sel, ', allow_empty == True')
    reg = data.domain.create_region('reg', sel, kind='vertex',
                                    allow_empty=True)
    av = []
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'r.r1 -c cell 1, 4, 5'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel)
    _ok = (nm.setdiff1d(r1.cells[0], [1, 4, 5]) == reg.cells[0]).all()
    tst.report('  cells:', _ok)
    ok = ok and _ok

    sel = 'cell 8, 3 +c cell 1, 4, 5'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel)
    cells = [1,  3,  4,  5,  8]
    _ok = (reg.cells == nm.array(cells)).all()
    tst.report('  cells:', _ok)
    ok = ok and _ok

    sel = 'cell 8, 3, 2 *c cell 8, 4, 2, 7'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel)
    cells = [2,  8]
    _ok = (reg.cells == nm.array(cells)).all()
    tst.report('  cells:', _ok)
    ok = ok and _ok

    assert ok

def test_region_by_groups():
    """
    Test region selection by cell groups and vertex groups
    using unified group interface.
    """
    from sfepy import data_dir
    from sfepy.discrete.fem import Mesh, FEDomain

    mesh = Mesh.from_file(data_dir
                          + '/meshes/2d/special/square_triquad.mesh')
    domain = FEDomain('test domain', mesh)

    unique_cell_groups = mesh.get_unique_cell_groups()
    unique_vertex_groups = mesh.get_unique_vertex_groups()

    tst.report('unique cell groups:', unique_cell_groups)
    tst.report('unique vertex groups:', unique_vertex_groups)

    ok = True

    for cg in unique_cell_groups[:2]:
        sel = 'cells of group %d' % cg
        tst.report('select:', sel)
        reg = domain.create_region('region_cg_%d' % cg, sel, kind='cell')

        cells_from_region = reg.cells
        cells_from_mesh = []
        for desc in mesh.descs:
            cell_groups = mesh.get_cell_groups(desc)
            cmesh = mesh.get_cmesh(desc)
            cells_of_type = nm.where(cmesh.cell_types
                                     == cmesh.key_to_index[desc])[0]
            cells_of_group = cells_of_type[nm.where(cell_groups == cg)[0]]
            cells_from_mesh.extend(cells_of_group.tolist())

        cells_from_mesh = nm.array(cells_from_mesh)
        _ok = (nm.sort(cells_from_region) == nm.sort(cells_from_mesh)).all()
        tst.report('  cells match:', _ok)
        if not _ok:
            tst.report('  region cells:', cells_from_region)
            tst.report('  mesh cells:', cells_from_mesh)
        ok = ok and _ok

    for vg in unique_vertex_groups[:3]:
        sel = 'vertices of group %d' % vg
        tst.report('select:', sel)
        reg = domain.create_region('region_vg_%d' % vg, sel, kind='vertex')

        vertices_from_region = reg.vertices
        vertices_from_mesh = nm.where(mesh.get_vertex_groups() == vg)[0]

        _ok = (nm.sort(vertices_from_region)
               == nm.sort(vertices_from_mesh)).all()
        tst.report('  vertices match:', _ok)
        if not _ok:
            tst.report('  region vertices:', vertices_from_region)
            tst.report('  mesh vertices:', vertices_from_mesh)
        ok = ok and _ok

    tst.report('region by groups test:', 'PASSED' if ok else 'FAILED')
    assert ok

def test_save_regions_as_groups():
    """
    Test that save_regions_as_groups correctly uses unified group interface.
    """
    import tempfile
    import os.path as op
    from sfepy import data_dir
    from sfepy.discrete.fem import Mesh, FEDomain

    mesh = Mesh.from_file(data_dir
                          + '/meshes/2d/special/square_triquad.mesh')
    domain = FEDomain('test domain', mesh)

    omega = domain.create_region('Omega', 'all')
    gamma = domain.create_region('Gamma', 'vertices of surface', 'facet')

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = op.join(tmpdir, 'test_region_groups.vtk')
        domain.save_regions_as_groups(filename)

        mesh_restored = Mesh.from_file(filename)

        ok = mesh_restored.n_nod == mesh.n_nod
        tst.report('node count match:', ok)
        assert ok

        ok = mesh_restored.n_el == mesh.n_el
        tst.report('element count match:', ok)
        assert ok

    tst.report('save regions as groups test: PASSED')
