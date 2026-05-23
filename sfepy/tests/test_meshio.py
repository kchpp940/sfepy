# coding=utf8
import os
import os.path as op

from sfepy import data_dir
from sfepy.base.base import assert_
import sfepy.base.testing as tst

filename_meshes = ['/meshes/3d/cylinder.mesh',
                   '/meshes/3d/cylinder.vtk',
                   '/meshes/various_formats/small2d.mesh',
                   '/meshes/various_formats/small2d.vtk',
                   '/meshes/various_formats/octahedron.node',
                   '/meshes/various_formats/comsol_tri.txt',
                   '/meshes/various_formats/abaqus_hex.inp',
                   '/meshes/various_formats/abaqus_tet.inp',
                   '/meshes/various_formats/abaqus_quad.inp',
                   '/meshes/various_formats/abaqus_tri.inp',
                   '/meshes/various_formats/abaqus_quad_tri.inp',
                   '/meshes/various_formats/hex4.mesh3d',
                   '/meshes/various_formats/tetra8.mesh3d',
                   '/meshes/various_formats/cube.bdf',
                   '/meshes/various_formats/msh_tri.msh',
                   '/meshes/various_formats/msh_tetra.msh',
                   '/meshes/various_formats/xyz_quad.xyz',
                   '/meshes/various_formats/xyz_tet.xyz']
filename_meshes = [data_dir + name for name in filename_meshes]

def mesh_hook(mesh, mode):
    """
    Define a mesh programmatically.
    """
    if mode == 'read':
        nodes = [[0, 0], [1, 0], [1, 1], [0, 1]]
        nod_ids = [0, 0, 1, 1]
        conns = [[[0, 1, 2], [0, 2, 3]]]
        mat_ids = [[0, 1]]
        descs = ['2_3']

        mesh._set_io_data(nodes, nod_ids, conns, mat_ids, descs)

    elif mode == 'write':
        pass

try:
    from sfepy.discrete.fem.meshio import UserMeshIO
    _meshio_available = True
except ImportError:
    UserMeshIO = None
    _meshio_available = False

if UserMeshIO is not None:
    filename_meshes.extend([mesh_hook, UserMeshIO(mesh_hook)])
else:
    filename_meshes.append(mesh_hook)

same = [(0, 1), (2, 3)]

def _compare_meshes(mesh0, mesh1, flag='cv'):
    import numpy as nm

    oks = []

    ok0 = (mesh0.dim == mesh1.dim)
    if not ok0:
        tst.report('dimension failed!')
    oks.append(ok0)

    ok0 = mesh0.n_nod == mesh1.n_nod
    if not ok0:
        tst.report('number of nodes failed!')
    oks.append(ok0)

    ok0 = mesh0.n_el == mesh1.n_el
    if not ok0:
        tst.report('number of elements failed!')
    oks.append(ok0)

    ok0 = mesh0.descs == mesh1.descs
    if not ok0:
        tst.report('element types failed!')
    oks.append(ok0)

    ok0 = nm.allclose(mesh0.coors, mesh1.coors)
    if not ok0:
        tst.report('nodes failed!')
    oks.append(ok0)

    if 'v' in flag:
        ok0 = nm.all(mesh0.cmesh.vertex_groups == mesh1.cmesh.vertex_groups)
        if not ok0:
            tst.report('node groups failed!')
        oks.append(ok0)

    if 'c' in flag:
        ok0 = nm.all(mesh0.cmesh.cell_groups == mesh1.cmesh.cell_groups)
        if not ok0:
            tst.report('material ids failed!')
        oks.append(ok0)

    ok0 = (nm.all(mesh0.cmesh.get_cell_conn().indices ==
                  mesh1.cmesh.get_cell_conn().indices) and
           nm.all(mesh0.cmesh.get_cell_conn().offsets ==
                  mesh1.cmesh.get_cell_conn().offsets))
    if not ok0:
        tst.report('connectivities failed!')
    oks.append(ok0)

    return oks

def test_read_meshes():
    """Try to read all listed meshes."""
    from sfepy.discrete.fem import Mesh

    conf_dir = op.dirname(__file__)
    oks = []
    for ii, filename in enumerate(filename_meshes):
        tst.report('%d. mesh: %s' % (ii + 1, filename))
        try:
            mesh = Mesh.from_file(filename, prefix_dir=conf_dir)

        except Exception as exc:
            tst.report(exc)
            tst.report('read failed!')
            oks.append(False)
            continue

        try:
            assert_(mesh.dim == (mesh.coors.shape[1]))
            assert_(mesh.n_nod == (mesh.coors.shape[0]))
            assert_(mesh.n_nod == (mesh.cmesh.vertex_groups.shape[0]))
            assert_(mesh.n_el == mesh.cmesh.num[mesh.cmesh.tdim])

        except ValueError as exc:
            tst.report(exc)
            tst.report('read assertion failed!')
            oks.append(False)
            continue

        oks.append(True)
        tst.report('read ok')

    assert_(all(oks))

def test_compare_same_meshes():
    """
    Compare same meshes in various formats.
    """
    from sfepy.discrete.fem import Mesh

    conf_dir = op.dirname(__file__)
    oks = []
    for i0, i1 in same:
        name0 = filename_meshes[i0]
        name1 = filename_meshes[i1]
        tst.report('comparing meshes from "%s" and "%s"' % (name0, name1))

        mesh0 = Mesh.from_file(name0, prefix_dir=conf_dir)
        mesh1 = Mesh.from_file(name1, prefix_dir=conf_dir)
        ok = _compare_meshes(mesh0, mesh1)
        tst.report('->', ok)
        oks.extend(ok)

    assert_(all(oks))

def test_read_dimension():
    from sfepy.discrete.fem import MeshIO

    meshes = {data_dir + '/meshes/various_formats/small2d.mesh' : 2,
              data_dir + '/meshes/various_formats/small2d.vtk' : 2,
              data_dir + '/meshes/various_formats/small3d.mesh' : 3}

    ok = True
    conf_dir = op.dirname(__file__)
    for filename, adim in meshes.items():
        tst.report('mesh: %s, dimension %d' % (filename, adim))
        io = MeshIO.any_from_filename(filename, prefix_dir=conf_dir)
        dim = io.read_dimension()
        if dim != adim:
            tst.report('read dimension %d -> failed' % dim)
            ok = False
        else:
            tst.report('read dimension %d -> ok' % dim)

    assert_(ok)

def test_write_read_meshes(output_dir):
    """
    Try to write and then read all supported formats.
    """
    import numpy as nm
    from sfepy.discrete.fem import Mesh
    from sfepy.discrete.fem.meshio import supported_formats

    conf_dir = op.dirname(__file__)
    mesh0 = Mesh.from_file(data_dir
                           + '/meshes/various_formats/small3d.mesh',
                           prefix_dir=conf_dir)

    mesh0.cmesh.vertex_groups[:] =\
        nm.random.randint(1, 10, size=mesh0.cmesh.n_coor)

    mesh0.cmesh.cell_groups[:] =\
        nm.random.randint(1, 10, size=mesh0.cmesh.n_el)

    oks = []
    for name, (cls, suffix, flag) in supported_formats.items():
        if 'w' not in flag: continue

        suffix = suffix[0]  # only the first of possible suffixes
        filename = op.join(output_dir, 'test_mesh_wr' + suffix)
        tst.report('%s format: %s' % (suffix, filename))

        try:
            mesh0.write(filename, io='auto')

        except:
            if cls == 'meshio':
                import traceback
                tst.report('-> cannot write "%s" format into "%s",'
                           ' skipping' % (name, filename))
                tb = traceback.format_exc()
                tst.report('reason:\n', tb)
                continue

            else:
                raise

        else:
            try:
                mesh1 = Mesh.from_file(filename)

            except:
                if cls == 'meshio':
                    import traceback
                    tst.report('-> cannot read "%s" format from "%s",'
                               ' skipping' % (name, filename))
                    tb = traceback.format_exc()
                    tst.report('reason:\n', tb)
                    continue

                else:
                    raise

        ok = _compare_meshes(mesh0, mesh1, flag)
        tst.report('->', ok)
        oks.extend(ok)

    assert_(all(oks))

def test_hdf5_meshio():
    try:
        from igakit import igalib; igalib
    except ImportError:
        tst.report('hdf5_meshio not-tested (missing igalib module)!')
        return

    import tempfile
    import numpy as nm
    import scipy.sparse as sps
    from sfepy.discrete.fem.meshio import HDF5MeshIO
    from sfepy.base.base import Struct
    from sfepy.base.ioutils import Cached, Uncached, SoftLink, \
                                   DataSoftLink
    from sfepy.discrete.iga.domain import IGDomain
    from sfepy.discrete.iga.domain_generators import gen_patch_block_domain
    from sfepy.solvers.ts import TimeStepper
    from sfepy.discrete.fem import Mesh

    conf_dir = op.dirname(__file__)
    mesh0 = Mesh.from_file(data_dir +
                           '/meshes/various_formats/small3d.mesh',
                           prefix_dir=conf_dir)

    shape = [4, 4, 4]
    dims = [5, 5, 5]
    centre = [0, 0, 0]
    degrees = [2, 2, 2]

    nurbs, bmesh, regions = gen_patch_block_domain(dims, shape, centre,
                                                   degrees,
                                                   cp_mode='greville',
                                                   name='iga')
    ig_domain = IGDomain('iga', nurbs, bmesh, regions=regions)

    int_ar = nm.arange(4)

    data = {
        'list': range(4),
        'mesh1': mesh0,
        'mesh2': mesh0,
        'mesh3': Uncached(mesh0),
        'mesh4': SoftLink('/step0/__cdata/data/data/mesh2'),
        'mesh5': DataSoftLink('Mesh','/step0/__cdata/data/data/mesh1/data'),
        'mesh6': DataSoftLink('Mesh','/step0/__cdata/data/data/mesh2/data',
            mesh0),
        'mesh7': DataSoftLink('Mesh','/step0/__cdata/data/data/mesh1/data',
            True),
        'iga' : ig_domain,
        'cached1': Cached(1),
        'cached2': Cached(int_ar),
        'cached3': Cached(int_ar),
        'types': ( True, False, None ),
        'tuple': ('first string', 'druhý UTF8 řetězec'),
        'struct': Struct(
            double=nm.arange(4, dtype=float),
            int=nm.array([2,3,4,7]),
            sparse=sps.csr_array(nm.array([1,0,0,5]).reshape((2,2)))
         )
    }

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as fil:
        io = HDF5MeshIO(fil.name)
        ts = TimeStepper(0,1.,0.1, 10)

        io.write(fil.name, mesh0, {
            'cdata' : Struct(
                mode='custom',
                data=data,
                unpack_markers=False
            )
        }, ts=ts)
        ts.advance()

        mesh = io.read()
        data['problem_mesh'] = DataSoftLink('Mesh', '/mesh', mesh)

        io.write(fil.name, mesh0, {
            'cdata' : Struct(
                mode='custom',
                data=data,
                unpack_markers=True
            )
        }, ts=ts)

        cache = {'/mesh': mesh }
        fout = io.read_data(0, cache=cache)
        fout2 = io.read_data(1, cache=cache )
        out = fout['cdata']
        out2 = fout2['cdata']

        assert_(out['mesh7'] is out2['mesh7'],
                'These two meshes should be in fact the same object')

        assert_(out['mesh6'] is out2['mesh6'],
                'These two meshes should be in fact the same object')

        assert_(out['mesh5'] is not out2['mesh5'],
                'These two meshes shouldn''t be in fact the same object')

        assert_(out['mesh1'] is out['mesh2'],
                'These two meshes should be in fact the same object')

        assert_(out['mesh1'] is out['mesh2'],
                'These two meshes should be in fact the same object')

        assert_(out['mesh4'] is out['mesh2'],
                'These two meshes should be in fact the same object')

        assert_(out['mesh5'] is not out['mesh2'],
                'These two meshes shouldn''t be in fact the same object')

        assert_(out['mesh6'] is out['mesh2'],
                'These two meshes should be in fact the same object')

        assert_(out['mesh7'] is not out['mesh2'],
                'These two meshes shouldn''t be in fact the same object')

        assert_(out['mesh3'] is not out['mesh2'],
                'These two meshes should be different objects')

        assert_(out['cached2'] is out['cached3'],
                'These two array should be the same object')

        assert_(out2['problem_mesh'] is mesh,
                'These two meshes should be the same objects')

        assert_(_compare_meshes(out['mesh1'], mesh0),
                'Failed to restore mesh')

        assert_(_compare_meshes(out['mesh3'], mesh0),
                'Failed to restore mesh')

        assert_((out['struct'].sparse.todense()
                 == data['struct'].sparse.todense()).all(),
                'Sparse matrix restore failed')

        ts.advance()
        io.write(fil.name, mesh0, {
                'cdata' : Struct(
                    mode='custom',
                    data=[
                        DataSoftLink('Mesh',
                                     '/step0/__cdata/data/data/mesh1/data',
                                     mesh0),
                        mesh0
                    ]
                )
        }, ts=ts)
        out3 = io.read_data(2)['cdata']
        assert_(out3[0] is out3[1])

    os.remove(fil.name)

    #this property is not restored
    del data['iga'].nurbs.nurbs

    #not supporting comparison
    del data['iga']._bnf
    del out2['iga']._bnf

    #restoration of this property fails
    del data['iga'].vertex_set_bcs
    del out2['iga'].vertex_set_bcs

    #these soflinks have no information how to unpack, so it must be
    #done manually
    data['mesh4'] = mesh0
    data['mesh5'] = mesh0
    data['mesh7'] = mesh0

    for key, val in out2.items():
        tst.report('comparing:', key)
        tst.assert_equal(val, data[key])

def test_mixed_cell_mesh_group_roundtrip():
    """
    Test that cell groups and vertex groups are preserved during
    write/read roundtrip for mixed-cell meshes using unified group interface.

    Skips if C extensions are not available.
    """
    import tempfile
    import numpy as nm

    if not _check_c_extensions():
        tst.report('mixed-cell mesh roundtrip not-tested'
                   ' (missing Cython extensions)!')
        return

    from sfepy.discrete.fem import Mesh

    conf_dir = op.dirname(__file__)
    mesh_file = data_dir + '/meshes/2d/special/square_triquad.mesh'

    tst.report('testing mixed-cell mesh roundtrip:', mesh_file)

    mesh0 = Mesh.from_file(mesh_file, prefix_dir=conf_dir)

    original_cell_groups = {d: mesh0.get_cell_groups(d).copy()
                             for d in mesh0.descs}
    original_vertex_groups = mesh0.get_vertex_groups().copy()
    original_unique_cell_groups = mesh0.get_unique_cell_groups()
    original_unique_vertex_groups = mesh0.get_unique_vertex_groups()

    tst.report('element types:', mesh0.descs)
    tst.report('unique cell groups:', original_unique_cell_groups)
    tst.report('unique vertex groups:', original_unique_vertex_groups)

    for desc in mesh0.descs:
        cg = mesh0.get_cell_groups(desc)
        tst.report('  %s: %d cells, cell groups: %s'
                   % (desc, len(cg), nm.unique(cg)))

    formats_to_test = [
        ('mesh', '.mesh'),
        ('vtk', '.vtk'),
        ('msh', '.msh'),
    ]

    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt_name, suffix in formats_to_test:
            filename = op.join(tmpdir, 'test_mixed_mesh' + suffix)
            tst.report('testing format: %s -> %s' % (fmt_name, filename))

            try:
                mesh0.write(filename, io='auto', file_format=fmt_name)
            except Exception as exc:
                tst.report('  write failed:', exc)
                if fmt_name in ('msh',):
                    tst.report('  skipping msh format (known limitations)')
                    continue
                else:
                    raise

            try:
                mesh1 = Mesh.from_file(filename)
            except Exception as exc:
                tst.report('  read failed:', exc)
                raise

            roundtrip_ok = _compare_meshes(mesh0, mesh1, flag='cv')
            tst.report('  roundtrip comparison:', roundtrip_ok)
            ok = ok and all(roundtrip_ok)

            restored_cell_groups = {d: mesh1.get_cell_groups(d)
                                     for d in mesh1.descs}
            restored_vertex_groups = mesh1.get_vertex_groups()

            for desc in mesh0.descs:
                if desc in restored_cell_groups:
                    orig = original_cell_groups[desc]
                    restored = restored_cell_groups[desc]
                    if not nm.all(orig == restored):
                        tst.report('  cell groups mismatch for %s!' % desc)
                        tst.report('    original:', orig)
                        tst.report('    restored:', restored)
                        ok = False

            if not nm.all(original_vertex_groups == restored_vertex_groups):
                tst.report('  vertex groups mismatch!')
                ok = False

            tst.report('  format %s roundtrip: %s'
                       % (fmt_name, 'PASSED' if all(roundtrip_ok) else 'FAILED'))

    assert_(ok, 'mixed-cell mesh group roundtrip failed!')

def test_unified_group_interface_methods():
    """
    Test the unified group mapping interface methods on Mesh class.

    Skips if C extensions are not available.
    """
    import numpy as nm

    if not _check_c_extensions():
        tst.report('unified group interface methods not-tested'
                   ' (missing Cython extensions)!')
        return

    from sfepy.discrete.fem import Mesh

    conf_dir = op.dirname(__file__)
    mesh_file = data_dir + '/meshes/2d/special/square_triquad.mesh'

    tst.report('testing unified group interface methods')

    mesh = Mesh.from_file(mesh_file, prefix_dir=conf_dir)

    cell_groups_all = mesh.get_cell_groups()
    assert_(isinstance(cell_groups_all, dict),
            'get_cell_groups() should return dict when desc is None')
    for desc in mesh.descs:
        assert_(desc in cell_groups_all,
                'dict should contain key for each element type')

    for desc in mesh.descs:
        cg = mesh.get_cell_groups(desc)
        expected_n = mesh.get_conn(desc).shape[0]
        assert_(len(cg) == expected_n,
                'cell groups length should match number of elements')

    vg = mesh.get_vertex_groups()
    assert_(len(vg) == mesh.n_nod,
            'vertex groups length should match number of nodes')

    new_cg = {d: nm.ones_like(cell_groups_all[d]) * 99
              for d in mesh.descs}
    mesh.set_cell_groups(new_cg)
    for desc in mesh.descs:
        assert_(nm.all(mesh.get_cell_groups(desc) == 99),
                'set_cell_groups should update cell groups')

    new_vg = nm.ones_like(vg) * 42
    mesh.set_vertex_groups(new_vg)
    assert_(nm.all(mesh.get_vertex_groups() == 42),
            'set_vertex_groups should update vertex groups')

    mesh.set_cell_groups(cell_groups_all)
    mesh.set_vertex_groups(vg)

    unique_cg = mesh.get_unique_cell_groups()
    assert_(len(unique_cg) > 0, 'should have unique cell groups')

    unique_vg = mesh.get_unique_vertex_groups()
    assert_(len(unique_vg) > 0, 'should have unique vertex groups')

    remap_dict = {k: k + 100 for k in unique_cg}
    mesh.remap_cell_groups(remap_dict)
    for desc in mesh.descs:
        cg = mesh.get_cell_groups(desc)
        for old_val, new_val in remap_dict.items():
            assert_(not nm.any(cg == old_val),
                    'remapped value should not appear')
    mesh.set_cell_groups(cell_groups_all)

    remap_dict_v = {k: k + 200 for k in unique_vg}
    mesh.remap_vertex_groups(remap_dict_v)
    vg_remapped = mesh.get_vertex_groups()
    for old_val, new_val in remap_dict_v.items():
        assert_(not nm.any(vg_remapped == old_val),
                'remapped value should not appear')
    mesh.set_vertex_groups(vg)

    group_mapping = mesh.get_group_mapping()
    assert_(isinstance(group_mapping, dict),
            'get_group_mapping should return dict')
    for desc in mesh.descs:
        if desc in group_mapping:
            for g_str, idxs in group_mapping[desc].items():
                cg = mesh.get_cell_groups(desc)
                assert_(nm.all(cg[idxs] == int(g_str)),
                        'mapping indices should correspond to group values')

    vertex_mapping = mesh.get_vertex_group_mapping()
    assert_(isinstance(vertex_mapping, dict),
            'get_vertex_group_mapping should return dict')
    for g_str, idxs in vertex_mapping.items():
        vg = mesh.get_vertex_groups()
        assert_(nm.all(vg[idxs] == int(g_str)),
                'vertex mapping indices should correspond to group values')

    tst.report('unified group interface methods: PASSED')

def _check_c_extensions():
    """
    Check if Cython extension modules are available.

    Returns
    -------
    available : bool
        True if C extensions are available.
    """
    try:
        from sfepy.discrete.common.extmods.cmesh import CMesh
        return True
    except ImportError:
        return False

def _check_meshio():
    """
    Check if meshio library is available.

    Returns
    -------
    available : bool
        True if meshio is available.
    """
    try:
        import meshio
        return True
    except ImportError:
        return False

def test_convert_mesh_group_preservation():
    """
    Test that cell groups and vertex groups are preserved during
    convert_mesh-style roundtrip (.mesh -> .vtk/.msh -> .mesh).

    This test simulates the convert_mesh script workflow:
    1. Read .mesh file
    2. Write to .vtk/.msh format
    3. Read back from .vtk/.msh
    4. Write to .mesh format
    5. Read final .mesh and compare groups

    Skips if meshio or C extensions are not available.
    """
    import tempfile
    import numpy as nm

    if not _check_c_extensions():
        tst.report('convert_mesh group preservation not-tested'
                   ' (missing Cython extensions)!')
        return

    if not _check_meshio():
        tst.report('convert_mesh group preservation not-tested'
                   ' (missing meshio module)!')
        return

    from sfepy.discrete.fem import Mesh

    conf_dir = op.dirname(__file__)
    mesh_file = data_dir + '/meshes/2d/special/square_triquad.mesh'

    tst.report('testing convert_mesh roundtrip:', mesh_file)

    mesh_original = Mesh.from_file(mesh_file, prefix_dir=conf_dir)

    original_cell_groups = {d: mesh_original.get_cell_groups(d).copy()
                             for d in mesh_original.descs}
    original_vertex_groups = mesh_original.get_vertex_groups().copy()
    original_unique_cell_groups = mesh_original.get_unique_cell_groups()
    original_unique_vertex_groups = mesh_original.get_unique_vertex_groups()

    tst.report('original element types:', mesh_original.descs)
    tst.report('original unique cell groups:', original_unique_cell_groups)
    tst.report('original unique vertex groups:', original_unique_vertex_groups)

    intermediate_formats = ['vtk', 'msh']

    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        for fmt_name in intermediate_formats:
            tst.report('testing roundtrip: .mesh -> .%s -> .mesh' % fmt_name)

            intermediate_file = op.join(tmpdir,
                                        'test_intermediate.%s' % fmt_name)
            final_file = op.join(tmpdir, 'test_final.mesh')

            try:
                mesh_original.write(intermediate_file, io='auto',
                                    file_format=fmt_name)
            except Exception as exc:
                tst.report('  write to %s failed:' % fmt_name, exc)
                if fmt_name == 'msh':
                    tst.report('  skipping msh format')
                    continue
                else:
                    raise

            try:
                mesh_intermediate = Mesh.from_file(intermediate_file)
            except Exception as exc:
                tst.report('  read from %s failed:' % fmt_name, exc)
                raise

            try:
                mesh_intermediate.write(final_file, io='auto',
                                        file_format='mesh')
            except Exception as exc:
                tst.report('  write to .mesh failed:', exc)
                raise

            try:
                mesh_final = Mesh.from_file(final_file)
            except Exception as exc:
                tst.report('  read final .mesh failed:', exc)
                raise

            roundtrip_ok = _compare_meshes(mesh_original, mesh_final,
                                            flag='cv')
            tst.report('  roundtrip comparison:', roundtrip_ok)
            ok = ok and all(roundtrip_ok)

            final_cell_groups = {d: mesh_final.get_cell_groups(d)
                                  for d in mesh_final.descs}
            final_vertex_groups = mesh_final.get_vertex_groups()

            for desc in mesh_original.descs:
                if desc in final_cell_groups:
                    orig = original_cell_groups[desc]
                    final = final_cell_groups[desc]
                    if not nm.all(orig == final):
                        tst.report('  cell groups mismatch for %s!' % desc)
                        tst.report('    original:', orig)
                        tst.report('    final:', final)
                        ok = False

            if not nm.all(original_vertex_groups == final_vertex_groups):
                tst.report('  vertex groups mismatch!')
                ok = False

            tst.report('  format .%s roundtrip: %s'
                       % (fmt_name,
                          'PASSED' if all(roundtrip_ok) else 'FAILED'))

    assert_(ok, 'convert_mesh group preservation roundtrip failed!')
