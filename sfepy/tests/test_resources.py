import os
import os.path as op

from sfepy.base.base import assert_
from sfepy.base.resources import (
    resolve_resource,
    resolve_mesh,
    resolve_example,
    resolve_output,
    locate_resource,
    get_data_dir,
    get_pkg_dir,
    get_example_dir,
    get_mesh_dir,
    ResourceLocator,
)


def test_get_data_dir():
    data_dir = get_data_dir()
    assert_(op.isabs(data_dir))
    assert_(op.isdir(data_dir))
    assert_(op.isfile(op.join(data_dir, 'LICENSE')))


def test_get_pkg_dir():
    pkg_dir = get_pkg_dir()
    assert_(op.isabs(pkg_dir))
    assert_(op.isdir(pkg_dir))
    assert_(op.isfile(op.join(pkg_dir, '__init__.py')))
    assert_(op.basename(pkg_dir) == 'sfepy')


def test_get_example_dir():
    example_dir = get_example_dir()
    assert_(op.isdir(example_dir))
    assert_(op.basename(example_dir) == 'examples')


def test_get_mesh_dir():
    mesh_dir = get_mesh_dir()
    assert_(op.isdir(mesh_dir))
    assert_(op.basename(mesh_dir) == 'meshes')


def test_resolve_resource():
    path = resolve_resource('meshes/3d/cylinder.mesh')
    assert_(op.isabs(path))
    assert_(op.isfile(path))


def test_resolve_resource_not_found():
    path = resolve_resource('nonexistent_file.xyz')
    assert_(op.isabs(path))


def test_resolve_resource_must_exist():
    from sfepy.base.resources import resolve_resource as rr
    try:
        rr('nonexistent_file.xyz', must_exist=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('expected FileNotFoundError')


def test_resolve_resource_absolute():
    abs_path = op.realpath('/tmp/test.txt')
    path = resolve_resource(abs_path)
    assert_(path == abs_path)


def test_resolve_mesh():
    path = resolve_mesh('3d/cylinder.mesh')
    assert_(op.isabs(path))
    assert_(op.isfile(path))
    assert_(path.endswith('meshes/3d/cylinder.mesh'))


def test_resolve_example():
    path = resolve_example('diffusion/poisson.py')
    assert_(op.isabs(path))
    assert_(op.isfile(path))
    assert_(path.endswith('examples/diffusion/poisson.py'))


def test_resolve_output():
    path = resolve_output('results.vtk')
    assert_(op.isabs(path))
    assert_(path.endswith('results.vtk'))


def test_resolve_output_with_dir():
    path = resolve_output('results.vtk', output_dir='/custom/out')
    assert_(path == '/custom/out/results.vtk')


def test_locate_resource():
    results = list(locate_resource('cylinder.mesh', root_dir='meshes'))
    assert_(len(results) >= 1)
    assert_(any('cylinder.mesh' in r for r in results))


def test_resource_locator_class():
    locator = ResourceLocator()
    path = locator.resolve_mesh('3d/cylinder.mesh')
    assert_(op.isfile(path))

    path2 = locator.resolve_example('diffusion/poisson.py')
    assert_(op.isfile(path2))

    base_dirs = locator.base_dirs
    assert_(len(base_dirs) >= 2)
    assert_(get_data_dir() in base_dirs)


def test_resource_locator_custom_dirs(tmp_path):
    custom_dir = str(tmp_path)
    locator = ResourceLocator(base_dirs=[custom_dir])
    path = locator.resolve('test.txt')
    assert_(path.startswith(custom_dir))


def test_backward_compat_data_dir():
    from sfepy import data_dir
    assert_(op.isabs(data_dir))
    assert_(op.isdir(data_dir))
    assert_(data_dir == get_data_dir())


def test_backward_compat_base_dir():
    from sfepy import base_dir
    assert_(op.isabs(base_dir))
    assert_(op.isdir(base_dir))
    assert_(base_dir == get_pkg_dir())


def test_resolve_from_external_cwd(tmp_path):
    orig_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))

        data_dir = get_data_dir()
        assert_(op.isdir(data_dir))

        mesh_path = resolve_mesh('3d/cylinder.mesh')
        assert_(op.isfile(mesh_path))

        example_path = resolve_example('diffusion/poisson.py')
        assert_(op.isfile(example_path))

        output_path = resolve_output('test_out.vtk')
        assert_(op.isabs(output_path))
        assert_(str(tmp_path) in output_path)

    finally:
        os.chdir(orig_cwd)


def test_indir_works_externally(tmp_path):
    from sfepy.base.ioutils import InDir

    orig_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))

        example_path = resolve_example('diffusion/poisson.py')
        indir = InDir(example_path)
        assert_(op.isdir(indir.dir))

        other = indir('other_file.txt')
        assert_(op.isabs(other))
        assert_(indir.dir in other)

    finally:
        os.chdir(orig_cwd)