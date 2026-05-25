import os, glob

from .config import in_source_tree, top_dir, site_config
from .version import __version__

# Backward-compatible: data_dir points to the data root (repo root in
# source tree, sfepy package dir when installed), and base_dir is the
# sfepy package directory.
data_dir = os.path.realpath(top_dir)
base_dir = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))

from .base.resources import (  # noqa: E402
    resolve_resource,
    resolve_example,
    resolve_mesh,
    resolve_output,
    locate_resource,
    get_data_dir,
    get_pkg_dir,
    get_example_dir,
    get_mesh_dir,
    ResourceLocator,
)

def get_paths(pattern):
    """
    Get files/paths matching the given pattern in the sfepy source tree.
    """
    if not in_source_tree:
        pattern = '../' + pattern

    files = glob.glob(os.path.normpath(os.path.join(top_dir, pattern)))
    return files

def test(*args):
    """
    Run all the package tests.

    Equivalent to running ``pytest sfepy/tests/`` in the base directory of
    SfePy. Allows an installed version of SfePy to be tested.

    To test an installed version of SfePy use

    .. code-block:: bash

       $ python -c "import sfepy; sfepy.test()"

    Parameters
    ----------
    *args : positional arguments
        Arguments passed to pytest.
    """
    import pytest  # pylint: disable=import-outside-toplevel

    args = list(args)
    if all(arg.startswith('-') for arg in args):
        # Add the default path only if no path is given explicitly.
        path = os.path.join(os.path.split(__file__)[0], 'tests')
        args = [path] + args

    return pytest.main(args=args)
