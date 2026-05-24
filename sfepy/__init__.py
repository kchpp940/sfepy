import os, glob

from .config import in_source_tree, top_dir, site_config
from .version import __version__
from .base.deps import (show_deps_info, dependency_info,  # noqa: F401
                        dep_manager, fatal_dependency_error,
                        format_dependency_banner,
                        DependencyMissingError)

data_dir = os.path.realpath(top_dir)
base_dir = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))

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


def diagnostics():
    """
    Print a quick diagnostics summary: SfePy version, installation
    paths and the state of every registered optional dependency.

    This is the function users should call when something fails because
    of a missing dependency – it pinpoints the problem in a single
    glance.
    """
    import sys

    lines = []
    lines.append('SfePy %s' % __version__)
    lines.append('  python  : %s' % sys.version.replace('\n', ' '))
    lines.append('  top_dir : %s' % top_dir)
    lines.append('  base_dir: %s' % base_dir)
    lines.append('  in_source_tree: %s' % in_source_tree)
    print('\n'.join(lines))
    show_deps_info(check_all=True)
