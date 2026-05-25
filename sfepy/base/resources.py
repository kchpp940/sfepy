"""
Unified resource location mechanism for SfePy.

This module provides a consistent way to locate data files (meshes,
example configurations, output samples, documentation references)
regardless of whether SfePy is being run from a source tree, an
installed package, or a working directory outside the project.

Usage
-----
>>> from sfepy.base.resources import resolve_resource, get_data_dir
>>> mesh_path = resolve_resource('meshes/3d/cylinder.mesh')
>>> data_dir = get_data_dir()

For finding example files:

>>> from sfepy.base.resources import resolve_example
>>> example_path = resolve_example('diffusion/poisson.py')
"""
import os
import os.path as op
import sys
from warnings import warn


def _get_pkg_dir():
    """Return the sfepy package directory (absolute, real path)."""
    return op.dirname(op.dirname(op.normpath(op.realpath(__file__))))


def _get_install_data_dirs():
    """
    Return directories where ``data_files`` may have installed data.

    ``setuptools`` ``data_files`` with prefix ``'sfepy'`` installs files
    under ``<sys.prefix>/sfepy/`` (or ``<sys.base_prefix>/sfepy/`` for
    system installs).  This function returns those candidate paths.
    """
    dirs = []
    for prefix in (sys.prefix, sys.base_prefix):
        candidate = op.normpath(op.realpath(op.join(prefix, 'sfepy')))
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def _get_data_dir():
    """
    Determine the primary data directory.

    In a source tree, this is the repository root (where ``LICENSE`` lives).
    In an installed package, ``data_files`` may place ``LICENSE`` under
    ``<sys.prefix>/sfepy/``, so we check there as well.

    As a last resort we check for the ``meshes`` subdirectory as a marker.
    """
    pkg_dir = _get_pkg_dir()

    candidates = []
    for up_dir in ('..', '.'):
        candidates.append(op.normpath(op.realpath(op.join(pkg_dir, up_dir))))
    candidates.extend(_get_install_data_dirs())

    for candidate in candidates:
        if op.isfile(op.join(candidate, 'LICENSE')):
            return candidate

    for candidate in candidates:
        if op.isdir(op.join(candidate, 'meshes')):
            return candidate

    warn('cannot determine SfePy data directory; falling back to package dir')
    return pkg_dir


def _get_resource_dirs():
    """
    Build the ordered list of directories to search for resources.

    Order:
    1. Current working directory (for user-supplied resources)
    2. Data directory (repo root in source, <prefix>/sfepy in installed)
    3. Package directory (sfepy/)
    4. Parent of package directory (one level above sfepy/)
    5. Install data directories (<sys.prefix>/sfepy, <sys.base_prefix>/sfepy)
    """
    dirs = []

    cwd = os.getcwd()
    if cwd not in dirs:
        dirs.append(cwd)

    data_dir = _get_data_dir()
    if data_dir not in dirs:
        dirs.append(data_dir)

    pkg_dir = _get_pkg_dir()
    if pkg_dir not in dirs:
        dirs.append(pkg_dir)

    parent_dir = op.dirname(pkg_dir)
    if parent_dir not in dirs:
        dirs.append(parent_dir)

    for d in _get_install_data_dirs():
        if d not in dirs:
            dirs.append(d)

    return dirs


_resource_dirs = None
_data_dir = None


def _init_resource_dirs():
    """Initialize the module-level resource directory cache."""
    global _resource_dirs, _data_dir

    _data_dir = _get_data_dir()
    _resource_dirs = _get_resource_dirs()


_init_resource_dirs()


def get_data_dir():
    """
    Return the primary data directory.

    This is the directory where mesh data and other bundled resources
    are expected to live.  In a source tree it is the repository root;
    in an installed package it is ``<sys.prefix>/sfepy/`` (the location
    where ``data_files`` are installed by ``setuptools``).
    """
    return _data_dir


def get_pkg_dir():
    """Return the sfepy package directory."""
    return _get_pkg_dir()


def get_example_dir():
    """Return the ``sfepy/examples`` directory."""
    return op.join(_get_pkg_dir(), 'examples')


def get_mesh_dir():
    """Return the ``meshes`` directory."""
    return op.join(get_data_dir(), 'meshes')


def resolve_resource(relative_path, search_dirs=None, must_exist=False):
    """
    Resolve a relative resource path by searching in the configured
    resource directories.

    Parameters
    ----------
    relative_path : str
        A path relative to one of the resource directories, e.g.
        ``'meshes/3d/cylinder.mesh'`` or
        ``'examples/diffusion/poisson.py'``.
    search_dirs : list of str, optional
        Directories to search in order.  If ``None`` the default
        resource directories are used.
    must_exist : bool, optional
        If ``True``, raise ``FileNotFoundError`` when the resource
        cannot be found.  If ``False`` (default) return the
        first-probe path even if it does not exist.

    Returns
    -------
    str
        The absolute path to the resource, or the first attempted
        absolute path when ``must_exist`` is ``False``.
    """
    if op.isabs(relative_path):
        return relative_path

    if search_dirs is None:
        search_dirs = _resource_dirs

    last_attempt = None
    for d in search_dirs:
        full = op.normpath(op.join(d, relative_path))
        if op.exists(full):
            return full
        if last_attempt is None:
            last_attempt = full

    if must_exist and (last_attempt is None or not op.exists(last_attempt)):
        raise FileNotFoundError(
            f'resource not found: {relative_path!r}; '
            f'searched in: {search_dirs}'
        )

    return last_attempt if last_attempt is not None else op.join(
        search_dirs[0], relative_path
    )


def resolve_example(example_path, must_exist=False):
    """
    Resolve the path to an example file under ``sfepy/examples``.

    Parameters
    ----------
    example_path : str
        Path relative to the examples directory, e.g.
        ``'diffusion/poisson.py'``.
    must_exist : bool, optional
        Raise ``FileNotFoundError`` if the file is missing.

    Returns
    -------
    str
        Absolute path to the example file.
    """
    return resolve_resource(op.join('examples', example_path),
                            must_exist=must_exist)


def resolve_mesh(mesh_path, must_exist=False):
    """
    Resolve the path to a mesh file under ``meshes/``.

    Parameters
    ----------
    mesh_path : str
        Path relative to the meshes directory, e.g.
        ``'3d/cylinder.mesh'``.
    must_exist : bool, optional
        Raise ``FileNotFoundError`` if the file is missing.

    Returns
    -------
    str
        Absolute path to the mesh file.
    """
    return resolve_resource(op.join('meshes', mesh_path),
                            must_exist=must_exist)


def resolve_output(relative_path, output_dir=None):
    """
    Resolve an output file path.

    If *output_dir* is given the result is relative to that directory;
    otherwise the current working directory is used.

    Parameters
    ----------
    relative_path : str
        Relative path for the output file, e.g. ``'results.vtk'``.
    output_dir : str, optional
        Base directory for output.  Defaults to ``os.getcwd()``.

    Returns
    -------
    str
        Absolute path for the output file.
    """
    if op.isabs(relative_path):
        return relative_path

    base = output_dir if output_dir else os.getcwd()
    return op.normpath(op.join(base, relative_path))


def locate_resource(pattern, root_dir=None, recursive=True):
    """
    Locate files matching a glob pattern inside the resource
    directories.

    Parameters
    ----------
    pattern : str
        Glob pattern, e.g. ``'*.mesh'``.
    root_dir : str, optional
        Sub-directory inside each resource dir to search, e.g.
        ``'meshes'``.  Defaults to searching the whole resource dir.
    recursive : bool, optional
        If ``True`` (default) use ``**``-style recursive glob.

    Yields
    ------
    str
        Absolute paths of matching files.
    """
    import glob

    if recursive:
        pattern = op.join('**', pattern)

    seen = set()
    for d in _resource_dirs:
        search_root = op.join(d, root_dir) if root_dir else d
        for match in glob.glob(op.join(search_root, pattern),
                               recursive=recursive):
            real = op.realpath(match)
            if real not in seen:
                seen.add(real)
                yield match


def get_paths(pattern):
    """
    Get files/paths matching the given pattern in the sfepy source tree.

    This is a backwards-compatible wrapper around
    :func:`locate_resource`.
    """
    import glob

    from sfepy.config import in_source_tree

    data_dir = get_data_dir()
    if not in_source_tree:
        pattern = '../' + pattern

    files = glob.glob(op.normpath(op.join(data_dir, pattern)))
    return files


class ResourceLocator:
    """
    Object-oriented resource locator.

    Useful when you need to resolve many resources against a common
    set of base directories, e.g. inside a single problem description
    file.

    Parameters
    ----------
    base_dirs : list of str, optional
        Directories to search, in priority order.  If ``None`` the
        default resource directories are used.
    """

    def __init__(self, base_dirs=None):
        self._base_dirs = base_dirs if base_dirs is not None else list(
            _resource_dirs
        )

    @property
    def base_dirs(self):
        return list(self._base_dirs)

    def resolve(self, relative_path, must_exist=False):
        """Resolve *relative_path* using this locator's directories."""
        return resolve_resource(relative_path, self._base_dirs,
                                must_exist=must_exist)

    def resolve_mesh(self, mesh_path, must_exist=False):
        return resolve_resource(
            op.join('meshes', mesh_path), self._base_dirs,
            must_exist=must_exist
        )

    def resolve_example(self, example_path, must_exist=False):
        return resolve_resource(
            op.join('examples', example_path), self._base_dirs,
            must_exist=must_exist
        )

    def resolve_output(self, relative_path, output_dir=None):
        return resolve_output(relative_path, output_dir)
