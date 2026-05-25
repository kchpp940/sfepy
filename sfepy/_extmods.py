"""
Centralized entry point for loading the Cython/C extension modules shipped
with SfePy.

The build system (CMake + scikit-build) produces a set of platform-specific
shared objects (``*.so`` / ``*.pyd``) placed alongside the pure Python
sources in the various ``extmods/`` sub-packages.  In a source tree those
artifacts are missing until the user runs the build step.  Historically,
importing ``sfepy.terms`` or another heavy module would then raise a bare
``ImportError`` for a specific sub-module, which is hard to diagnose.

This module consolidates:

* the registry of extension modules (see :data:`EXTENSIONS`),
* a unified helper for importing a single extension with a guarded,
  user-friendly error message (:func:`import_extension`),
* a bulk importer that can be used from the top-level ``sfepy`` package
  to fail fast with a single, descriptive message (:func:`import_all`),
* and small inspection helpers (:func:`list_extensions`,
  :func:`check_extensions`) used by the test and installation scripts.

The module is intentionally dependency-light so it can be imported even
when no extensions have been compiled yet.
"""

import os
import sys
import platform
from importlib import import_module


EXTENSIONS = {
    'common': (
        'sfepy.discrete.common.extmods',
        ('_fmfield', 'cmapping', 'assemble', 'cmesh',
         'crefcoors', '_geommech'),
    ),
    'fem': (
        'sfepy.discrete.fem.extmods',
        ('bases', 'lobatto_bases'),
    ),
    'iga': (
        'sfepy.discrete.iga.extmods',
        ('igac',),
    ),
    'mechanics': (
        'sfepy.mechanics.extmods',
        ('ccontres',),
    ),
    'terms': (
        'sfepy.terms.extmods',
        ('terms',),
    ),
}


EXTENSION_GROUPS = tuple(EXTENSIONS.keys())


PRESETS = {
    'solver': ('common', 'fem', 'terms'),
    'postprocess': ('common', 'fem'),
    'full': EXTENSION_GROUPS,
}


def list_extensions(groups=None):
    """
    Return the list of fully qualified extension module names known to
    SfePy.

    When *groups* is given (an iterable of group names such as
    ``('common', 'fem')``, or a preset name like ``'solver'``) only the
    extensions belonging to those groups are returned.  Use ``None`` or
    ``'all'`` to include every group.
    """
    if groups is None or groups == 'all':
        groups = EXTENSION_GROUPS
    elif isinstance(groups, str):
        groups = PRESETS.get(groups, (groups,))

    names = []
    for group in groups:
        pkg, mod_names = EXTENSIONS[group]
        for mod in mod_names:
            names.append('%s.%s' % (pkg, mod))
    return names


def _platform_tag():
    """Return a short, human-readable description of the current platform."""
    return '%s-%s (%s)' % (platform.system(), platform.machine(),
                           sys.version.split()[0])


def _ext_suffixes():
    """Return the file suffixes used for compiled extension modules."""
    import importlib.machinery as mach
    return [ext for ext in mach.EXTENSION_SUFFIXES]


def _build_hint():
    return (
        'SfePy Cython/C extensions have not been compiled for %s.\n'
        'Run the build step to produce the shared objects, e.g.:\n'
        '    pip install -e .\n'
        'or, using the classic in-place build:\n'
        '    python setup.py build_ext --inplace\n'
        'Searched extension suffixes: %s'
    ) % (_platform_tag(), ', '.join(_ext_suffixes()))


class ExtensionImportError(ImportError):
    """
    Raised when a compiled SfePy extension cannot be imported.

    Compared to a plain :class:`ImportError` this class carries the name
    of the missing extension and a hint describing how to produce the
    missing artifacts.
    """

    def __init__(self, extension_name, original_error=None):
        self.extension_name = extension_name
        self.original_error = original_error

        msg = ('cannot import compiled extension %r\n%s'
               % (extension_name, _build_hint()))
        if original_error is not None:
            msg += '\nUnderlying error: %s' % (original_error,)

        try:
            super().__init__(msg, name=extension_name, path=None)
        except TypeError:
            super().__init__(msg)


def import_extension(name, package=None):
    """
    Import a compiled extension module.

    Parameters
    ----------
    name : str
        Either an absolute module name (e.g.
        ``'sfepy.terms.extmods.terms'``) or, when *package* is given, a
        module name relative to *package*.
    package : str, optional
        The package the *name* is relative to.

    Returns
    -------
    module
        The imported extension module.

    Raises
    ------
    ExtensionImportError
        If the extension cannot be imported, e.g. because the build step
        has not been run.
    """

    if package is not None:
        full_name = '%s.%s' % (package, name)
    else:
        full_name = name

    try:
        return import_module(full_name)
    except ImportError as exc:
        # Only re-wrap errors that actually correspond to the requested
        # extension.  ImportError may also be raised transitively when
        # the extension itself depends on a further module that cannot
        # be imported – in that case we still want to provide our own
        # hint, but we preserve the original error as context.
        raise ExtensionImportError(full_name, original_error=exc) from exc


def _resolve_package_dir(pkg_name):
    """
    Return the on-disk directory of *pkg_name* without triggering the
    package's ``__init__.py`` (which may transitively try to import
    compiled extensions).

    The function walks the package hierarchy manually using ``sys.path``
    and the file system, so no user code inside any of the sub-packages
    is executed.
    """
    parts = pkg_name.split('.')
    if not parts:
        return None

    # Find the top-level package by walking ``sys.path``.
    top_name = parts[0]
    top_dir = None
    for path in sys.path:
        if not path or not os.path.isdir(path):
            continue
        candidate = os.path.join(path, top_name)
        init = os.path.join(candidate, '__init__.py')
        if os.path.isfile(init):
            top_dir = candidate
            break
    if top_dir is None:
        return None

    current = top_dir
    for part in parts[1:]:
        candidate = os.path.join(current, part)
        init = os.path.join(candidate, '__init__.py')
        if os.path.isfile(init):
            current = candidate
        else:
            # ``part`` is not a regular sub-package – we cannot resolve it
            # without actually importing the parent, so give up rather
            # than trigger a cascade of side effects.
            return None

    return current


def find_extension_path(name, package=None):
    """
    Return the on-disk path of a compiled extension module, or ``None``
    if it is not present.

    This is primarily useful for diagnostic scripts – the function does
    *not* attempt to import the module, it only checks that the
    corresponding shared object exists next to the pure-Python package.
    """

    if package is not None:
        full_name = '%s.%s' % (package, name)
    else:
        full_name = name

    pkg_name, _, mod_name = full_name.rpartition('.')

    pkg_dir = _resolve_package_dir(pkg_name)
    if not pkg_dir:
        return None

    for suffix in _ext_suffixes():
        candidate = os.path.join(pkg_dir, mod_name + suffix)
        if os.path.isfile(candidate):
            return candidate

    return None


def check_extensions(include_private=True, groups=None):
    """
    Inspect the file system and return a mapping from extension name to
    a two-tuple ``(status, info)`` where *status* is ``'ok'`` when the
    compiled object was found and ``'missing'`` otherwise, and *info* is
    the file path (when found) or a short hint string (when not).

    If *include_private* is ``False`` only the names that are commonly
    imported from user code are reported – ``_fmfield`` and ``_geommech``
    are then skipped.

    When *groups* is given (an iterable of group names such as
    ``('common', 'fem')``, or a preset name like ``'solver'``) only the
    extensions belonging to those groups are inspected.
    """

    if isinstance(groups, str):
        groups = PRESETS.get(groups, (groups,))

    report = {}
    for full_name in list_extensions(groups):
        _, _, mod_name = full_name.rpartition('.')
        if (not include_private) and mod_name.startswith('_'):
            continue

        path = find_extension_path(full_name)
        if path is None:
            report[full_name] = ('missing',
                                 'compiled extension not found on disk')
        else:
            report[full_name] = ('ok', path)

    return report


def import_all(raise_on_error=True):
    """
    Try to import every compiled extension.

    When *raise_on_error* is ``True`` (the default) the function either
    returns the list of successfully imported modules or raises a single
    :class:`ExtensionImportError` describing the first missing
    extension.  When ``False`` a tuple ``(modules, errors)`` is
    returned instead, which is handy for diagnostic scripts that want to
    report the full set of problems in one go.
    """

    modules = []
    errors = []
    for full_name in list_extensions():
        try:
            modules.append(import_extension(full_name))
        except ExtensionImportError as exc:
            if raise_on_error:
                raise
            errors.append(exc)

    if raise_on_error:
        return modules

    return modules, errors


def preflight_check(stream=None, exit_on_fail=False, groups=None):
    """
    Verify that the required compiled extension modules are present on
    disk.

    If any extension is missing, print a concise report listing the
    missing modules together with a hint describing how to produce
    the missing artifacts, and return ``False``.  Return ``True`` if
    everything is in place.

    The report is written to *stream* (default: ``sys.stderr``) so that
    it does not get mixed up with normal script output on ``stdout``.

    When *exit_on_fail* is a truthy value, the function calls
    ``sys.exit(exit_on_fail)`` instead of returning ``False``.  Pass an
    integer (e.g. ``2``) to make CI pipelines pick up the failure.

    When *groups* is given (an iterable of group names such as
    ``('common', 'fem', 'terms')``, or a preset name like ``'solver'``)
    only the extensions belonging to those groups are checked.  The
    available presets are defined in :data:`PRESETS`.

    This function is intentionally safe to call even before any heavy
    sub-packages have been imported: it only walks the file system and
    never executes ``__init__.py`` files beyond the top-level
    ``sfepy`` package.
    """

    import sys as _sys

    if stream is None:
        stream = _sys.stderr

    if isinstance(groups, str):
        groups = PRESETS.get(groups, (groups,))

    report = check_extensions(include_private=False, groups=groups)
    missing = [name for name, (status, _) in report.items()
               if status == 'missing']

    if not missing:
        return True

    short_names = [name.rpartition('.')[-1] for name in missing]

    scope_label = ''
    if groups is not None and groups != 'all':
        scope_label = ' [%s]' % ', '.join(groups)

    lines = [
        '',
        '=' * 72,
        'SfePy preflight check%s: compiled extension modules are missing.'
        % scope_label,
        '',
        'The following extension(s) could not be found on disk:',
    ]
    for name in short_names:
        lines.append('    - %s' % name)
    lines.append('')
    lines.append(_build_hint())
    lines.append('=' * 72)
    lines.append('')

    stream.write('\n'.join(lines) + '\n')
    stream.flush()

    if exit_on_fail:
        _sys.exit(int(exit_on_fail))

    return False
