"""
Central management of optional dependencies and backend-capability
detection for SfePy.

This module exposes a single :class:`DependencyManager` instance,
``dep_manager``, that every SfePy module should use when it needs to
import an optional dependency.  The goals are:

* Keep the "try/except import" idiom in a single place, with
  consistent, human-readable diagnostics.
* Give every missing dependency a standard install hint and a short
  description of what feature will be unavailable.
* Let ``sfepy`` entry points (``simple.py``, ``resview.py``, ...) and
  core modules produce the same failure messages, so that users can
  easily locate the root cause when something is missing.
* Provide a cheap ``sfepy.show_deps_info()`` helper that summarises
  the state of every optional backend at a glance.

Usage inside a module::

    from sfepy.base.deps import dep_manager

    # Import-or-None (downgrade mode) – preferred when the feature is
    # genuinely optional.
    meshio = dep_manager.optional_import('meshio')
    if meshio is None:
        # fall back to a pure-python implementation, skip the feature,
        # or raise later with dep_manager.require().

    # Hard requirement (error mode) – for functions that cannot work
    # without the dependency.
    PETSc = dep_manager.require('petsc4py')
    # raises DependencyMissingError with a helpful install hint

From the top-level ``sfepy`` package::

    import sfepy
    sfepy.show_deps_info()        # prints a nice table
    info = sfepy.dependency_info()  # returns a dict
"""
import importlib
import sys
from collections import OrderedDict


class DependencyMissingError(ImportError):
    """
    Raised by :meth:`DependencyManager.require` when an optional
    dependency cannot be imported.  The message always includes the
    registered install hint so the user knows how to fix the
    situation.
    """


class _Dependency:
    """Metadata for a single optional dependency."""

    __slots__ = ('name', 'import_name', 'install_hint', 'feature',
                 '_module', '_error', '_checked')

    def __init__(self, name, import_name, install_hint, feature):
        self.name = name
        self.import_name = import_name
        self.install_hint = install_hint
        self.feature = feature
        self._module = None
        self._error = None
        self._checked = False

    @property
    def available(self):
        return self._checked and self._module is not None

    @property
    def error(self):
        return self._error

    def _check_once(self, on_missing):
        if self._checked:
            return self._module
        self._checked = True
        try:
            self._module = importlib.import_module(self.import_name)
        except Exception as exc:  # ImportError, ModuleNotFoundError, ...
            self._error = exc
            self._module = None
            on_missing(self)
        return self._module


class DependencyManager:
    """
    Registry + lazy importer for SfePy's optional dependencies.

    Each dependency is registered once via :meth:`register` and can
    then be imported with either :meth:`optional_import` (return
    ``None`` on failure, e.g. for graceful feature downgrade) or
    :meth:`require` (raise a descriptive
    :class:`DependencyMissingError`).

    A callable ``on_missing(name, dep)`` may be supplied; it is
    invoked the first time a missing dependency is detected.  This is
    useful to emit a single warning per session instead of one per
    import site.
    """

    def __init__(self, on_missing=None):
        self._deps = OrderedDict()
        self._on_missing = on_missing or (lambda dep: None)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, name, import_name, install_hint='', feature=''):
        """
        Register an optional dependency.

        Parameters
        ----------
        name : str
            Short identifier, used as a key.  Examples: ``"meshio"``,
            ``"petsc"``, ``"c-ext"``.
        import_name : str
            Module name passed to :func:`importlib.import_module`.
            Use a dotted path for sub-modules such as
            ``"petsc4py.PETSc"``.
        install_hint : str
            Short pip/conda instruction shown to the user when the
            dependency is missing.
        feature : str
            One-line description of the feature that this dependency
            enables.
        """
        if name in self._deps:
            return
        self._deps[name] = _Dependency(name, import_name, install_hint,
                                      feature)

    # ------------------------------------------------------------------
    # import helpers
    # ------------------------------------------------------------------
    def _get(self, name):
        try:
            return self._deps[name]
        except KeyError:
            raise KeyError(
                "dependency %r is not registered.  "
                "Available: %s" % (name, ', '.join(self._deps)))

    def optional_import(self, name):
        """
        Import a registered dependency, returning ``None`` and
        silently recording the failure if the dependency is absent.
        """
        dep = self._get(name)
        return dep._check_once(self._on_missing)

    def require(self, name, context=None):
        """
        Import a registered dependency or raise
        :class:`DependencyMissingError` with a message that explains
        how to install it.

        Parameters
        ----------
        name : str
            Registered dependency name.
        context : str, optional
            Extra text appended to the error message, e.g. the name
            of the feature that cannot run without this dependency.
        """
        dep = self._get(name)
        module = dep._check_once(self._on_missing)
        if module is None:
            parts = [
                "optional dependency %r (import '%s') is not available."
                % (dep.name, dep.import_name),
            ]
            if dep.feature:
                parts.append("  feature: %s" % dep.feature)
            if dep.install_hint:
                parts.append("  install hint: %s" % dep.install_hint)
            if context:
                parts.append("  context: %s" % context)
            if dep.error is not None:
                parts.append("  original error: %s: %s"
                             % (type(dep.error).__name__, dep.error))
            raise DependencyMissingError('\n'.join(parts))
        return module

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def available(self, name):
        """Return ``True`` if the registered dependency is importable."""
        dep = self._get(name)
        if not dep._checked:
            dep._check_once(self._on_missing)
        return dep.available

    def status(self, name):
        """
        Return one of ``"available"``, ``"missing"`` or ``"unknown"``
        for the given dependency name.
        """
        dep = self._get(name)
        if not dep._checked:
            return 'unknown'
        return 'available' if dep.available else 'missing'

    def summary(self, check_all=False):
        """
        Return an :class:`collections.OrderedDict` mapping dependency
        names to ``(status, feature, install_hint, error)`` tuples.

        If ``check_all`` is True every registered dependency is
        probed before the summary is built.
        """
        if check_all:
            for dep in self._deps.values():
                dep._check_once(self._on_missing)
        out = OrderedDict()
        for name, dep in self._deps.items():
            out[name] = (self.status(name), dep.feature,
                         dep.install_hint, dep.error)
        return out

    def format_summary(self, check_all=False):
        """Return a human-readable, multi-line summary string."""
        lines = ['SfePy optional dependencies:']
        for name, (status, feature, hint, err) in self.summary(check_all).items():
            tag = {'available': '[OK]',
                   'missing':   '[MISSING]',
                   'unknown':   '[?]'}.get(status, status)
            line = '  %-9s %-12s' % (tag, name)
            if feature:
                line += '  -- %s' % feature
            if status == 'missing' and hint:
                line += '\n             install: %s' % hint
            if status == 'missing' and err is not None:
                line += '\n             reason:  %s: %s' % (type(err).__name__, err)
            lines.append(line)
        return '\n'.join(lines)


# ----------------------------------------------------------------------
# Default global instance + canonical registry
# ----------------------------------------------------------------------

dep_manager = DependencyManager()

# ------------------------------------------------------------------
# IO / meshing
# ------------------------------------------------------------------
dep_manager.register(
    'meshio', 'meshio',
    install_hint='pip install meshio',
    feature='reading and writing meshes in formats other than '
            'SfePy/HDF5 (e.g. vtk, med, msh, xdmf, ...)',
)

# ------------------------------------------------------------------
# Parallel / linear algebra backends
# ------------------------------------------------------------------
dep_manager.register(
    'mpi4py', 'mpi4py',
    install_hint='conda install -c conda-forge mpi4py  (or build '
                 'mpi4py against your system MPI)',
    feature='basic MPI communication (required together with '
            'petsc4py for distributed solvers)',
)
dep_manager.register(
    'petsc4py', 'petsc4py',
    install_hint='conda install -c conda-forge petsc4py  (or build '
                 'PETSc from source and install petsc4py)',
    feature='distributed linear/non-linear solvers and '
            'partitioned meshes',
)
dep_manager.register(
    'petsc4py.PETSc', 'petsc4py.PETSc',
    install_hint='install PETSc + petsc4py (see "petsc4py" entry)',
    feature='PETSc API object (PETSc Mat/Vec/KSP, ...)',
)
dep_manager.register(
    'slepc4py', 'slepc4py',
    install_hint='conda install -c conda-forge slepc4py',
    feature='SLEPc eigenvalue solvers (used together with PETSc)',
)
dep_manager.register(
    'pyamg', 'pyamg',
    install_hint='pip install pyamg',
    feature='algebraic multigrid preconditioner for scipy-based '
            'linear solvers',
)
dep_manager.register(
    'pymetis', 'pymetis',
    install_hint='pip install pymetis  (requires METIS)',
    feature='graph partitioning for distributed meshes (fallback: '
            'naive partitioning)',
)

# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------
dep_manager.register(
    'matplotlib', 'matplotlib',
    install_hint='pip install matplotlib',
    feature='2D plotting (all ``plot_*.py`` scripts, logs, '
            'condition numbers, ...)',
)
dep_manager.register(
    'mayavi', 'mayavi',
    install_hint='conda install -c conda-forge mayavi',
    feature='3D interactive visualisation of FE results',
)
dep_manager.register(
    'pyvista', 'pyvista',
    install_hint='pip install pyvista',
    feature='alternative 3D visualisation backend (VTK-based)',
)

# ------------------------------------------------------------------
# Symbolic / auxiliary
# ------------------------------------------------------------------
dep_manager.register(
    'sympy', 'sympy',
    install_hint='pip install sympy',
    feature='symbolic differentiation, term verification and '
            'material-coefficient utilities',
)
dep_manager.register(
    'IPython', 'IPython',
    install_hint='pip install ipython',
    feature='interactive IPython shell used by the debug helpers',
)
dep_manager.register(
    'tables', 'tables',
    install_hint='pip install tables  (or conda install -c conda-forge pytables)',
    feature='HDF5 file I/O via PyTables (used for result storage)',
)

# ------------------------------------------------------------------
# SfePy's own C extension modules
# ------------------------------------------------------------------
dep_manager.register(
    'c-ext-common', 'sfepy.discrete.common.extmods',
    install_hint='build SfePy in-place with "make" or install '
                 'the package ("pip install -e ."); a C compiler '
                 'and the Python development headers are required',
    feature='compiled C helpers shared by all discretisations '
            '(reference mappings, geometry, ...)',
)
dep_manager.register(
    'c-ext-common.cmapping', 'sfepy.discrete.common.extmods.cmapping',
    install_hint='build SfePy in-place with "make" or install '
                 'the package ("pip install -e ."); a C compiler '
                 'and the Python development headers are required',
    feature='compiled C reference mapping helpers',
)
dep_manager.register(
    'c-ext-fem', 'sfepy.discrete.fem.extmods',
    install_hint='build SfePy in-place with "make" or install '
                 'the package ("pip install -e ."); a C compiler '
                 'and the Python development headers are required',
    feature='compiled C helpers used by the FEM discretisation '
            '(assembling, mappings, quadratures, ...)',
)
dep_manager.register(
    'c-ext-terms', 'sfepy.terms.extmods',
    install_hint='build SfePy in-place with "make" or install '
                 'the package ("pip install -e .")',
    feature='compiled C helpers for the term library',
)
dep_manager.register(
    'c-ext-terms.terms', 'sfepy.terms.extmods.terms',
    install_hint='build SfePy in-place with "make" or install '
                 'the package ("pip install -e .")',
    feature='compiled C term implementations',
)
dep_manager.register(
    'c-ext-iga', 'sfepy.discrete.iga.extmods',
    install_hint='build SfePy in-place with "make" or install '
                 'the package ("pip install -e .")',
    feature='compiled C helpers used by the isogeometric '
            'discretisation',
)


# ----------------------------------------------------------------------
# Convenience helpers, re-exported by ``sfepy.__init__`` so users can
# type ``sfepy.show_deps_info()`` after a plain ``import sfepy``.
# ----------------------------------------------------------------------

def dependency_info(check_all=True):
    """Return the status dict for every registered optional dependency."""
    return dep_manager.summary(check_all=check_all)


def show_deps_info(check_all=True, stream=None):
    """Print a human-readable summary of the optional dependencies."""
    stream = stream if stream is not None else sys.stdout
    stream.write(dep_manager.format_summary(check_all=check_all))
    stream.write('\n')


def format_dependency_banner(exc):
    """Return the canonical banner used when a script entry point aborts on
    a :class:`DependencyMissingError`.

    The banner is shared by every top-level script (``simple.py``,
    ``convert_mesh.py``, ``resview.py``, ``probe.py``, ...) so that users
    always see the same information: which dependency is missing, which
    feature is affected, how to install it, and where to look for a full
    diagnostic report.
    """
    lines = []
    lines.append('=== SfePy dependency error ===')
    if isinstance(exc, DependencyMissingError):
        lines.append(str(exc))
    else:
        lines.append('%s: %s' % (type(exc).__name__, exc))
    lines.append('')
    lines.append(
        'Run `python -c "import sfepy; sfepy.diagnostics()"` for '
        'a full summary of every optional dependency.'
    )
    return '\n'.join(lines)


def fatal_dependency_error(exc, stream=None):
    """Print the canonical dependency-error banner to *stream* (default
    ``sys.stderr``) and raise ``SystemExit(1)``.

    Call this from script entry points when a :class:`DependencyMissingError`
    bubbles up so that the same message is shown regardless of the entry
    point::

        try:
            main()
        except DependencyMissingError as exc:
            fatal_dependency_error(exc)
    """
    stream = stream if stream is not None else sys.stderr
    stream.write(format_dependency_banner(exc))
    stream.write('\n')
    raise SystemExit(1)


__all__ = [
    'DependencyManager',
    'DependencyMissingError',
    'dep_manager',
    'dependency_info',
    'show_deps_info',
    'format_dependency_banner',
    'fatal_dependency_error',
]
