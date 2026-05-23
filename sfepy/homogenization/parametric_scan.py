"""
Parametric-scan helpers for the homogenization engine.

This module provides two ways of running a parametric sweep:

1. **App-native mode** (recommended).  Declare the sweep families directly
   in the problem file ``options`` dict and run
   ``HomogenizationApp(conf, None, None)()`` -- the app detects the sweep
   options and drives everything itself, producing a merged
   :class:`Coefficients` plus a ``*_sweep.h5`` archive and a
   ``*_sweep.index.json`` recovery index.  See the docstring of
   :func:`run_from_conf` for a quick example.

2. **Programmatic mode**.  :class:`ParametricSweep` builds on top of the
   app-native helpers and is kept for backwards compatibility / for use from
   Jupyter notebooks.  It now delegates to :func:`run_from_conf` internally.

Typical usage (app-native)::

    # in my_micro.py:
    options = {
        'coefs': 'coefs',
        'requirements': 'requirements',
        'volume': {'expression': 'ev_volume.i.Y(u)'},
        'output_dir': 'output',
        'coefs_filename': 'coefs',
        'sweep_materials': [
            {'D': {'Ym': Dm, 'Yc': Dc_soft}},
            {'D': {'Ym': Dm, 'Yc': Dc_hard}},
        ],
        'sweep_frequencies': [0.0, 10.0, 20.0],
        'sweep_group_by': ['frequency'],
    }

    # then from the command line:
    #   sfepy-run my_micro.py
    # or from Python:
    #   from sfepy.homogenization.parametric_scan import run_from_conf
    #   merged = run_from_conf('my_micro.py')
    #   merged.group_by('frequency')
    #   merged.select_case('frequency_10__material_index_1')
"""

from __future__ import print_function
import itertools as it
import os
import os.path as op
import re
import json

import numpy as nm

from sfepy.base.base import Struct


# ---------------------------------------------------------------------------
# Helpers (also used by HomogenizationApp internally)
# ---------------------------------------------------------------------------
_SAFE = re.compile(r'[^A-Za-z0-9_.\-+]')


def _sanitize(txt):
    return _SAFE.sub('_', str(txt))


def _format_scalar(v):
    if isinstance(v, float):
        return '%g' % v
    return _sanitize(v)


def _label_from_params(params):
    """Build a compact, filesystem-safe label from a flat params dict."""
    parts = []
    for k in sorted(params.keys()):
        if k in ('materials', 'regions'):
            continue
        v = params[k]
        if isinstance(v, (tuple, list)):
            vstr = 'x'.join(_format_scalar(x) for x in v)
        else:
            vstr = _format_scalar(v)
        parts.append('%s_%s' % (_sanitize(k), vstr))
    return '__'.join(parts) if parts else 'base'


def _apply_materials(problem_conf, materials):
    if materials is None:
        return
    conf_mats = getattr(problem_conf, 'materials', None)
    if conf_mats is None:
        problem_conf.materials = materials
        return
    if isinstance(conf_mats, dict) and isinstance(materials, dict):
        for mk, mval in materials.items():
            if (mk in conf_mats
                    and isinstance(conf_mats[mk], (tuple, list))
                    and isinstance(mval, (tuple, list))):
                conf_mats[mk] = type(conf_mats[mk])(mval)
            else:
                conf_mats[mk] = mval
    else:
        problem_conf.materials = materials


def _apply_regions(problem_conf, regions):
    if regions is None:
        return
    if isinstance(regions, dict):
        cur = getattr(problem_conf, 'regions', None)
        if isinstance(cur, dict):
            cur.update(regions)
        else:
            problem_conf.regions = regions
    else:
        problem_conf.regions = regions


def _apply_frequency(problem_conf, frequency):
    if frequency is None:
        return
    problem_conf.omega = 2.0 * nm.pi * frequency


# ---------------------------------------------------------------------------
# Public entry point: run a sweep directly from a problem conf
# ---------------------------------------------------------------------------
def run_from_conf(conf, verbose=True):
    """Run a homogenization sweep described by a problem configuration.

    The configuration is expected to carry its sweep parameters in the
    ``options`` dict (``sweep_materials``, ``sweep_frequencies``,
    ``sweep_regions``, ``sweep_combinations``, etc. -- see
    :class:`sfepy.homogenization.homogen_app.HomogenizationApp`).

    Parameters
    ----------
    conf : str or ProblemConf
        Path to a problem-description file, or an already-loaded
        ``ProblemConf``.
    verbose : bool
        Passed through to ``HomogenizationApp.call``.

    Returns
    -------
    merged : Coefficients
        A merged ``Coefficients`` with ``labels``, ``_sweep_cases`` and
        the methods ``select_case`` / ``group_by``.
    """
    from sfepy.base.conf import ProblemConf, get_standard_keywords
    from sfepy.homogenization.homogen_app import HomogenizationApp

    if isinstance(conf, str):
        required, other = get_standard_keywords()
        conf = ProblemConf.from_file(conf, required, other)

    app = HomogenizationApp(conf, None, None)
    return app(verbose=verbose)


# ---------------------------------------------------------------------------
# CaseResults (kept for backwards compatibility)
# ---------------------------------------------------------------------------
class CaseResults(object):
    """Light-weight container for a batch of homogenization runs.

    Attributes
    ----------
    cases : list of CaseEntry
        Each entry holds ``label``, ``flat_index``, ``params``, ``coefs``
        (a ``Coefficients`` instance) and ``output_name``.
    coef_names : list of str
        The names of all coefficient attributes seen across the cases.
    """

    def __init__(self, cases, coef_names):
        self.cases = list(cases)
        self.coef_names = list(coef_names)

    def __len__(self):
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.cases[item]
        for c in self.cases:
            if c.label == item:
                return c
        raise KeyError('no case with label %r' % item)

    def labels(self):
        return [c.label for c in self.cases]

    def params(self):
        return [c.params for c in self.cases]

    def coefficients(self, name):
        return [getattr(c.coefs, name, None) for c in self.cases]

    def stack(self, name, fill=None):
        vals = [getattr(c.coefs, name, None) for c in self.cases]
        clean = [v for v in vals if v is not None]
        if not clean:
            return nm.empty(0)
        arr = nm.asarray(clean[0])
        if arr.ndim == 0:
            out = nm.full(len(vals), fill if fill is not None else nm.nan)
            for i, v in enumerate(vals):
                if v is not None:
                    out[i] = v
            return out
        shape = (len(vals),) + arr.shape
        out = nm.empty(shape, dtype=arr.dtype)
        for i, v in enumerate(vals):
            if v is None:
                out[i] = fill if fill is not None else nm.nan
            else:
                out[i] = v
        return out

    def group_by(self, *keys):
        groups = {}
        for c in self.cases:
            key = tuple(c.params.get(k) for k in keys)
            groups.setdefault(key, []).append(c)
        if len(keys) == 1:
            return {k[0]: v for k, v in groups.items()}
        return groups

    def to_dict(self):
        out = {}
        for name in self.coef_names:
            vals = [getattr(c.coefs, name, None) for c in self.cases]
            try:
                out[name] = nm.array(vals)
            except (ValueError, TypeError):
                arr = nm.empty(len(vals), dtype=object)
                arr[:] = vals
                out[name] = arr
        return out

    def save(self, filename, index_filename=None):
        from sfepy.base.ioutils import write_dict_hdf5

        index = []
        payload = {
            'labels': nm.array([c.label for c in self.cases],
                               dtype='S%d' % max(1, max(len(c.label)
                                                       for c in self.cases))),
            'flat_index': nm.arange(len(self.cases), dtype=nm.int32),
        }
        for c in self.cases:
            entry = {'label': c.label, 'flat_index': c.flat_index,
                     'params': c.params, 'output_name': c.output_name}
            index.append(entry)
            cgrp = {}
            for name in self.coef_names:
                v = getattr(c.coefs, name, None)
                if v is None:
                    continue
                if isinstance(v, nm.ndarray):
                    cgrp[name] = v
                elif isinstance(v, (float, int)):
                    cgrp[name] = nm.asarray(v)
            if cgrp:
                payload['cases/%s' % c.label] = cgrp

        write_dict_hdf5(filename, payload)

        if index_filename is None:
            base, _ = op.splitext(filename)
            index_filename = base + '.index.json'
        with open(index_filename, 'w') as fd:
            json.dump(index, fd, indent=2, default=_json_fallback)


def _json_fallback(obj):
    if isinstance(obj, nm.ndarray):
        return obj.tolist()
    if isinstance(obj, (nm.floating,)):
        return float(obj)
    if isinstance(obj, (nm.integer,)):
        return int(obj)
    raise TypeError('Object of type %s is not JSON serializable'
                    % type(obj).__name__)


def load_case_results(filename):
    """Restore a sweep archive into ``(labels, flat_index, data)``."""
    import tables as pt
    fd = pt.open_file(filename, mode='r')
    try:
        labels = [s.decode() if isinstance(s, bytes) else s
                  for s in fd.root._f_get_child('labels').read()]
        flat_index = list(fd.root._f_get_child('flat_index').read())
        data = {}
        cases = fd.root._f_get_child('cases')
        for label in labels:
            cgrp = cases._f_get_child(label)
            for leaf in cgrp._f_leaves.values():
                coef_name = leaf._v_name
                data[(label, coef_name)] = leaf.read()
    finally:
        fd.close()
    return labels, flat_index, data


def _default_index_path(h5_or_index):
    """Return the ``*_sweep.index.json`` path derived from a ``*_sweep.h5``
    path, or the input unchanged if it is already a json path."""
    if h5_or_index.endswith('.h5'):
        base = h5_or_index[:-len('.h5')]
        if base.endswith('_sweep'):
            return base + '.index.json'
        return base + '_sweep.index.json'
    return h5_or_index


class SweepIndex(object):
    """High-level entry point to a saved parametric-sweep index.

    The class loads the ``*_sweep.index.json`` file (optionally deriving it
    from the ``*_sweep.h5`` path) and exposes per-label accessors for the
    coefficient file, the recovery file and the associated context dict.

    Parameters
    ----------
    index_file : str
        Path to ``*_sweep.index.json``, or to ``*_sweep.h5`` (in which case
        the index path is derived automatically).

    Examples
    --------
    >>> idx = SweepIndex('output/coefs_sweep.h5')
    >>> idx.labels
    ['frequency_0__material_index_0', 'frequency_0__material_index_1', ...]
    >>> coefs = idx.get_coefs('frequency_10__material_index_0')
    >>> rec = idx.get_recovery('frequency_10__material_index_0')
    >>> ctx = idx.get_context('frequency_10__material_index_0')
    """

    def __init__(self, index_file):
        path = _default_index_path(index_file)
        with open(path, 'r') as fd:
            data = json.load(fd)

        self._path = path
        self.meta = data.get('meta', {})
        raw_cases = data.get('cases', [])
        self._by_label = {c['label']: c for c in raw_cases}
        self._by_index = {c['flat_index']: c for c in raw_cases}
        self.labels = list(self._by_label.keys())

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def _entry(self, label_or_index):
        if isinstance(label_or_index, int):
            return self._by_index.get(label_or_index)
        return self._by_label.get(label_or_index)

    def has(self, label_or_index):
        """Return ``True`` if the index contains an entry for the given
        label or flat index."""
        return self._entry(label_or_index) is not None

    def __contains__(self, item):
        return self.has(item)

    def __len__(self):
        return len(self._by_label)

    def __iter__(self):
        return iter(self.labels)

    # ------------------------------------------------------------------
    # Context access
    # ------------------------------------------------------------------
    def get_context(self, label_or_index):
        """Return the ``context`` dict for a given label / flat index, or
        ``None`` if unknown."""
        e = self._entry(label_or_index)
        if e is None:
            return None
        return e.get('context')

    # ------------------------------------------------------------------
    # Coefficient access
    # ------------------------------------------------------------------
    def get_coefs_file(self, label_or_index):
        """Return the absolute path to the per-case coefficients HDF5 file,
        or ``None`` if unknown / missing."""
        e = self._entry(label_or_index)
        if e is None:
            return None
        return e.get('coefs_file')

    def get_coefs(self, label_or_index):
        """Load and return the per-case ``Coefficients`` for a given label
        or flat index."""
        from sfepy.homogenization.coefficients import Coefficients

        path = self.get_coefs_file(label_or_index)
        if path is None or not op.exists(path):
            raise IOError('coefficient file not found for %r: %s'
                          % (label_or_index, path))
        return Coefficients.from_file_hdf5(path)

    # ------------------------------------------------------------------
    # Recovery access
    # ------------------------------------------------------------------
    def get_recovery_file(self, label_or_index):
        """Return the absolute path to the per-case recovery HDF5 file, or
        ``None`` if the case was not run with recovery."""
        e = self._entry(label_or_index)
        if e is None:
            return None
        return e.get('recovery_file')

    def get_recovery_vtk(self, label_or_index):
        """Return the absolute path to the per-case recovery VTK file, or
        ``None`` if unavailable."""
        e = self._entry(label_or_index)
        if e is None:
            return None
        return e.get('recovery_vtk')

    def get_recovery(self, label_or_index):
        """Load the per-case recovery HDF5 file as a numpy dict.

        Returns ``None`` when the file is absent.
        """
        path = self.get_recovery_file(label_or_index)
        if path is None or not op.exists(path):
            return None
        from sfepy.base.ioutils import read_dict_hdf5
        return read_dict_hdf5(path)

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------
    def group_by(self, *keys):
        """Return ``{tuple_of_values: [label, ...]}`` grouping cases by the
        values of the requested context/params keys."""
        groups = {}
        for label, entry in self._by_label.items():
            ctx = entry.get('context') or {}
            params = ctx.get('params', entry.get('params', {}))
            key = tuple(params.get(k, ctx.get(k)) for k in keys)
            groups.setdefault(key, []).append(label)
        if len(keys) == 1:
            return {k[0]: v for k, v in groups.items()}
        return groups


# ---------------------------------------------------------------------------
# ParametricSweep (kept for backwards compatibility)
# ---------------------------------------------------------------------------
class CaseEntry(Struct):
    """Single sweep case record."""
    pass


class ParametricSweep(object):
    """Drive multiple homogenization runs from a single problem description.

    This class is kept for backwards compatibility.  New code should prefer
    the app-native approach via ``options`` dict (see module docstring).

    Parameters
    ----------
    materials : list, optional
        Material definitions to iterate over.
    frequencies : list of float, optional
        Frequency values (Hz).  ``problem_conf.omega`` is set to ``2*pi*f``.
    regions : list, optional
        Region-dict overrides.
    combinations : callable or list, optional
        If callable, used instead of the Cartesian product.
    apply_hook : callable(problem_conf, params) or None
        Extra hook applied after the built-in parameter assignment.
    base_conf : str or ProblemConf
        Path or already-loaded conf.
    output_dir : str
        Root output directory.
    coefs_filename : str
        Stem for per-case coefficient files.
    extra_options : dict, optional
        Extra options merged on top for every case.
    app_class : type, optional
        Application class (default ``HomogenizationApp``).
    """

    def __init__(self, materials=None, frequencies=None, regions=None,
                 combinations=None, apply_hook=None,
                 base_conf=None, output_dir='output',
                 coefs_filename='coefs',
                 extra_options=None, app_class=None):
        self.materials = list(materials) if materials else [None]
        self.frequencies = list(frequencies) if frequencies else [None]
        self.regions = list(regions) if regions else [None]
        self.combinations = combinations
        self.apply_hook = apply_hook
        self.base_conf = base_conf
        self.output_dir = output_dir
        self.coefs_filename = coefs_filename
        self.extra_options = dict(extra_options) if extra_options else {}
        if app_class is None:
            from sfepy.homogenization.homogen_app import HomogenizationApp
            self.app_class = HomogenizationApp
        else:
            self.app_class = app_class

    def iter_cases(self):
        if callable(self.combinations):
            for raw in self.combinations(self.materials,
                                         self.frequencies,
                                         self.regions):
                params = dict(raw)
                yield params, _label_from_params(params)
            return

        for mi, m in enumerate(self.materials):
            for fi, f in enumerate(self.frequencies):
                for ri, r in enumerate(self.regions):
                    params = {'materials': m, 'frequency': f, 'regions': r}
                    if len(self.materials) > 1:
                        params['material_index'] = mi
                    if len(self.frequencies) > 1:
                        params['frequency'] = f
                    if len(self.regions) > 1:
                        params['region_index'] = ri
                    yield params, _label_from_params(params)

    def _load_conf(self):
        from sfepy.base.conf import ProblemConf, get_standard_keywords
        if isinstance(self.base_conf, str):
            required, other = get_standard_keywords()
            return ProblemConf.from_file(self.base_conf, required, other)
        return self.base_conf.copy() if hasattr(self.base_conf, 'copy') \
            else self.base_conf

    def _apply_params(self, problem_conf, params):
        m = params.get('materials')
        if callable(m):
            m(problem_conf, params)
        else:
            _apply_materials(problem_conf, m)
        _apply_frequency(problem_conf, params.get('frequency'))
        _apply_regions(problem_conf, params.get('regions'))
        if self.apply_hook is not None:
            self.apply_hook(problem_conf, params)

    def _override_options(self, problem_conf, label):
        opts = getattr(problem_conf, 'options', None)
        if opts is None:
            opts = {}
            problem_conf.options = opts
        case_dir = op.join(self.output_dir, label)

        def _ensure(name, value):
            try:
                cur = opts.get(name, None)
            except AttributeError:
                cur = getattr(opts, name, None)
            if cur is None:
                try:
                    opts[name] = value
                except TypeError:
                    setattr(opts, name, value)

        _ensure('output_dir', case_dir)
        _ensure('coefs_filename',
                '%s_%s' % (self.coefs_filename, label))
        for k, v in self.extra_options.items():
            _ensure(k, v)

    def build_conf(self, params, label):
        conf = self._load_conf()
        self._override_options(conf, label)
        self._apply_params(conf, params)
        return conf

    def run(self, runner=None, verbose=False):
        """Execute the full sweep and return a :class:`CaseResults`.

        When ``runner`` is ``None``, the method delegates to
        :func:`run_from_conf` by injecting the sweep parameters into the
        problem conf options and letting ``HomogenizationApp`` drive the
        sweep natively.  A :class:`CaseResults` wrapper is then built from
        the merged ``Coefficients``.
        """
        if runner is not None:
            return self._run_legacy(runner=runner, verbose=verbose)

        # Build a conf with the sweep families injected into options so
        # that HomogenizationApp._run_sweep picks them up.
        conf = self._load_conf()
        coptions = getattr(conf, 'options', None)
        if coptions is None:
            coptions = {}
            conf.options = coptions

        def _ensure(name, value):
            try:
                if coptions.get(name, None) is None:
                    coptions[name] = value
            except AttributeError:
                if getattr(coptions, name, None) is None:
                    setattr(coptions, name, value)

        if self.materials != [None]:
            _ensure('sweep_materials', self.materials)
        if self.frequencies != [None]:
            _ensure('sweep_frequencies', self.frequencies)
        if self.regions != [None]:
            _ensure('sweep_regions', self.regions)
        if callable(self.combinations):
            _ensure('sweep_combinations', self.combinations)
        if self.apply_hook is not None:
            _ensure('sweep_hook', self.apply_hook)
        _ensure('output_dir', self.output_dir)
        _ensure('coefs_filename', self.coefs_filename)
        for k, v in self.extra_options.items():
            _ensure(k, v)

        merged = run_from_conf(conf, verbose=verbose)

        # Build CaseResults wrapper for backwards compatibility.
        sweep_cases = getattr(merged, '_sweep_cases', None)
        if sweep_cases is None:
            sweep_cases = []

        cases = []
        coef_names = []
        seen = set()
        for sc in sweep_cases:
            entry = CaseEntry(label=sc.label,
                              flat_index=sc.flat_index,
                              params=sc.params,
                              coefs=sc.coefs,
                              output_name=sc.output_name)
            for name in getattr(sc.coefs, '__dict__', {}).keys():
                if name not in seen:
                    seen.add(name)
                    coef_names.append(name)
            cases.append(entry)
        return CaseResults(cases, coef_names)

    def _run_legacy(self, runner=None, verbose=False):
        """Original manual sweep loop (used only when a custom ``runner`` is
        supplied)."""
        cases = []
        coef_names = []
        seen = set()
        for flat_index, (params, label) in enumerate(self.iter_cases()):
            if verbose:
                print('[sweep %d] %s' % (flat_index, label))
            try:
                os.makedirs(op.join(self.output_dir, label))
            except OSError:
                pass
            conf = self.build_conf(params, label)
            coefs = runner(conf)
            for name in getattr(coefs, '__dict__', {}).keys():
                if name not in seen:
                    seen.add(name)
                    coef_names.append(name)
            entry = CaseEntry(label=label,
                              flat_index=flat_index,
                              params=params,
                              coefs=coefs,
                              output_name='%s_%s' % (self.coefs_filename,
                                                     label))
            cases.append(entry)
        return CaseResults(cases, coef_names)


def run_homog_app(conf, app_class=None):
    """Default driver: instantiate ``HomogenizationApp`` from ``conf`` and
    return its ``.call()`` result."""
    from sfepy.base.conf import ProblemConf
    from sfepy.homogenization.homogen_app import HomogenizationApp

    if app_class is None:
        app_class = HomogenizationApp
    if isinstance(conf, str):
        from sfepy.base.conf import get_standard_keywords
        required, other = get_standard_keywords()
        conf = ProblemConf.from_file(conf, required, other)
    app = app_class(conf, None, None)
    return app(verbose=False)
