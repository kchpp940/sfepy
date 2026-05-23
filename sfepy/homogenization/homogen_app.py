import os
import os.path as op
import shutil
import re

import numpy as nm

from sfepy.base.base import get_default, Struct, ordered_iteritems, output
from sfepy.base.ioutils import read_dict_hdf5, write_dict_hdf5
from sfepy.homogenization.coefficients import Coefficients
from sfepy.homogenization.engine import HomogenizationEngine
from sfepy.applications import PDESolverApp
import sfepy.discrete.fem.periodic as per
import sfepy.linalg as la
import sfepy.base.multiproc as multi


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


class SweepCase(Struct):
    """A single parametric-sweep case record."""
    pass


class HomogenizationApp(HomogenizationEngine):
    @staticmethod
    def process_options(options):
        """
        Application options setup. Sets default values for missing
        non-compulsory options.
        """
        get = options.get

        volume = get('volume', None)
        volumes = get('volumes', None)
        if volume is None and volumes is None:
            raise ValueError('missing "volume" in options!')

        return Struct(print_digits=get('print_digits', 3),
                      float_format=get('float_format', '%8.3e'),
                      coefs_filename=get('coefs_filename', 'coefs'),
                      tex_names=get('tex_names', None),
                      coefs=get('coefs', None, 'missing "coefs" in options!'),
                      requirements=get('requirements', None,
                                       'missing "requirements" in options!'),
                      return_all=get('return_all', False),
                      mesh_update_variable=get('mesh_update_variable', None),
                      macro_data=get('macro_data', None),
                      micro_update=get('micro_update', {}),
                      n_micro=get('n_micro', None),
                      multiprocessing=get('multiprocessing', True),
                      use_mpi=get('use_mpi', False),
                      store_micro_idxs=get('store_micro_idxs', []),
                      sweep_hook=get('sweep_hook', None),
                      sweep_label=get('sweep_label', None),
                      sweep_materials=get('sweep_materials', None),
                      sweep_frequencies=get('sweep_frequencies', None),
                      sweep_regions=get('sweep_regions', None),
                      sweep_combinations=get('sweep_combinations', None),
                      sweep_save_merged=get('sweep_save_merged', True),
                      sweep_group_by=get('sweep_group_by', None),
                      volume=volume,
                      volumes=volumes)

    def __init__(self, conf, options, output_prefix, **kwargs):
        PDESolverApp.__init__(self, conf, options, output_prefix,
                              init_equations=False)

        self.setup_options()
        self.n_micro = kwargs.get('n_micro',
                                  self.app_options.get('n_micro', None))
        self.updating_corrs = None
        self.micro_state_cache = {}
        self.multiproc_mode = None
        self.micro_states = None if self.n_micro is None else {}

        macro_data = self.app_options.macro_data
        if macro_data is not None:
            self.n_micro = macro_data[list(macro_data.keys())[0]].shape[0]
            self.setup_macro_data(macro_data)

        if self.n_micro is not None:
            for k in self.app_options.micro_update:
               if not k == 'coors':
                   self.micro_states[k] = None

            coors = self.problem.domain.get_mesh_coors()
            c_sh = (self.n_micro,) + coors.shape
            self.micro_states['coors'] = nm.empty(c_sh, dtype=nm.float64)

            mac_ids = kwargs.get('mac_ids',
                                  self.app_options.get('mac_ids', None))
            self.micro_states['id'] = []
            for im in range(self.n_micro):
                self.micro_states['coors'][im] = coors
                self.micro_states['id'].append(
                    mac_ids[im] if mac_ids is not None else im)

        output_dir = self.problem.output_dir

        if conf._filename is not None:
            shutil.copyfile(conf._filename,
                            op.join(output_dir, op.basename(conf._filename)))

    def setup_options(self):
        PDESolverApp.setup_options(self)
        po = HomogenizationApp.process_options
        self.app_options += po(self.conf.options)
        if hasattr(self, 'he'):
            self.he.setup_options()

    def setup_macro_data(self, data):
        self.macro_data = data
        self.problem.homogenization_macro_data = self.macro_data

    def get_micro_cache_key(self, key, icoor, itime):
        tt = '' if itime is None else '_t%03d' % itime
        return '%s_%d%s' % (key, icoor, tt)

    def update_micro_states(self):
        def calculate_local_update(state, corrs, var, macro_vals, mul=1):
            for ic, corr in enumerate(corrs):
                if state is None:
                    sh = corr.states[corr.components[0]][var].shape \
                        if hasattr(corr, 'states') else corr.state[var].shape
                    state = nm.zeros((len(corrs),) + sh, dtype=nm.float64)
                else:
                    sh = state[ic].shape

                if hasattr(corr, 'states'):
                    corr_arr = nm.array(
                        [corr.states[jj][var] for jj in corr.components]).T
                    mval = macro_vals[ic].reshape((corr_arr.shape[1], 1))
                    state[ic] += mul * nm.dot(corr_arr, mval).reshape(sh)
                else:
                    if macro_vals is None:
                        state[ic] += mul * corr.state[var].reshape(sh)
                    else:
                        state[ic] += mul\
                            * (corr.state[var] * macro_vals[ic]).reshape(sh)

            return state

        micro_update = self.app_options.micro_update
        for key, upd_obj in micro_update.items():
            if '_prev' in key:
                continue

            state = self.micro_states[key]

            if key + '_prev' in micro_update and state is not None:
                self.micro_states[key + '_prev'] = state.copy()

            if key == 'coors':
                if hasattr(upd_obj, '__call__'):
                    upd_obj(state, self.macro_data, self.problem)
                else:
                    mtx_e = self.macro_data[upd_obj[0][2]]
                    state += la.dot_sequences(state, mtx_e, 'ABT')

            if hasattr(upd_obj, '__call__'):
                upd_obj(state, self.macro_data, self.problem)
            else:
                if self.updating_corrs is not None:
                    for v in upd_obj:
                        if len(v) == 4:
                            cname, vname, mname, mul = v
                        else:
                            cname, vname, mname = v
                            mul = 1

                        macro_data = None if mname is None \
                            else self.macro_data[mname]
                        if cname is not None:
                            state0 = calculate_local_update(
                                state, self.updating_corrs[cname],
                                vname, macro_data, mul)
                        else:
                            state += macro_data[..., 0]

                        if state0 is not state:
                            self.micro_states[key] = state0
                            state = state0

    # ------------------------------------------------------------------
    # Sweep helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_sweep(opts):
        return any(getattr(opts, k, None) is not None for k in
                   ('sweep_materials', 'sweep_frequencies', 'sweep_regions',
                    'sweep_combinations'))

    @staticmethod
    def _iter_sweep_cases(opts):
        """Yield (params, label) for every sweep combination.

        params carries ``materials``, ``frequency``, ``regions`` plus the
        convenience ``material_index`` / ``region_index`` when the
        corresponding family has more than one element.
        """
        mats = list(opts.sweep_materials) if opts.sweep_materials else [None]
        freqs = list(opts.sweep_frequencies) if opts.sweep_frequencies else [None]
        regs = list(opts.sweep_regions) if opts.sweep_regions else [None]

        if callable(opts.sweep_combinations):
            for raw in opts.sweep_combinations(mats, freqs, regs):
                params = dict(raw)
                yield params, _label_from_params(params)
            return

        for mi, m in enumerate(mats):
            for fi, f in enumerate(freqs):
                for ri, r in enumerate(regs):
                    params = {'materials': m, 'frequency': f, 'regions': r}
                    if len(mats) > 1:
                        params['material_index'] = mi
                    if len(freqs) > 1:
                        params['frequency'] = f
                    if len(regs) > 1:
                        params['region_index'] = ri
                    yield params, _label_from_params(params)

    @staticmethod
    def _apply_params(problem_conf, params, sweep_hook=None):
        """Apply a single case's parameters to a problem conf (in-place).

        Materials and regions are merged at the leaf level -- only the keys
        explicitly listed in ``params['materials']`` / ``params['regions']``
        are overridden; everything else is kept from the original conf.
        """
        # --- materials (deep merge, not wholesale replacement) ---
        mats = params.get('materials')
        if callable(mats):
            mats(problem_conf, params)
        elif mats is not None:
            cur = getattr(problem_conf, 'materials', None)
            if cur is None:
                problem_conf.materials = mats
            elif isinstance(cur, dict) and isinstance(mats, dict):
                for mk, mval in mats.items():
                    if (mk in cur
                            and isinstance(cur[mk], (tuple, list))
                            and isinstance(mval, (tuple, list))):
                        cur[mk] = type(cur[mk])(mval)
                    elif isinstance(cur[mk], dict) and isinstance(mval, dict):
                        cur[mk].update(mval)
                    else:
                        cur[mk] = mval
            else:
                problem_conf.materials = mats

        # --- frequency ---
        freq = params.get('frequency')
        if freq is not None:
            problem_conf.omega = 2.0 * nm.pi * freq

        # --- regions (deep merge) ---
        regs = params.get('regions')
        if isinstance(regs, dict):
            cur = getattr(problem_conf, 'regions', None)
            if cur is None:
                problem_conf.regions = regs
            elif isinstance(cur, dict):
                for rk, rval in regs.items():
                    if isinstance(cur.get(rk), dict) and isinstance(rval, dict):
                        cur[rk].update(rval)
                    else:
                        cur[rk] = rval
        elif regs is not None:
            problem_conf.regions = regs

        if sweep_hook is not None:
            sweep_hook(problem_conf, params)

    def _run_sweep(self, verbose=False, ret_all=False):
        """Execute a full parametric sweep declared in options.

        Returns
        -------
        coefs : Coefficients
            A merged ``Coefficients`` whose attribute arrays are indexed by
            flat case index.  A ``labels`` attribute carries the per-case
            label so that ``coefs.select_case(label)`` works.
        deps : list or None
            ``None`` when ``ret_all`` is False; otherwise a list of
            dependency dicts (one per case).
        """
        from sfepy.base.conf import ProblemConf, get_standard_keywords

        opts = self.app_options
        sweep_hook = opts.sweep_hook

        # Pre-compute the list of cases so the number is known.
        cases = list(self._iter_sweep_cases(opts))

        all_coefs = []
        all_deps = [] if ret_all else None
        labels = []
        indices = []
        case_records = []

        for flat_index, (params, label) in enumerate(cases):
            if verbose:
                output('[sweep %d/%d] %s'
                       % (flat_index + 1, len(cases), label))

            case_dir = op.join(self.problem.output_dir, label)
            try:
                os.makedirs(case_dir)
            except OSError:
                pass

            # Build a fresh ProblemConf for this case, based on the
            # original filename if available.
            if self.conf._filename is not None:
                required, other = get_standard_keywords()
                case_conf = ProblemConf.from_file(self.conf._filename,
                                                  required, other)
            else:
                case_conf = self.conf

            # Strip all sweep_* options from the sub-case conf to prevent
            # recursive sweep detection in the child HomogenizationApp.
            coptions = getattr(case_conf, 'options', None)
            if coptions is None:
                coptions = {}
                case_conf.options = coptions

            _SWEEP_KEYS = (
                'sweep_materials', 'sweep_frequencies', 'sweep_regions',
                'sweep_combinations', 'sweep_save_merged',
                'sweep_group_by', 'sweep_hook', 'sweep_label',
            )
            for sk in _SWEEP_KEYS:
                try:
                    coptions.pop(sk, None)
                except AttributeError:
                    if hasattr(coptions, sk):
                        delattr(coptions, sk)

            def _ensure(name, value):
                try:
                    if coptions.get(name, None) is None:
                        coptions[name] = value
                except AttributeError:
                    if getattr(coptions, name, None) is None:
                        setattr(coptions, name, value)

            _ensure('output_dir', case_dir)
            _ensure('coefs_filename', '%s_%s' % (opts.coefs_filename, label))

            # Apply sweep parameters.
            self._apply_params(case_conf, params, sweep_hook=sweep_hook)

            # Instantiate a sub-app and run it.
            sub_app = HomogenizationApp(case_conf, None, None)
            aux = sub_app(verbose=False, ret_all=ret_all)
            if ret_all:
                case_coefs, case_deps = aux
                all_deps.append(case_deps)
            else:
                case_coefs = aux

            all_coefs.append(case_coefs)
            labels.append(label)
            indices.append(flat_index)

            # Build the output file paths for this case (for the index).
            coefs_h5 = op.join(case_dir,
                               '%s_%s.h5' % (opts.coefs_filename, label))
            coefs_txt = op.join(case_dir,
                                '%s_%s.txt' % (opts.coefs_filename, label))
            recovery_h5 = op.join(case_dir, 'recovery.h5')
            recovery_vtk = op.join(case_dir, 'recovery.vtk')

            # Build a serialisable ``context`` dict that records everything
            # needed to identify / reproduce this case without re-running
            # the sweep.
            context = {
                'problem_file': op.abspath(self.conf._filename)
                if self.conf._filename is not None else None,
                'output_dir': op.abspath(case_dir),
                'coefs_file': op.abspath(coefs_h5),
                'coefs_txt': op.abspath(coefs_txt),
                'recovery_file': op.abspath(recovery_h5)
                if op.exists(recovery_h5) else None,
                'recovery_vtk': op.abspath(recovery_vtk)
                if op.exists(recovery_vtk) else None,
                'frequency': params.get('frequency'),
                'material_index': params.get('material_index'),
                'region_index': params.get('region_index'),
                'params': params,
            }

            case_records.append(SweepCase(
                label=label,
                flat_index=flat_index,
                params=params,
                coefs=case_coefs,
                output_name='%s_%s' % (opts.coefs_filename, label),
                coefs_file=coefs_h5,
                coefs_txt=coefs_txt,
                recovery_file=context['recovery_file'],
                recovery_vtk=context['recovery_vtk'],
                context=context,
            ))

        # --- Merge ---
        merged = Coefficients.merge(all_coefs, labels=labels)
        merged._sweep_cases = case_records
        merged._sweep_labels = labels
        merged._sweep_indices = indices

        # --- Grouping ---
        group_by = opts.sweep_group_by
        if group_by is not None and isinstance(group_by, (list, tuple)):
            merged._sweep_groups = self._group_cases(case_records, group_by)

        # --- Save merged archive ---
        if opts.sweep_save_merged:
            self._save_sweep_archive(merged, case_records)

        if ret_all:
            return merged, all_deps
        else:
            return merged

    @staticmethod
    def _group_cases(case_records, keys):
        """Return ``{tuple_of_key_values: [label, ...]}`` grouping."""
        groups = {}
        for c in case_records:
            key = tuple(c.params.get(k) for k in keys)
            groups.setdefault(key, []).append(c.label)
        return groups

    def _save_sweep_archive(self, merged, case_records):
        """Save a single HDF5 archive + JSON index for the whole sweep.

        The JSON index records an explicit ``label -> {coefs_file,
        recovery_file, recovery_vtk, params, context}`` mapping so that a
        downstream tool can restore individual cases without re-running the
        sweep.
        """
        opts = self.app_options
        base = op.join(self.problem.output_dir, opts.coefs_filename)
        h5file = base + '_sweep.h5'
        idxfile = base + '_sweep.index.json'

        payload = {
            'labels': nm.array(merged.labels,
                               dtype='S%d' % max(1, max(len(l)
                                                        for l in merged.labels))),
            'flat_index': nm.arange(len(merged.labels), dtype=nm.int32),
        }
        for c in case_records:
            cgrp = {}
            for name, val in c.coefs.__dict__.items():
                if isinstance(val, nm.ndarray):
                    cgrp[name] = val
                elif isinstance(val, (float, int)):
                    cgrp[name] = nm.asarray(val)
            if cgrp:
                payload['cases/%s' % c.label] = cgrp

        write_dict_hdf5(h5file, payload)

        import json

        index = {
            'meta': {
                'sweep_h5': op.abspath(h5file),
                'output_dir': op.abspath(self.problem.output_dir),
                'coefs_filename': opts.coefs_filename,
                'num_cases': len(case_records),
            },
            'cases': [
                {
                    'label': c.label,
                    'flat_index': c.flat_index,
                    'params': c.params,
                    'coefs_file': op.abspath(c.coefs_file)
                    if getattr(c, 'coefs_file', None) else None,
                    'coefs_txt': op.abspath(c.coefs_txt)
                    if getattr(c, 'coefs_txt', None) else None,
                    'recovery_file': op.abspath(c.recovery_file)
                    if getattr(c, 'recovery_file', None) else None,
                    'recovery_vtk': op.abspath(c.recovery_vtk)
                    if getattr(c, 'recovery_vtk', None) else None,
                    'output_name': c.output_name,
                    'context': getattr(c, 'context', None),
                }
                for c in case_records
            ],
        }
        with open(idxfile, 'w') as fd:
            json.dump(index, fd, indent=2,
                      default=lambda o: (o.tolist()
                                         if isinstance(o, nm.ndarray)
                                         else float(o)
                                         if isinstance(o, (nm.floating,))
                                         else int(o)
                                         if isinstance(o, (nm.integer,))
                                         else None))

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------
    def call(self, verbose=False, ret_all=None, itime=None, iiter=None):
        """
        Call the homogenization engine and compute the homogenized
        coefficients.

        If any of ``sweep_materials``/``sweep_frequencies``/``sweep_regions``
        /``sweep_combinations`` is set in options, a full parametric sweep
        is executed and a merged ``Coefficients`` object is returned.

        Parameters
        ----------
        verbose : bool
            If True, print the computed coefficients.
        ret_all : bool or None
            If not None, it can be used to override the 'return_all' option.
            If True, also the dependencies are returned.
        time_tag: str
            The time tag used in file names.

        Returns
        -------
        coefs : Coefficients instance
            The homogenized coefficients.
        dependencies : dict
            The dependencies, if `ret_all` is True.
        """
        opts = self.app_options

        ret_all = get_default(ret_all, opts.return_all)

        # --- Sweep mode ---
        if self._is_sweep(opts):
            return self._run_sweep(verbose=verbose, ret_all=ret_all)

        sweep_hook = getattr(opts, 'sweep_hook', None)
        sweep_label = getattr(opts, 'sweep_label', None)
        if sweep_hook is not None:
            sweep_hook(self.problem, sweep_label)

        force_init_he = hasattr(self.problem, 'force_init_he')\
            and self.problem.force_init_he
        if not hasattr(self, 'he') or force_init_he:
            volumes = {}
            if hasattr(opts, 'volumes') and (opts.volumes is not None):
                volumes.update(opts.volumes)
            elif hasattr(opts, 'volume') and (opts.volume is not None):
                volumes['total'] = opts.volume
            else:
                volumes['total'] = 1.0

            self.he = HomogenizationEngine(self.problem, self.options,
                                           volumes=volumes)

        if self.micro_states is not None:
            self.update_micro_states()
            self.he.set_micro_states(self.micro_states)

        multiproc_mode = None
        if opts.multiprocessing and multi.use_multiprocessing:
            multiproc, multiproc_mode = multi.get_multiproc(mpi=opts.use_mpi)

            if multiproc_mode is not None:
                upd_var = self.app_options.mesh_update_variable
                if upd_var is not None:
                    uvar = self.problem.create_variables([upd_var])[upd_var]
                    uvar.field.mappings0 = multiproc.get_dict('mappings0',
                                                              soft_set=True)
                per.periodic_cache = multiproc.get_dict('periodic_cache',
                                                        soft_set=True)

        time_tag = ('' if itime is None else '_t%03d' % itime)\
            + ('' if iiter is None else '_i%03d' % iiter)

        aux = self.he(ret_all=ret_all, time_tag=time_tag)
        if ret_all:
            coefs, dependencies = aux
            self.updating_corrs = {}
            for upd_obj in opts.micro_update.values():
                if upd_obj is not None and not hasattr(upd_obj, '__call__'):
                    for v in upd_obj:
                        cr = v[0]
                        if cr is not None:
                            self.updating_corrs[cr] = dependencies[cr]
        else:
            coefs = aux

        if coefs is not None:
            coefs = Coefficients(**coefs.to_dict())

            if verbose:
                prec = nm.get_printoptions()['precision']
                if hasattr(opts, 'print_digits'):
                    nm.set_printoptions(precision=opts.print_digits)
                print(coefs)
                nm.set_printoptions(precision=prec)

            ms_cache = self.micro_state_cache
            for ii in self.app_options.store_micro_idxs:
                for k in self.micro_states.keys():
                    key = self.get_micro_cache_key(k, ii, itime)
                    ms_cache[key] = self.micro_states[k][ii]

            coef_save_name = op.join(opts.output_dir, opts.coefs_filename)
            coefs.to_file_hdf5(coef_save_name + '%s.h5' % time_tag)
            coefs.to_file_txt(coef_save_name + '%s.txt' % time_tag,
                              opts.tex_names,
                              opts.float_format)

        if ret_all:
            return coefs, dependencies
        else:
            return coefs
