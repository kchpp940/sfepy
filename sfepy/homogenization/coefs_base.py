import os
import hashlib

import numpy as nm
from collections.abc import Iterable

from sfepy.base.base import assert_, get_default, Struct
from sfepy.discrete.evaluate import eval_equations
from .utils import iter_sym, iter_nonsym, create_pis, create_scalar_pis,\
    rm_multi


def _stable_hash(obj):
    """Hash an arbitrary object into a short hex string (deterministic)."""
    raw = repr(obj) if not isinstance(obj, (bytes, bytearray)) else bytes(obj)
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    return hashlib.sha256(raw).hexdigest()[:16]


class CoefficientContext(Struct):
    """
    Unified context that uniquely identifies a homogenization coefficient
    computation across engine, coefficients, coefs_base, and recovery.

    Captures material parameters, frequency points, region dependencies,
    output file names, time tag, and micro-state identifier.  When any of
    these change, `get_cache_key()` returns a different string, preventing
    stale cache hits when a problem configuration is reused with different
    parameters.

    The context is meant to be built in two phases:

    1. Construction-time ("configuration" summary) from the problem's
       ``conf``, output file name, extra user-provided ids, etc.
    2. Runtime enrichment via :meth:`update_from_problem` which records
       hashes of the *actual* material values that enter the cell
       problem, the current time/frequency state, and the region
       selection expressions.

    Only the :meth:`get_cache_key` result should ever be used as a dict
    key or file-level discriminant.

    Parameters
    ----------
    problem : Problem or None
        The microscopic problem.  Static material values and region
        names are extracted from it.
    time_tag : str
        Label for the current time step / iteration, used in file names.
    micro_id : int or None
        Identifier of the current microstructure when multiple RVEs are
        processed.
    output_file_base : str or None
        Base name for output files.
    extra_ids : dict or None
        Additional key-value pairs to include in the cache key.
    """

    def __init__(self, problem=None, time_tag='', micro_id=None,
                 output_file_base=None, extra_ids=None, **kwargs):
        Struct.__init__(self, time_tag=time_tag, micro_id=micro_id,
                        output_file_base=output_file_base, **kwargs)

        self.problem_name = 'unknown'
        self.conf_material_hash = 'nomat'
        self.runtime_material_hash = 'nomat'
        self.region_hash = 'noreg'
        self.region_expressions_hash = 'noreg'
        self.frequency = None
        self.ts_step = None
        self.ts_time = None

        if problem is not None:
            self.problem_name = getattr(problem.conf, '_filename',
                                        'unknown') or 'unknown'
            self.conf_material_hash = self._hash_conf_materials(problem)
            self.region_hash = self._hash_region_names(problem)
            self.region_expressions_hash = \
                self._hash_region_expressions(problem)
            self.frequency = getattr(problem, 'frequency', None)
            ts = getattr(problem, 'ts', None)
            if ts is not None:
                self.ts_step = getattr(ts, 'step', None)
                self.ts_time = getattr(ts, 'time', None)

        self.extra_ids = extra_ids or {}

    # ------------------------------------------------------------------
    # Configuration-level hashing (static problem definition)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_conf_materials(problem):
        """Hash the material *definitions* from ``problem.conf``.

        Supports both the long syntax (``dict`` with ``'values'``,
        ``'function'``, ...) and the transformed syntax (``Struct``
        instances produced by :func:`transform_materials`, which have
        ``.values``, ``.function``, ``.kind``, ``.flags``).
        """
        if hasattr(problem.conf, 'materials') and problem.conf.materials:
            parts = []
            for k in sorted(problem.conf.materials.keys()):
                mat = problem.conf.materials[k]
                if isinstance(mat, dict):
                    for mk in sorted(mat.keys()):
                        v = mat[mk]
                        if mk == 'values':
                            if hasattr(v, 'to_list'):
                                v = v.to_list()
                            parts.append(f'{k}.{mk}={v}')
                        elif mk == 'function':
                            parts.append(f'{k}.{mk}=func')
                        else:
                            parts.append(f'{k}.{mk}={v}')
                else:
                    # Struct produced by transform_materials.
                    s_parts = [f'name={getattr(mat, "name", k)}']
                    vals = getattr(mat, 'values', None)
                    if vals is not None:
                        if isinstance(vals, dict):
                            for vk in sorted(vals.keys()):
                                v = vals[vk]
                                if isinstance(v, nm.ndarray):
                                    v = _stable_hash(
                                        (v.shape, v.dtype.name,
                                         nm.ascontiguousarray(v).tobytes()))
                                s_parts.append(f'v.{vk}={v}')
                        else:
                            if isinstance(vals, nm.ndarray):
                                vals = _stable_hash(
                                    (vals.shape, vals.dtype.name,
                                     nm.ascontiguousarray(vals).tobytes()))
                            s_parts.append(f'values={vals}')
                    if getattr(mat, 'function', None) is not None:
                        s_parts.append('func=1')
                    kind = getattr(mat, 'kind', None)
                    if kind is not None:
                        s_parts.append(f'kind={kind}')
                    flags = getattr(mat, 'flags', None)
                    if flags:
                        s_parts.append(f'flags={flags}')
                    parts.append('|'.join(s_parts))
            return _stable_hash('|'.join(parts))
        return 'nomat'

    @staticmethod
    def _hash_region_names(problem):
        try:
            regions = problem.domain.regions
            names = sorted(regions.keys())
            return _stable_hash(','.join(names))
        except Exception:
            return 'noreg'

    @staticmethod
    def _hash_region_expressions(problem):
        """Hash the region *selector expressions* (not just names)."""
        try:
            regions = problem.domain.regions
            parts = []
            for rn in sorted(regions.keys()):
                reg = regions[rn]
                parts.append(f'{rn}:{getattr(reg, "definition", "")}'
                             f'#{getattr(reg, "parse_def", "")}')
            return _stable_hash('|'.join(parts))
        except Exception:
            return 'noreg'

    # ------------------------------------------------------------------
    # Runtime enrichment (actual material values in the cell problem)
    # ------------------------------------------------------------------

    def update_from_problem(self, problem):
        """
        Pull runtime values out of the live problem so that the cache
        key changes whenever the actual materials/time/frequency/regions
        differ, even if ``conf.materials`` and region names are reused.

        This is called inside :meth:`HomogenizationWorker.calculate`
        right before the cell problem is solved, so ``equations``
        (and therefore ``problem.get_materials()``) have already been
        set up.

        Parameters
        ----------
        problem : Problem or None
            The live microscopic problem.  If ``None``, nothing is
            updated.
        """
        if problem is None:
            return

        self.problem_name = getattr(problem.conf, '_filename',
                                    'unknown') or 'unknown'

        # Runtime materials - the *actual* values evaluated at the
        # current quadrature points / time.
        mats = None
        try:
            mats = problem.get_materials()
        except Exception:
            mats = None

        if mats is not None and len(mats):
            parts = []
            for name in sorted(mats.names):
                mat = mats[name]
                mat_parts = [f'name={name}']
                # datas is a dict of {key: value_arrays_or_dicts}
                datas = getattr(mat, 'datas', {}) or {}
                for dk in sorted(datas.keys()):
                    dv = datas[dk]
                    try:
                        if isinstance(dv, nm.ndarray):
                            dv_hash = _stable_hash(
                                (dv.shape, dv.dtype.name,
                                 nm.ascontiguousarray(dv).tobytes()))
                        else:
                            dv_hash = _stable_hash(dv)
                    except Exception:
                        dv_hash = 'err'
                    mat_parts.append(f'{dk}={dv_hash}')
                extra_args = getattr(mat, 'extra_args', {}) or {}
                for ek in sorted(extra_args.keys()):
                    ev = extra_args[ek]
                    if isinstance(ev, nm.ndarray):
                        ev = _stable_hash(
                            (ev.shape, ev.dtype.name,
                             nm.ascontiguousarray(ev).tobytes()))
                    mat_parts.append(f'ea.{ek}={ev}')
                parts.append('|'.join(mat_parts))
            self.runtime_material_hash = _stable_hash('||'.join(parts))
        else:
            # Fall back to the configuration-level summary.
            self.runtime_material_hash = self.conf_material_hash

        # Time / frequency (frequencies come from the time stepper or
        # directly from the problem).
        self.frequency = getattr(problem, 'frequency', self.frequency)
        ts = getattr(problem, 'ts', None)
        if ts is not None:
            self.ts_step = getattr(ts, 'step', self.ts_step)
            self.ts_time = getattr(ts, 'time', self.ts_time)

        # Region names and expressions - re-hash in case regions were
        # modified at runtime.
        self.region_hash = self._hash_region_names(problem)
        self.region_expressions_hash = \
            self._hash_region_expressions(problem)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def get_cache_key(self):
        """Return a deterministic, unique cache key string.

        The key incorporates both configuration-level material/region
        summaries and the runtime material values / time state whenever
        :meth:`update_from_problem` was called.
        """
        parts = [
            self.problem_name,
            self.conf_material_hash,
            self.runtime_material_hash,
            self.region_hash,
            self.region_expressions_hash,
            str(self.frequency),
            str(self.ts_step),
            str(self.ts_time),
            self.time_tag,
            str(self.micro_id) if self.micro_id is not None else 'na',
            str(self.output_file_base) if self.output_file_base else 'na',
        ]
        for k in sorted(self.extra_ids.keys()):
            parts.append(f'{k}={self.extra_ids[k]}')
        return '|'.join(str(p) for p in parts)

    def get_file_tag(self):
        """Return a file-system-safe tag for output names."""
        parts = []
        if self.time_tag:
            parts.append(self.time_tag)
        if self.micro_id is not None:
            parts.append(f'_mic{self.micro_id}')
        return ''.join(parts)


class MiniAppBase(Struct):
    def any_from_conf(name, problem, kwargs):
        try:
            cls = kwargs['class']
        except KeyError:
            raise KeyError("set 'class' for MiniApp %s!" % name)
        obj = cls(name, problem, kwargs)
        return obj
    any_from_conf = staticmethod(any_from_conf)

    def __init__(self, name, problem, kwargs):
        Struct.__init__(self, name=name, problem=problem, **kwargs)

        if self.problem is not None:
            self.problem.clear_equations()

        self.set_default('requires', [])
        self.set_default('is_linear', False)
        self.set_default('dtype', nm.float64)
        self.set_default('term_mode', None)
        self.set_default('set_volume', 'total')

        # Application-specific options.
        self.app_options = self.process_options()

    def process_options(self):
        """
        Setup application-specific options.

        Subclasses should implement this method as needed.

        Returns
        -------
        app_options : Struct instance
            The application options.
        """

    def init_solvers(self, problem):
        """
        Setup solvers. Use local options if these are defined,
        otherwise use the global ones.

        For linear problems, assemble the matrix and try to presolve the
        linear system.
        """

        if hasattr(self, 'solvers'):
            opts = self.solvers

        else:
            opts = problem.conf.options

        problem.set_conf_solvers(problem.conf.solvers, opts)
        problem.init_solvers()
        problem.set_linear(self.is_linear)

        if self.is_linear:
            set_presolve = True
            for v in problem.conf.solvers.values():
                if v.kind.startswith('ls.') and hasattr(v, 'use_presolve'):
                    set_presolve = False

            ls_conf = problem.get_solver().nls.lin_solver.conf
            if set_presolve and hasattr(ls_conf, 'use_presolve'):
                ls_conf.use_presolve = True

    def _get_volume(self, volume):
        if isinstance(volume, dict):
            return volume[self.set_volume]

        else:
            return volume

class CorrSolution(Struct):
    """
    Class for holding solutions of corrector problems.
    """

    def iter_solutions(self):
        if hasattr(self, 'components'):
            for indx in self.components:
                key = ('%d' * len(indx)) % indx
                yield key, self.states[indx]

        else:
            yield '', self.state

    def iter_time_steps(self):
        if hasattr(self, 'n_step') and self.n_step > 0:
            for ii in range(self.n_step):
                yield self.get_ts_val(ii)
        else:
            yield self

    def get_ts_val(self, step):
        if hasattr(self, 'states'):
            states = nm.zeros(self.states.shape, dtype=object)
            for idx in self.components:
                state = {k: v[step] for k, v in self.states[idx].items()}
                states[idx] = state

            out = CorrSolution(name=self.name,
                               states=states,
                               components=self.components)

        else:
            state = {k: v[step] for k, v in self.state.items()}
            out = CorrSolution(name=self.name,
                               state=state)

        return out

    def get_output(self, is_dump=False, var_map=None):

        out = {}
        for key, sol in self.iter_solutions():
            for var_name in sol.keys():
                if var_map is not None and var_name in var_map:
                    vname = var_map[var_name]
                else:
                    vname = var_name

                dof_vector = sol[var_name]
                if len(dof_vector.shape) == 1:
                    dof_vector = dof_vector[:, None]

                if is_dump:
                    skey = var_name + '_' + key if key else var_name
                    out[skey] = Struct(name='dump', mode='vertex',
                                       data=dof_vector,
                                       shape=dof_vector.shape,
                                       var_name=vname)
                else:
                    new_key = var_name + '_' + key if key else var_name
                    out[new_key] = dof_vector

        return out


class CorrMiniApp(MiniAppBase):

    def __init__(self, name, problem, kwargs):
        MiniAppBase.__init__(self, name, problem, kwargs)
        self.output_dir = self.problem.output_dir
        self.set_default('save_name', None)

        if self.save_name is not None:
            self.save_name = os.path.normpath(os.path.join(self.output_dir,
                                                           self.save_name))

    def setup_output(self, save_formats=None, post_process_hook=None,
                     split_results_by=None):
        """Instance attributes have precedence!"""
        self.set_default('save_formats', save_formats)
        self.set_default('post_process_hook', post_process_hook)
        self.set_default('split_results_by', split_results_by)

    def get_save_name_base(self):
        return self.save_name

    def get_save_name(self, save_format='.h5', stamp=''):
        save_name_base = self.get_save_name_base()
        if save_name_base is not None:
            return '.'.join((save_name_base + stamp, save_format))

    def get_output(self, corr_sol, is_dump=False, extend=True,
                   variables=None, var_map=None):
        if variables is None:
            variables = self.problem.get_variables()
        to_output = variables.create_output

        if is_dump:
            extend = False

        out = {}
        for key, sol in corr_sol.iter_solutions():
            for var_name in sol.keys():
                if var_name not in variables.ordered_state\
                    and var_map is not None\
                    and var_name in var_map:
                    vname = var_map[var_name]
                else:
                    vname = var_name

                dof_vector = sol[var_name]

                if is_dump:
                    skey = var_name + '_' + key if key else var_name
                    var = variables[vname]
                    shape = (var.n_dof // var.n_components,
                             var.n_components)
                    out[skey] = Struct(name='dump', mode='vertex',
                                       data=dof_vector,
                                       shape=shape,
                                       var_name=vname,
                                       region_name=var.field.region.name)

                else:
                    aux = to_output(dof_vector,
                                    var_info={vname: (True, var_name)},
                                    extend=extend)
                    if self.post_process_hook is not None:
                        aux = self.post_process_hook(aux, self.problem,
                                                     None,
                                                     extend=extend)

                    for _key, val in aux.items():
                        if key:
                            new_key = _key + '_' + key

                        else:
                            new_key = _key
                        out[new_key] = val

        return out

    def save(self, state, problem, variables=None, ts=None, var_map=None):
        if ts is not None:
            n_digit = int(nm.log10(ts.n_step)) + 1
            time_stamp = ('_%s' % ('%%0%dd' % n_digit)) % ts.step
        else:
            time_stamp = ''

        for save_format in self.save_formats:
            if self.get_save_name_base() is not None:
                if save_format in ['h5']:
                    save_name = self.get_save_name(save_format)
                    is_dump, split_results_by, extend = True, 'none', False
                else:
                    save_name = self.get_save_name(save_format, time_stamp)
                    split_results_by, is_dump = self.split_results_by, False
                    extend = split_results_by is None

                out = self.get_output(state, extend=extend, is_dump=is_dump,
                                      variables=variables, var_map=var_map)

                problem.save_state(save_name, out=out,
                                   split_results_by=split_results_by, ts=ts)

class ShapeDimDim(CorrMiniApp):

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        clist, pis = create_pis(problem, self.variables[0])

        corr_sol = CorrSolution(name=self.name,
                                states=pis,
                                components=clist)
        self.save(corr_sol, problem,
                  variables=problem.create_variables([self.variables[0]]))

        return corr_sol

class ShapeDim(CorrMiniApp):

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        clist, pis = create_scalar_pis(problem, self.variables[0])

        corr_sol = CorrSolution(name=self.name,
                                states=pis,
                                components=clist)

        self.save(corr_sol, problem,
                  variables=problem.create_variables([self.variables[0]]))

        return corr_sol

class OnesDim(CorrMiniApp):

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)
        var_name = self.variables[0]
        var = problem.get_variables(auto_create=True)[var_name]

        dim = problem.domain.mesh.dim
        nnod = var.n_nod
        e00 = nm.zeros((nnod, dim), dtype=var.dtype)
        e1 = nm.ones((nnod,), dtype=var.dtype)

        ones = nm.zeros((dim,), dtype=object)
        clist = []
        for ir in range(dim):
            aux = e00.copy()
            aux[:,ir] = e1
            ones[ir] = {var_name : nm.ascontiguousarray(aux)}
            clist.append((ir,))

        corr_sol = CorrSolution(name=self.name,
                                states=ones,
                                components=clist)

        self.save(corr_sol, problem,
                  variables=problem.create_variables([self.variables[0]]))

        return corr_sol


class CorrEval(CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)
        expr = self.expression
        for req in map(rm_multi, self.requires):
            expr = expr.replace(req, "data['%s']" % req)

        val = eval(expr)


        if type(val) is dict:
            corr_sol = CorrSolution(name=self.name,
                                    state=val)
        elif type(val) is nm.ndarray:
            if val.dtype == object:
                corr_sol = CorrSolution(name=self.name,
                                        states=val,
                                        components=['data'])
            else:
                ndof, ndim = val.shape
                state = {self.variable: val.reshape((ndof * ndim,))}
                corr_sol = CorrSolution(name=self.name,
                                        state=state)
        else:
            corr_sol = val

        cvars = problem.create_variables([self.variable])
        self.save(corr_sol, problem, variables=cvars)

        return corr_sol


class CorrNN(CorrMiniApp):
    """ __init__() kwargs:
        {
             'ebcs' : [],
             'epbcs' : [],
             'equations' : {},
             'set_variables' : None,
        },
    """

    def set_variables_default(variables, ir, ic, set_var, data):
        for (var, req, comp) in set_var:
            variables[var].set_data(data[req].states[ir,ic][comp])

    set_variables_default = staticmethod(set_variables_default)

    def __init__(self, name, problem, kwargs):
        """When dim is not in kwargs, problem dimension is used."""
        CorrMiniApp.__init__(self, name, problem, kwargs)
        self.set_default('dim', problem.get_dim())

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        problem.set_equations(self.equations)

        problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                           lcbc_names=self.get('lcbcs', []))

        problem.update_materials(problem.ts)

        self.init_solvers(problem)

        variables = problem.get_variables()

        states = nm.zeros((self.dim, self.dim), dtype=object)
        clist = []
        for ir in range(self.dim):
            for ic in range(self.dim):
                if isinstance(self.set_variables, list):
                    self.set_variables_default(variables, ir, ic,
                                               self.set_variables, data)
                else:
                    self.set_variables(variables, ir, ic, **data)

                problem.homogen_corr_id = (self.name, (ir, ic))
                state = problem.solve(update_materials=False,
                                      save_results=False)
                assert_(state.has_ebc())
                states[ir,ic] = state.get_state_parts()

                clist.append((ir, ic))

        corr_sol = CorrSolution(name=self.name,
                                states=states,
                                components=clist)

        self.save(corr_sol, problem)

        return corr_sol

class CorrN(CorrMiniApp):

    def set_variables_default(variables, ir, set_var, data):
        for (var, req, comp) in set_var:
            variables[var].set_data(data[req].states[ir][comp])

    set_variables_default = staticmethod(set_variables_default)

    def __init__(self, name, problem, kwargs):
        """When dim is not in kwargs, problem dimension is used."""
        CorrMiniApp.__init__(self, name, problem, kwargs)
        self.set_default('dim', problem.get_dim())

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        problem.set_equations(self.equations)

        problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                           lcbc_names=self.get('lcbcs', []))

        problem.update_materials(problem.ts)

        self.init_solvers(problem)

        variables = problem.get_variables()

        states = nm.zeros((self.dim,), dtype=object)
        clist = []
        for ir in range(self.dim):
            if isinstance(self.set_variables, list):
                self.set_variables_default(variables, ir,
                                           self.set_variables, data)
            else:
                self.set_variables(variables, ir, **data)

            problem.homogen_corr_id = (self.name, (ir,))
            state = problem.solve(update_materials=False,
                                  save_results=False)
            assert_(state.has_ebc())
            states[ir] = state.get_state_parts()

            clist.append((ir,))

        corr_sol = CorrSolution(name=self.name,
                                states=states,
                                components=clist)

        self.save(corr_sol, problem)

        return corr_sol

class CorrDimDim(CorrNN):
    pass

class CorrDim(CorrN):
    pass

class CorrOne(CorrMiniApp):

    def set_variables_default(variables, set_var, data):
        for (var, req, comp) in set_var:
            variables[var].set_data(data[req].state[comp])

    set_variables_default = staticmethod(set_variables_default)

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        problem.set_equations(self.equations)

        problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                           lcbc_names=self.get('lcbcs', []))

        problem.update_materials(problem.ts)

        self.init_solvers(problem)

        variables = problem.get_variables()

        if hasattr(self, 'set_variables'):
            if isinstance(self.set_variables, list):
                self.set_variables_default(variables, self.set_variables,
                                           data)
            else:
                self.set_variables(variables, **data)

        problem.homogen_corr_id = (self.name, None)
        state = problem.solve(update_materials=False,
                              save_results=False)
        assert_(state.has_ebc())

        corr_sol = CorrSolution(name=self.name,
                                state=state.get_state_parts())

        self.save(corr_sol, problem)

        return corr_sol

class CorrSetBCS(CorrMiniApp):

    def __call__(self, problem=None, data=None):
        from sfepy.base.base import select_by_names
        from sfepy.discrete.variables import Variables
        from sfepy.discrete.conditions import Conditions

        problem = get_default(problem, self.problem)

        conf_ebc = select_by_names(problem.conf.ebcs, self.ebcs)
        conf_epbc = select_by_names(problem.conf.epbcs, self.epbcs)
        ebcs = Conditions.from_conf(conf_ebc, problem.domain.regions)
        epbcs = Conditions.from_conf(conf_epbc, problem.domain.regions)

        conf_variables = select_by_names(problem.conf.variables, self.variable)
        variables = Variables.from_conf(conf_variables, problem.fields)
        variables.equation_mapping(ebcs, epbcs, problem.ts, problem.functions)
        variables.init_state()
        variables.fill_state(0.0)
        variables.apply_ebc()

        corr_sol = CorrSolution(name=self.name,
                                state=variables.get_state_parts())

        self.save(corr_sol, problem, variables=variables)

        return corr_sol

class CorrEqPar(CorrOne):
    """
    The corrector which equation can be parametrized via 'eq_pars',
    the dimension is given by the number of parameters.

    Example:

        'equations': 'dw_diffusion.5.Y(mat.k, q, p) =
                      dw_integrate.5.%s(q)',
        'eq_pars': ('bYMp', 'bYMm'),
        'class': cb.CorrEqPar,

    """

    def __init__(self, name, problem, kwargs):
        """When dim is not in kwargs, problem dimension is used."""
        CorrMiniApp.__init__(self, name, problem, kwargs)
        self.set_default('dim', len(self.eq_pars))

    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        states = nm.zeros((self.dim,), dtype=object)
        clist = []

        eqns ={}
        for ir in range(self.dim):
            for key_eq, val_eq in self.equations.items():
                eqns[key_eq] = val_eq % self.eq_pars[ir]

            problem.set_equations(eqns)

            problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                               lcbc_names=self.get('lcbcs', []))

            problem.update_materials(problem.ts)

            self.init_solvers(problem)

            variables = problem.get_variables()

            if hasattr(self, 'set_variables'):
                if isinstance(self.set_variables, list):
                    self.set_variables_default(variables, self.set_variables,
                                               data)
                else:
                    self.set_variables(variables, **data)

            problem.homogen_corr_id = (self.name, (ir,))
            state = problem.solve(update_materials=False,
                                  save_results=False)
            assert_(state.has_ebc())

            states[ir] = state.get_state_parts()
            clist.append((ir,))

        corr_sol = CorrSolution(name=self.name,
                                states=states,
                                components=clist)

        self.save(corr_sol, problem)

        return corr_sol

class CoefDummy(MiniAppBase):
    """
    Dummy class serving for computing and returning its requirements.
    """

    def __call__(self, volume=None, problem=None, data=None):
        return data

class TSTimes(MiniAppBase):
    """Coefficient-like class, returns times of the time stepper."""
    def __call__(self, volume=None, problem=None, data=None):
        problem = get_default(problem, self.problem)
        problem.init_solvers()
        return problem.get_timestepper().times

class VolumeFractions(MiniAppBase):
    """Coefficient-like class, returns volume fractions of given regions within
    the whole domain."""
    def __call__(self, volume=None, problem=None, data=None):
        problem = get_default(problem, self.problem)

        vf = {}
        for region_name in self.regions:
            vkey = 'volume_%s' % region_name
            key = 'fraction_%s' % region_name

            equations, variables = problem.create_evaluable(
                self.expression % region_name)
            val = eval_equations(equations, variables).real

            vf[vkey] = nm.asarray(val, dtype=nm.float64)
            vf[key] = vf[vkey] / self._get_volume(volume)

        return vf


class CoefMN(MiniAppBase):
    @staticmethod
    def set_variables_default(variables, ir, ic, mode, set_var, data, dtype):
        def get_corr_state(corr, ir, ic):
            if hasattr(corr, 'states'):
                if ir is None:
                    return corr.states[ic]
                elif ic is None:
                    return corr.states[ir]
                else:
                    return corr.states[ir, ic]
            else:
                return corr.state

        if mode == 'row_only':
            act_set_var = set_var
        else:
            mode2var = {'row': 0, 'col': 1}
            aux = set_var[mode2var[mode]]
            act_set_var = aux[:] if isinstance(aux, list) else [aux]
            act_set_var += set_var[2:]

        for (var, req, comp) in act_set_var:
            if type(req) is tuple:
                val = get_corr_state(data[req[0]], ir, ic)[comp].copy()
                val = nm.asarray(val, dtype=dtype)
                for ii in req[1:]:
                    val += get_corr_state(data[ii], ir, ic)[comp]

            else:
                val = get_corr_state(data[req], ir, ic)[comp]

            variables[var].set_data(val)

    def __init__(self, name, problem, kwargs):
        """When dim is not in kwargs, problem dimension is used."""
        MiniAppBase.__init__(self, name, problem, kwargs)
        self.set_default('dim', problem.get_dim())

    def get_coef(self, row, col, volume, problem, data):
        problem = get_default(problem, self.problem)
        term_mode = self.term_mode
        equations, variables = problem.create_evaluable(self.expression,
                                                        term_mode=term_mode)

        coef = nm.zeros((len(row), len(col)), dtype=self.dtype)

        for ir, (irr, icr) in enumerate(row):
            if isinstance(self.set_variables, list):
                self.set_variables_default(variables, irr, icr, 'row',
                                           self.set_variables, data,
                                           self.dtype)
            else:
                self.set_variables(variables, irr, icr, 'row', **data)

            for ic, (irc, icc) in enumerate(col):
                if isinstance(self.set_variables, list):
                    self.set_variables_default(variables, irc, icc, 'col',
                                               self.set_variables, data,
                                               self.dtype)
                else:
                    self.set_variables(variables, irc, icc, 'col', **data)

                val = eval_equations(equations, variables, term_mode=term_mode)
                coef[ir, ic] = val

        coef /= self._get_volume(volume)

        return coef

    def __call__(self, volume, problem=None, data=None):
        if isinstance(self.dim, Iterable) and len(self.dim) >= 2:
            dim1, dim2 = self.dim[:2]
        else:
            dim1 = dim2 = self.dim

        row = [(ii, None) for ii in range(dim1)]
        col = [(None, ii) for ii in range(dim2)]

        return self.get_coef(row, col, volume, problem, data)


class CoefDimDim(CoefMN):
    pass


class CoefSymSym(CoefMN):
    iter_sym = staticmethod(iter_sym)
    is_sym = True

    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        isym = [ii for ii in self.iter_sym(problem.get_dim())]

        return self.get_coef(isym, isym, volume, problem, data)


class CoefNonSymNonSym(CoefSymSym):
    iter_sym = staticmethod(iter_nonsym)
    is_sym = False


class CoefDimSym(CoefMN):
    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        dim = problem.get_dim()
        row = [(ii, None) for ii in range(dim)]
        col = [ii for ii in iter_sym(dim)]

        return self.get_coef(row, col, volume, problem, data)

class CoefSymDim(CoefMN):
    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        dim = problem.get_dim()
        row = [ii for ii in iter_sym(dim)]
        col = [(ii, None) for ii in range(dim)]

        return self.get_coef(row, col, volume, problem, data)

class CoefN(CoefMN):
    @staticmethod
    def set_variables_default(variables, ir, ic, mode, set_var, data, dtype):
        mode = mode + '_only'
        CoefMN.set_variables_default(variables, ir, ic, mode, set_var, data,
                                     dtype)

    def get_coef(self, row, volume, problem, data):
        problem = get_default(problem, self.problem)
        term_mode = self.term_mode
        equations, variables = problem.create_evaluable(self.expression,
                                                        term_mode=term_mode)

        coef = nm.zeros((len(row),), dtype=self.dtype)

        for ii, (ir, ic) in enumerate(row):
            if isinstance(self.set_variables, list):
                self.set_variables_default(variables, ir, ic, 'row',
                                           self.set_variables, data, self.dtype)
            else:
                self.set_variables(variables, ir, ic, 'row', **data)

            val = eval_equations(equations, variables, term_mode=term_mode)
            coef[ii] = val

        coef /= self._get_volume(volume)

        return coef

    def __call__(self, volume, problem=None, data=None):
        row = [(ii, None) for ii in range(self.dim)]

        return self.get_coef(row, volume, problem, data)


class CoefDim(CoefN):
    pass


class CoefSym(CoefN):
    iter_sym = staticmethod(iter_sym)
    is_sym = True

    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        isym = [ii for ii in self.iter_sym(problem.get_dim())]

        return self.get_coef(isym, volume, problem, data)


class CoefNonSym(CoefSym):
    iter_sym = staticmethod(iter_nonsym)
    is_sym = False


class CoefOne(MiniAppBase):

    def set_variables_default(variables, set_var, data, dtype):
        for (var, req, comp) in set_var:
            if type(req) is tuple:
                val = data[req[0]].state[comp].copy()
                val = nm.asarray(val, dtype=dtype)
                for ii in req[1:]:
                    val += data[ii].state[comp]

            else:
                val = data[req].state[comp]

            variables[var].set_data(val)

    set_variables_default = staticmethod(set_variables_default)

    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)

        term_mode = self.term_mode
        equations, variables = problem.create_evaluable(self.expression,
                                                        term_mode=term_mode)

        if hasattr(self, 'set_variables'):
            if isinstance(self.set_variables, list):
                self.set_variables_default(variables, self.set_variables,
                                           data, self.dtype)
            else:
                self.set_variables(variables, **data)

        val = eval_equations(equations, variables,
                             term_mode=term_mode)

        coef = val / self._get_volume(volume)

        return coef


class CoefSum(MiniAppBase):

    def __call__(self, volume, problem=None, data=None):
        coef = nm.zeros_like(data[self.requires[0]])
        for req in map(rm_multi, self.requires):
            coef += data[req]

        return coef

class CoefEval(MiniAppBase):
    """
    Evaluate expression.
    """
    def __call__(self, volume, problem=None, data=None):
        expr = self.expression
        for req in map(rm_multi, self.requires):
            expr = expr.replace(req, "data['%s']" % req)

        coef = eval(expr)

        return coef

class CoefNone(MiniAppBase):

    def __call__(self, volume, problem=None, data=None):

        coef = 0.0

        return coef

class CoefExprPar(MiniAppBase):
    """
    The coefficient which expression can be parametrized via 'expr_pars',
    the dimension is given by the number of parameters.

    Example:

        'expression': 'dw_surface_ndot.5.Ys(mat_norm.k%d, corr1)',
        'expr_pars': [ii for ii in range(dim)],
        'class': cb.CoefExprPar,

    """
    def set_variables_default(variables, ir, set_var, data):
        for (var, req, comp) in set_var:
            if hasattr(data[req], 'states'):
                variables[var].set_data(data[req].states[ir][comp])

            else:
                variables[var].set_data(data[req].state[comp])

    set_variables_default = staticmethod(set_variables_default)

    def __init__(self, name, problem, kwargs):
        """When dim is not in kwargs, problem dimension is used."""
        MiniAppBase.__init__(self, name, problem, kwargs)
        dim = len(self.expr_pars)
        self.set_default('dim', dim)

    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)

        coef = nm.zeros((self.dim,), dtype=self.dtype)
        term_mode = self.term_mode

        for ir in range(self.dim):
            expression = self.expression % self.expr_pars[ir]
            equations, variables = \
              problem.create_evaluable(expression, term_mode=term_mode)

            if hasattr(self, 'set_variables'):
                if isinstance(self.set_variables, list):
                    self.set_variables_default(variables, ir,
                                               self.set_variables, data)
                else:
                    self.set_variables(variables, ir, **data)

            val = eval_equations(equations, variables,
                                 term_mode=term_mode)
            coef[ir] = val

        coef /= self._get_volume(volume)

        return coef
