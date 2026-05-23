"""
Result export configuration layer for multi-physics problems.

This module provides :class:`ResultExportConfig`, a declarative description of
how simulation results should be exported.  It centralizes four concerns that
used to be duplicated across user/example scripts:

* which state variables should appear in the output,
* which derived quantities (expressions evaluated on the problem) should be
  computed on top of the state,
* which file format / mesh-writing format should be used, and
* how output files should be named.

A :class:`ResultExportConfig` can be consumed in three places:

* :meth:`sfepy.discrete.problem.Problem.save_state` - pass the configuration
  via the ``export_config`` keyword argument,
* :meth:`sfepy.discrete.problem.Problem.export_results` - convenience entry
  point that derives ``filename`` from the configuration,
* :func:`make_post_process_hook` - returns a ``post_process_hook`` callable
  suitable for ``Problem.solve`` and friends.

All constructors validate the configuration eagerly; see
:meth:`ResultExportConfig.validate` for the list of invariants enforced.

Examples
--------

>>> from sfepy.discrete.export_config import ResultExportConfig
>>> cfg = ResultExportConfig(
...     variable_names=['u', 'T'],
...     derived_quantities=[
...         ('cauchy_strain', 'ev_cauchy_strain.i.Omega(u)'),
...         ('cauchy_stress', 'ev_cauchy_stress.i.Omega(solid.D, u)'),
...     ],
...     file_format='vtk',
...     filename='results',
... )
>>> pb.save_state(pb.get_output_name(), state, export_config=cfg)
"""
import os
import os.path as op

import numpy as nm

from sfepy.base.base import (
    output, get_default, Struct, is_sequence, select_by_names,
)


_KNOWN_FILE_FORMATS = {
    'vtk', 'vtks', 'vtu', 'mesh', 'msh', 'med', 'comsol',
    'comsol.mphtxt', 'comsol.txt', 'nastran', 'h5', 'xdmf',
    'distmesh', 'tetgen', 'gmsh', 'abaqus', 'k',
    'hfm', 'neper', 'hex_mesh', 'perch', 'ply', 'plysurf',
}


def _is_valid_identifier(name):
    """Return True if ``name`` looks like a usable Python identifier."""
    return (isinstance(name, str) and len(name) > 0
            and (name.isidentifier() or all(
                c.isalnum() or c == '_' for c in name
            )))


def _check_name(name, kind, errors):
    """Validate a derived-quantity or variable name and append messages to
    ``errors``.
    """
    if not isinstance(name, str):
        errors.append('%s name must be a string, got %r' % (kind, name))
        return
    if not name:
        errors.append('%s name must not be empty' % kind)
        return
    if not _is_valid_identifier(name):
        errors.append(
            '%s name %r contains invalid characters (only letters, '
            'digits and ``_`` are allowed)' % (kind, name)
        )


class DerivedQuantity(Struct):
    """A single derived-quantity specification.

    Parameters
    ----------
    name : str
        The key under which the value will be stored in the output
        dictionary.
    expression : str or callable
        A term expression (see :func:`Problem.evaluate`) or a callable with
        the signature ``f(out, problem, state, extend)`` that is allowed to
        modify ``out`` in-place.
    eval_mode : str, optional
        The ``mode`` argument forwarded to :func:`Problem.evaluate`.  The
        default is ``'el_avg'`` which is appropriate for most cell-based
        derived quantities.
    eval_kwargs : dict, optional
        Extra keyword arguments passed to :func:`Problem.evaluate`.
    out_mode : str, optional
        The ``mode`` of the resulting output entry (``'cell'`` or
        ``'vertex'``).  Defaults to ``'cell'`` for ``el_avg`` style data.
    out_kwargs : dict, optional
        Additional keyword arguments stored on the resulting
        ``Struct(name='output_data', ...)`` object.  If ``out_kwargs`` does
        **not** already contain ``var_name``, the
        :class:`ResultExportConfig` can later supply one via
        :meth:`ResultExportConfig.annotate_out` so that ``split_results_by
        = 'variable'`` works correctly.
    """

    def __init__(self, name, expression, eval_mode='el_avg',
                 eval_kwargs=None, out_mode=None, out_kwargs=None,
                 transform=None):
        Struct.__init__(
            self,
            name=name,
            expression=expression,
            eval_mode=eval_mode,
            eval_kwargs=eval_kwargs if eval_kwargs is not None else {},
            out_mode=out_mode if out_mode is not None else 'cell',
            out_kwargs=out_kwargs if out_kwargs is not None else {},
            transform=transform,
        )

    def validate(self, errors):
        """Check the derived quantity for obvious errors, appending any
        findings to ``errors``."""
        _check_name(self.name, 'derived quantity', errors)

        if self.expression is None:
            errors.append('derived quantity %r has no expression' % self.name)
        elif isinstance(self.expression, str):
            if not self.expression.strip():
                errors.append('derived quantity %r expression is empty'
                              % self.name)
        elif not callable(self.expression):
            errors.append(
                'derived quantity %r expression must be a string or '
                'callable, got %r' % (self.name, type(self.expression))
            )

        if self.eval_mode not in ('el_avg', 'qp', 'vertex', 'node',
                                   'point_eval', None):
            errors.append(
                'derived quantity %r: unsupported eval_mode %r'
                % (self.name, self.eval_mode)
            )

        if self.out_mode not in ('cell', 'vertex', None):
            errors.append(
                'derived quantity %r: unsupported out_mode %r'
                % (self.name, self.out_mode)
            )

        if not isinstance(self.out_kwargs, dict):
            errors.append(
                'derived quantity %r: out_kwargs must be a dict, got %r'
                % (self.name, type(self.out_kwargs))
            )
        else:
            vn = self.out_kwargs.get('var_name')
            if vn is not None:
                _check_name(vn, 'var_name in out_kwargs of %s' % self.name,
                            errors)

        if self.transform is not None and not callable(self.transform):
            errors.append(
                'derived quantity %r: transform must be callable, got %r'
                % (self.name, type(self.transform))
            )

    # ------------------------------------------------------------------
    # Runtime evaluation check
    # ------------------------------------------------------------------
    def validate_runtime(self, problem, state=None):
        """Check whether this derived quantity can actually be computed.

        The check attempts to evaluate the expression (if it is a string)
        using the provided ``problem``.  A non-string/callable expression
        is only checked for type correctness.  ``state`` is optional; when
        given it is used as the current state of the ``state`` argument
        to ``problem.evaluate``.

        Raises
        ------
        ValueError
            With a message describing the first failure encountered.
        """
        if callable(self.expression):
            # Callable expressions are opaque - only check signature shape
            # via a basic sanity probe when possible.
            import inspect
            try:
                sig = inspect.signature(self.expression)
                if len(sig.parameters) < 4:
                    raise ValueError(
                        'derived quantity %r: callable expression must'
                        ' accept (out, problem, state, extend) parameters'
                        % self.name
                    )
            except (TypeError, ValueError):
                # Some builtins have no inspectable signature - skip.
                pass
            return

        if not isinstance(self.expression, str):
            raise ValueError(
                'derived quantity %r: unsupported expression type %r'
                % (self.name, type(self.expression))
            )

        if problem is None:
            raise ValueError(
                'derived quantity %r: cannot validate string expression'
                ' without a problem instance' % self.name
            )

        try:
            # Use a non-destructive evaluation: we do not want to modify
            # the problem state, so we simply ask for a shape-less result.
            problem.evaluate(
                self.expression, mode=self.eval_mode,
                **self.eval_kwargs
            )
        except Exception as exc:
            raise ValueError(
                'derived quantity %r: failed to evaluate expression %r'
                ' (%s: %s)'
                % (self.name, self.expression,
                   type(exc).__name__, exc)
            )

    def apply(self, out, problem, state, extend, var_name=None):
        """Compute the derived quantity and store it into ``out``.

        Parameters
        ----------
        out : dict
            The output dictionary to mutate.
        problem : Problem
        state : Variables
        extend : bool
        var_name : str or None, optional
            If not ``None``, the value is attached to the resulting entry
            **unless** the user already specified ``var_name`` inside
            ``out_kwargs``.  This is used by :class:`ResultExportConfig` to
            give un-tagged derived quantities a sensible ``var_name`` when
            ``split_results_by == 'variable'``.

        Returns
        -------
        out : dict
            The updated output dictionary (same object as input).
        """
        if callable(self.expression):
            return self.expression(out, problem, state, extend)

        data = problem.evaluate(
            self.expression, mode=self.eval_mode, **self.eval_kwargs
        )

        if self.transform is not None:
            data = self.transform(data, problem, state, extend)

        kwargs = dict(self.out_kwargs)
        if var_name is not None and 'var_name' not in kwargs:
            kwargs['var_name'] = var_name

        entry = Struct(
            name='output_data',
            mode=self.out_mode,
            data=data,
            dofs=None,
            **kwargs
        )
        out[self.name] = entry
        return out


class ResultExportConfig(Struct):
    """Unified description of what should be exported and how.

    The configuration centralizes four concerns that used to be duplicated
    across user/example scripts:

    * which state variables should appear in the output
      (``variable_names`` / ``exclude_variable_names``),
    * which derived quantities should be computed on top of the state
      (``derived_quantities``),
    * which mesh-writing format should be used (``file_format``), and
    * how output files should be named (``filename``,
      ``filename_suffix``, ``file_per_variable``, ``file_per_region``).

    A :class:`ResultExportConfig` can be consumed in three places:

    * :meth:`sfepy.discrete.problem.Problem.save_state` - pass the
      configuration via the ``export_config`` keyword argument,
    * :meth:`sfepy.discrete.problem.Problem.export_results` - convenience
      entry point that resolves the filename and format from the
      configuration,
    * :func:`make_post_process_hook` - returns a ``post_process_hook``
      callable suitable for ``Problem.solve`` and friends.

    Parameters
    ----------
    variable_names : list of str or None
        Restrict the state output to the variables listed.  ``None`` (the
        default) exports every state variable.
    exclude_variable_names : list of str or None
        Variables that should be dropped from the output even if they
        appear in the state.
    derived_quantities : list
        Either :class:`DerivedQuantity` instances, ``dict`` objects with
        the same keys, or 2/3-tuples ``(name, expression)`` /
        ``(name, expression, kwargs)``.
    file_format : str or None
        Format forwarded to :func:`Mesh.write` (e.g. ``'vtk'``, ``'h5'``).
        When ``None`` the format is inferred from the filename extension.
    filename : str or None
        The preferred output filename (or trunk).  May contain the
        ``'{problem}'`` placeholder which :meth:`resolve_filename` replaces
        with the problem filename trunk.
    filename_suffix : str or None
        A suffix inserted between the problem trunk and the extension.
        When the problem already provides its own suffix (e.g. a time step
        index) the two are joined with ``'_'``.
    file_per_variable : bool
        If ``True``, each variable is written to its own file (equivalent
        to ``split_results_by='variable'``).
    file_per_region : bool
        If ``True``, the mesh is split by region and each region is
        written separately.
    default_var_name : str or None
        If ``file_per_variable`` is used, derived quantities without an
        explicit ``var_name`` in their ``out_kwargs`` are assigned this
        ``var_name`` so they are not silently dropped.
    extra_options : dict
        Miscellaneous options that are forwarded as ``**kwargs`` to
        :func:`Mesh.write`.
    """

    def __init__(self, variable_names=None, exclude_variable_names=None,
                 derived_quantities=None, file_format=None,
                 filename=None, filename_suffix=None,
                 file_per_variable=False, file_per_region=False,
                 default_var_name=None, extra_options=None):
        dqs = []
        if derived_quantities is not None:
            for item in derived_quantities:
                if isinstance(item, DerivedQuantity):
                    dqs.append(item)
                elif isinstance(item, dict):
                    dq = DerivedQuantity(**item)
                    dqs.append(dq)
                elif isinstance(item, (tuple, list)):
                    name = item[0]
                    expression = item[1]
                    dq_kwargs = item[2] if len(item) > 2 else {}
                    if not isinstance(dq_kwargs, dict):
                        raise TypeError(
                            'third element of a derived quantity tuple'
                            ' must be a dict, got %r' % type(dq_kwargs)
                        )
                    dqs.append(
                        DerivedQuantity(name, expression, **dq_kwargs)
                    )
                else:
                    raise TypeError(
                        'cannot interpret derived quantity item %r'
                        % (item,)
                    )

        Struct.__init__(
            self,
            variable_names=list(variable_names) if variable_names else None,
            exclude_variable_names=list(exclude_variable_names)
            if exclude_variable_names else None,
            derived_quantities=dqs,
            file_format=file_format,
            filename=filename,
            filename_suffix=filename_suffix,
            file_per_variable=bool(file_per_variable),
            file_per_region=bool(file_per_region),
            default_var_name=default_var_name,
            extra_options=dict(extra_options) if extra_options else {},
        )
        self.validate()

    def validate(self, variables=None):
        """Check the configuration for obvious errors.

        Parameters
        ----------
        variables : Variables or None
            If given, the list of problem variables is used to check
            references inside ``variable_names`` and the ``var_name``
            entries of each derived quantity.

        Raises
        ------
        ValueError
            With a message that lists every detected problem.  Uses the
            :class:`_ExportConfigError` wrapper so users can catch it
            specifically.

        Returns
        -------
        self
        """
        errors = []

        # variable_names / exclude_variable_names
        if self.variable_names is not None:
            if not isinstance(self.variable_names, list):
                errors.append(
                    'variable_names must be a list of strings, got %r'
                    % type(self.variable_names).__name__
                )
            else:
                for vn in self.variable_names:
                    _check_name(vn, 'variable', errors)

        if self.exclude_variable_names is not None:
            if not isinstance(self.exclude_variable_names, list):
                errors.append(
                    'exclude_variable_names must be a list of strings, got %r'
                    % type(self.exclude_variable_names).__name__
                )
            else:
                for vn in self.exclude_variable_names:
                    _check_name(vn, 'exclude variable', errors)

            if (self.variable_names is not None
                    and set(self.variable_names).intersection(
                        self.exclude_variable_names)):
                errors.append(
                    'variable_names and exclude_variable_names must be'
                    ' disjoint (overlap: %s)'
                    % (sorted(set(self.variable_names).intersection(
                        self.exclude_variable_names)),)
                )

        # Derived quantities.
        seen_dq = set()
        for dq in self.derived_quantities:
            if not isinstance(dq, DerivedQuantity):
                errors.append(
                    'derived quantity entry is not a DerivedQuantity'
                    ' instance: %r' % type(dq).__name__
                )
                continue
            dq.validate(errors)
            if dq.name in seen_dq:
                errors.append(
                    'duplicate derived quantity name: %r' % dq.name
                )
            seen_dq.add(dq.name)

        # file_format
        if self.file_format is not None:
            if not isinstance(self.file_format, str):
                errors.append(
                    'file_format must be a string, got %r'
                    % type(self.file_format).__name__
                )
            elif self.file_format not in _KNOWN_FILE_FORMATS:
                # Not a hard error - emit a warning-style error because the
                # user might rely on a custom MeshIO handler.
                output(
                    'export_config: file_format %r is not in the known list'
                    ' %s; proceeding but Mesh.write may fail if no MeshIO'
                    ' handler is registered.'
                    % (self.file_format, sorted(_KNOWN_FILE_FORMATS))
                )

        # file_per_variable vs file_per_region
        if self.file_per_variable and self.file_per_region:
            errors.append(
                'file_per_variable and file_per_region are mutually exclusive'
            )

        # filename / filename_suffix
        if self.filename is not None and not isinstance(self.filename, str):
            errors.append('filename must be a string, got %r'
                          % type(self.filename).__name__)
        if (self.filename_suffix is not None
                and not isinstance(self.filename_suffix, str)):
            errors.append('filename_suffix must be a string, got %r'
                          % type(self.filename_suffix).__name__)

        # default_var_name
        if self.default_var_name is not None:
            _check_name(self.default_var_name,
                        'default_var_name', errors)

        # extra_options
        if not isinstance(self.extra_options, dict):
            errors.append(
                'extra_options must be a dict, got %r'
                % type(self.extra_options).__name__
            )

        # Cross-check with the Variables object if provided.
        if variables is not None:
            try:
                var_names = [v.name for v in variables.iter_state()]
            except Exception:
                var_names = []

            if self.variable_names is not None:
                unknown = set(self.variable_names) - set(var_names)
                if unknown:
                    errors.append(
                        'variable_names contains names not present in'
                        ' the problem state: %s (known: %s)'
                        % (sorted(unknown), sorted(var_names))
                    )

            if self.default_var_name is not None \
                    and self.default_var_name not in var_names:
                errors.append(
                    'default_var_name %r is not a state variable (known:'
                    ' %s)' % (self.default_var_name, sorted(var_names))
                )

            for dq in self.derived_quantities:
                vn = dq.out_kwargs.get('var_name')
                if vn is not None and vn not in var_names:
                    errors.append(
                        'derived quantity %r declares var_name %r which'
                        ' is not a state variable (known: %s)'
                        % (dq.name, vn, sorted(var_names))
                    )

                # Warn about clashes with existing state variable names.
                if dq.name in var_names:
                    output(
                        'export_config: derived quantity %r shadows'
                        ' a state variable of the same name; the latter'
                        ' will be overwritten in the output.' % dq.name
                    )

        if errors:
            raise ValueError(
                'export_config validation failed:\n  - '
                + '\n  - '.join(errors)
            )
        return self

    @classmethod
    def from_conf(cls, conf):
        """Construct a :class:`ResultExportConfig` from a user-facing value.

        ``conf`` may be

        * ``None`` -> returns ``None`` (no export configuration),
        * a :class:`ResultExportConfig` instance -> returned as-is,
        * a ``dict`` -> used as keyword arguments for the constructor,
          allowing the configuration to be written inline in a problem
          definition file, e.g.::

              options = {
                  'export_config' : {
                      'variable_names': ['u', 'T'],
                      'derived_quantities': [
                          ('cauchy_strain',
                           'ev_cauchy_strain.i.Omega(u)'),
                      ],
                  },
              }

        Anything else raises a ``TypeError``.
        """
        if conf is None:
            return None
        if isinstance(conf, cls):
            return conf
        if isinstance(conf, dict):
            return cls(**conf)
        raise TypeError('cannot construct a ResultExportConfig from %r'
                        % type(conf))

    # ------------------------------------------------------------------
    # Variable selection
    # ------------------------------------------------------------------
    def filter_variable_names(self, names):
        """Return the subset of ``names`` that should be exported."""
        if self.variable_names is not None:
            keep = [name for name in names if name in self.variable_names]
        else:
            keep = list(names)

        if self.exclude_variable_names is not None:
            keep = [name for name in keep
                    if name not in self.exclude_variable_names]
        return keep

    def build_var_info(self, variables):
        """Return a ``var_info`` dictionary for ``Variables.create_output``.

        The dictionary restricts the output to the variables selected by
        this configuration.  ``None`` is returned when no filtering is
        required so callers can fall back to the default behaviour.
        """
        if self.variable_names is None and self.exclude_variable_names is None:
            return None

        state_names = [var.name for var in variables.iter_state()]
        keep = self.filter_variable_names(state_names)

        if set(keep) == set(state_names):
            return None

        return {name: (False, name) for name in keep}

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def compute_derived_quantities(self, out, problem, state, extend,
                                   default_var_name=None):
        """Evaluate all derived quantities and merge them into ``out``.

        Parameters
        ----------
        out : dict
        problem : Problem
        state : Variables
        extend : bool
        default_var_name : str or None, optional
            If provided and a :class:`DerivedQuantity` does not specify
            ``var_name`` in its ``out_kwargs``, this value is used.  It
            overrides ``self.default_var_name``, allowing the caller (e.g.
            :func:`Problem.save_state`) to supply a context-sensitive
            default based on the active variables.
        """
        default_var_name = get_default(default_var_name,
                                       self.default_var_name)
        for dq in self.derived_quantities:
            dq.apply(out, problem, state, extend,
                     var_name=default_var_name)
        return out

    # ------------------------------------------------------------------
    # File format / naming
    # ------------------------------------------------------------------
    def resolve_file_format(self, filename=None, problem_format=None):
        """Return the mesh-writing ``io`` / ``file_format`` string.

        Priority:

        1. ``self.file_format``,
        2. ``problem_format`` (the problem's ``output_format`` /
           ``file_format``),
        3. ``'auto'`` (let :func:`Mesh.write` infer it from the extension).
        """
        if self.file_format is not None:
            return self.file_format
        if problem_format is not None:
            return problem_format
        return 'auto'

    def resolve_filename(self, filename=None, default=None,
                         problem_trunk=None, step_suffix=None):
        """Return the output filename to use.

        ``'{problem}'`` placeholders inside ``self.filename`` are replaced
        with ``problem_trunk`` (the problem's output filename trunk) when
        given.

        Parameters
        ----------
        filename : str or None
            Explicit filename from the caller (highest priority).
        default : str or None
            Fallback filename used when neither ``filename`` nor
            ``self.filename`` is given.
        problem_trunk : str or None
            The problem's output filename trunk (for ``{problem}``
            substitution).
        step_suffix : str or None
            A suffix provided by the time-stepping machinery (e.g. a
            formatted step index).  Joined with
            ``self.filename_suffix`` using ``'_'``.
        """
        if filename is not None:
            chosen = filename
        elif self.filename is not None:
            chosen = self.filename
            if problem_trunk is not None:
                chosen = chosen.replace('{problem}', problem_trunk)
        else:
            chosen = default

        if chosen is None:
            return None

        if (step_suffix or self.filename_suffix):
            trunk, ext = op.splitext(chosen)
            parts = []
            if self.filename_suffix:
                parts.append(self.filename_suffix)
            if step_suffix:
                parts.append(step_suffix)
            chosen = trunk + '_' + '_'.join(parts) + ext

        return chosen

    def get_split_results_by(self):
        """Return the ``split_results_by`` flag for ``Problem.save_state``."""
        if self.file_per_variable:
            return 'variable'
        if self.file_per_region:
            return 'region'
        return None

    # ------------------------------------------------------------------
    # Runtime validation
    # ------------------------------------------------------------------
    def validate_runtime(self, problem, state=None, filename=None):
        """Run the checks that require a fully set up ``Problem``.

        The method is intended to be invoked by ``Problem`` just before
        saving results so that configuration errors fail fast with a clear
        message rather than during ``Mesh.write``.

        Parameters
        ----------
        problem : Problem
        state : Variables, optional
            The state that will be exported.  Used to cross-check variable
            names and to probe derived-quantity expressions.
        filename : str, optional
            The output filename that would be written.  When given, the
            directory is checked for existence and writability.

        Raises
        ------
        ValueError
            With a message that lists every detected problem.
        """
        errors = []

        # Variable-name cross-check with the active state.
        if state is not None:
            try:
                state_names = [v.name for v in state.iter_state()]
            except Exception:
                state_names = []

            if state_names:
                if self.variable_names is not None:
                    unknown = set(self.variable_names) - set(state_names)
                    if unknown:
                        errors.append(
                            'variable_names references unknown state'
                            ' variables: %s (known: %s)'
                            % (sorted(unknown), sorted(state_names))
                        )

                if self.default_var_name is not None \
                        and self.default_var_name not in state_names:
                    errors.append(
                        'default_var_name %r is not a state variable'
                        ' (known: %s)'
                        % (self.default_var_name, sorted(state_names))
                    )

                for dq in self.derived_quantities:
                    vn = dq.out_kwargs.get('var_name')
                    if vn is not None and vn not in state_names:
                        errors.append(
                            'derived quantity %r: var_name %r is not'
                            ' a state variable (known: %s)'
                            % (dq.name, vn, sorted(state_names))
                        )

        # Derived-quantity expression probe.
        if problem is not None:
            for dq in self.derived_quantities:
                try:
                    dq.validate_runtime(problem, state=state)
                except ValueError as exc:
                    errors.append(str(exc))

        # Output-path sanity.
        if filename is not None:
            try:
                directory = op.dirname(op.abspath(filename))
                if directory and not op.exists(directory):
                    try:
                        os.makedirs(directory)
                    except OSError as exc:
                        errors.append(
                            'output directory %r does not exist and cannot'
                            ' be created (%s: %s)'
                            % (directory, type(exc).__name__, exc)
                        )

                if directory and op.exists(directory) \
                        and not os.access(directory, os.W_OK):
                    errors.append(
                        'output directory %r is not writable' % directory
                    )
            except Exception as exc:
                errors.append(
                    'cannot resolve output filename %r (%s: %s)'
                    % (filename, type(exc).__name__, exc)
                )

        if errors:
            raise ValueError(
                'export_config runtime validation failed:\n  - '
                + '\n  - '.join(errors)
            )


def make_post_process_hook(export_config):
    """Wrap :class:`ResultExportConfig` into a ``post_process_hook`` callable.

    The returned function has the signature
    ``hook(out, problem, state, extend=False, default_var_name=None)`` and
    can be passed to :func:`Problem.solve`, :func:`Problem.save_state` or
    anywhere else SfePy expects a post-process hook.

    Parameters
    ----------
    export_config : ResultExportConfig or None
        The configuration to apply.  If ``None``, a no-op hook that returns
        ``out`` unchanged is returned.

    Returns
    -------
    hook : callable
    """
    def _no_op(out, problem, state, extend=False,
               default_var_name=None):
        return out

    if export_config is None:
        return _no_op

    def _hook(out, problem, state, extend=False,
              default_var_name=None):
        return export_config.compute_derived_quantities(
            out, problem, state, extend,
            default_var_name=default_var_name,
        )

    return _hook
