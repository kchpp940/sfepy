"""
Unified convergence history for nonlinear and time stepping solvers.

This module provides a lightweight structured container for recording solver
progress in a form that can be easily consumed by user-defined callbacks and
post-processing hooks attached to :class:`sfepy.discrete.problem.Problem`.

A :class:`ConvergenceHistory` instance is automatically attached to the solver
``status`` object (``status.history``) when ``status`` is an
:class:`sfepy.base.base.IndexedStruct` or dict-like.  Users that want to
collect convergence data can either:

* read ``status.history`` after :func:`Problem.solve()` returns, or
* pass a custom :class:`ConvergenceHistory` via ``status``.

The history is organized as:

.. code-block:: text

    history
      +-- nls  -> list of NLS runs (one per time step or solve call)
      |           each run contains ``iterations`` (per Newton step) and
      |           ``final`` (the aggregated NLS status).
      +-- ts   -> list of time step records with dt, tsc decisions, etc.
      +-- ls   -> list of linear solver statistics collected from NLS.

The records are simple dictionaries so they can be serialized to JSON/CSV or
plotted without additional conversion.
"""
from sfepy.base.base import Struct


class ConvergenceHistory(Struct):
    """
    Structured container for Newton iteration, linear solver residual and
    time step / adaptive step size history.

    Attributes
    ----------
    nls : list
        One entry per nonlinear solve.  Each entry is a dict with:

        - ``iterations``: list of per-iteration records
          (``err``, ``err_rel``, ``ls_iter``, ``ls_eps_a``, ``ls_eps_r``,
          ``time``, ``ok``).
        - ``final``: aggregated status dict (``n_iter``, ``ls_n_iter``,
          ``err0``, ``err``, ``condition``, ``time_stats``).
    ts : list
        One entry per time step.  Each entry is a dict with:

        - ``step``, ``time``, ``dt``, ``dt_new`` (only for adaptive solvers).
        - ``tsc``: dict with controller-specific information
          (``result``, ``u_err``, ``v_err``, ``emax``, ...).
    ls : list
        Aggregated linear solver statistics per nonlinear solve.
    """
    name = 'convergence_history'

    def __init__(self, **kwargs):
        Struct.__init__(self, nls=[], ts=[], ls=[], **kwargs)

    # ------------------------------------------------------------------ NLS --
    def begin_nls(self, extra=None):
        """
        Start a new nonlinear solve record.

        Returns
        -------
        record : dict
            The newly created record so callers can attach custom metadata.
        """
        record = {'iterations': [], 'final': {}}
        if extra:
            record.update(extra)
        self.nls.append(record)
        return record

    def record_nls_iteration(self, record, **fields):
        """
        Append a per-iteration entry to the given NLS ``record``.
        """
        record['iterations'].append(fields)

    def end_nls(self, record, **final):
        """
        Finalize an NLS record by attaching the aggregated final status.
        """
        record['final'].update(final)

    # ------------------------------------------------------------------- TS --
    def record_ts(self, **fields):
        """
        Append a new time step record.
        """
        self.ts.append(fields)
        return self.ts[-1]

    # ------------------------------------------------------------------- LS --
    def record_ls(self, **fields):
        """
        Append linear solver statistics.
        """
        self.ls.append(fields)
        return self.ls[-1]

    # ---------------------------------------------------------------- utils --
    def flatten_nls(self, include_ts=False, ts=None):
        """
        Flatten the nested NLS iteration data into a list of dicts, useful
        for CSV export or pandas ingestion.

        Parameters
        ----------
        include_ts : bool
            If True, copy the corresponding time step information onto each
            iteration record.
        ts : TimeStepper or None
            Optional time stepper; ignored if ``include_ts`` is False.
        """
        out = []
        for i_run, run in enumerate(self.nls):
            ts_info = run.get('ts', {}) if include_ts else {}
            for i_iter, it in enumerate(run.get('iterations', [])):
                row = {'run': i_run, 'iter': i_iter}
                row.update(ts_info)
                row.update(it)
                out.append(row)
        return out

    def summary(self):
        """
        Return a compact summary dict describing the recorded history.
        """
        n_runs = len(self.nls)
        n_iters = sum(len(r.get('iterations', [])) for r in self.nls)
        n_ts = len(self.ts)
        n_accept = sum(1 for r in self.ts
                       if r.get('tsc', {}).get('result') == 'accept')
        n_reject = sum(1 for r in self.ts
                       if r.get('tsc', {}).get('result') == 'reject')
        return {
            'n_nls_runs': n_runs,
            'n_newton_iterations': n_iters,
            'n_time_steps': n_ts,
            'n_ts_accepted': n_accept,
            'n_ts_rejected': n_reject,
        }


def attach_history(status):
    """
    Attach a :class:`ConvergenceHistory` instance to ``status`` if none is
    present.  Returns the history object.
    """
    if status is None:
        return None
    history = status.get('history', None) if hasattr(status, 'get') else None
    if history is None:
        history = ConvergenceHistory()
        try:
            status['history'] = history
        except Exception:
            status.history = history
    return history


def get_history(status):
    """
    Try to obtain a :class:`ConvergenceHistory` from ``status``.

    The function walks the chain ``status`` -> ``status.status`` -> ... so that
    nested solvers (for example NLS solvers invoked by a TS solver) can
    find the same history instance attached to the outermost status.
    """
    seen = set()
    cur = status
    while cur is not None:
        cid = id(cur)
        if cid in seen:
            break
        seen.add(cid)
        if hasattr(cur, 'get'):
            try:
                h = cur.get('history', None)
            except Exception:
                h = getattr(cur, 'history', None)
        else:
            h = getattr(cur, 'history', None)
        if h is not None:
            return h
        if hasattr(cur, 'get'):
            try:
                parent = cur.get('status', None)
            except Exception:
                parent = getattr(cur, 'status', None)
        else:
            parent = getattr(cur, 'status', None)
        cur = parent
    return None
