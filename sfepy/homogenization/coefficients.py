import json
import os.path as op

import numpy as nm

from sfepy.base.base import ordered_iteritems, Struct
from sfepy.base.ioutils import read_dict_hdf5, write_dict_hdf5
from sfepy.homogenization.utils import iter_sym

def coef_arrays_to_dicts(idict, format='%s/%d'):
    out = {}
    for k, v in idict.items():
        if isinstance(v, list):
            out.update({format %(k, ii): vv for ii, vv in enumerate(v)})
        else:
            out[k] = v

    return out

class Coefficients(Struct):
    """
    Class for storing (homogenized) material coefficients.
    """

    @staticmethod
    def from_file_hdf5(filename):
        obj = Coefficients()
        obj.__dict__ = read_dict_hdf5(filename)
        for key, val in obj.__dict__.items():
            if type(val) == list:
                for ii, vv in enumerate(val):
                    val[ii] = nm.array(vv, dtype=nm.float64)

        return obj

    @staticmethod
    def merge(coefficients, labels=None):
        """Merge several ``Coefficients`` instances into one.

        Each coefficient attribute becomes a list/array of length
        ``len(coefficients)``.  An extra attribute ``labels`` (if ``labels``
        is not ``None``) is added so that the merged object can be indexed
        by the label of the source case.

        Parameters
        ----------
        coefficients : iterable of Coefficients
            The per-case coefficients to merge.
        labels : iterable of str or None
            Case labels, used as the ``labels`` attribute on the returned
            object.
        """
        coefficients = list(coefficients)
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(coefficients):
                raise ValueError('length of labels must match the number of '
                                 'coefficients')
        merged = Coefficients()
        if labels is not None:
            merged.labels = list(labels)
        merged._num_cases = len(coefficients)

        attrs = set()
        for c in coefficients:
            if c is not None:
                attrs.update(getattr(c, '__dict__', {}).keys())

        for name in attrs:
            values = [getattr(c, name, None) if c is not None else None
                      for c in coefficients]
            try:
                merged.__dict__[name] = nm.array(values)
            except (ValueError, TypeError):
                merged.__dict__[name] = list(values)

        return merged

    @staticmethod
    def from_sweep_hdf5(filename):
        """Restore a merged ``Coefficients`` from a sweep HDF5 archive.

        Parameters
        ----------
        filename : str
            Path to the ``*_sweep.h5`` file written by
            ``HomogenizationApp._save_sweep_archive``.

        Returns
        -------
        merged : Coefficients
            A merged ``Coefficients`` with ``labels`` and per-case
            coefficient arrays under ``_cases/<label>/<coef_name>``.
        """
        import tables as pt

        fd = pt.open_file(filename, mode='r')
        try:
            raw_labels = fd.root._f_get_child('labels').read()
            labels = [s.decode() if isinstance(s, bytes) else s
                      for s in raw_labels]

            merged = Coefficients()
            merged.labels = labels
            merged._num_cases = len(labels)

            cases = {}
            for label in labels:
                try:
                    cgrp = fd.root._f_get_child('cases')._f_get_child(label)
                except LookupError:
                    continue
                for leaf in cgrp._f_leaves.values():
                    coef_name = leaf._v_name
                    if coef_name not in cases:
                        cases[coef_name] = [None] * len(labels)
                    idx = labels.index(label)
                    cases[coef_name][idx] = leaf.read()

            for coef_name, values in cases.items():
                try:
                    merged.__dict__[coef_name] = nm.array(values)
                except (ValueError, TypeError):
                    merged.__dict__[coef_name] = list(values)

            return merged
        finally:
            fd.close()

    def group_by(self, *keys):
        """Group case labels by sweep parameter values.

        Only works on merged ``Coefficients`` that carry
        ``_sweep_cases`` (i.e. produced by ``HomogenizationApp._run_sweep``).

        Parameters
        ----------
        keys : tuple of str
            Parameter names to group by (e.g. ``'material_index'``,
            ``'frequency'``).

        Returns
        -------
        groups : dict
            ``{tuple_of_values: [label, ...]}``.  When a single key is given
            the outer tuple is unwrapped.
        """
        sweep_cases = getattr(self, '_sweep_cases', None)
        if sweep_cases is None:
            raise ValueError('group_by requires _sweep_cases; '
                             'use Coefficients.from_sweep_hdf5() or '
                             'run HomogenizationApp with sweep options')

        groups = {}
        for c in sweep_cases:
            key = tuple(c.params.get(k) for k in keys)
            groups.setdefault(key, []).append(c.label)
        if len(keys) == 1:
            return {k[0]: v for k, v in groups.items()}
        return groups

    def select_case(self, case):
        """Return a new ``Coefficients`` with only the given case.

        ``case`` may be an ``int`` index or a ``str`` label (requires the
        instance to have a ``labels`` attribute).
        """
        if isinstance(case, str):
            labels = getattr(self, 'labels', None)
            if labels is None:
                raise ValueError('no labels on Coefficients; cannot index '
                                 'by name')
            try:
                idx = labels.index(case)
            except ValueError:
                raise KeyError('label %r not found' % case)
        else:
            idx = case
        out = Coefficients()
        for name, val in self.__dict__.items():
            if name in ('labels', '_num_cases', '_sweep_cases',
                        '_sweep_labels', '_sweep_indices', '_sweep_groups'):
                continue
            if isinstance(val, nm.ndarray):
                out.__dict__[name] = val[idx]
            elif isinstance(val, list):
                out.__dict__[name] = val[idx]
            else:
                out.__dict__[name] = val
        return out

    def to_file_hdf5(self, filename):
        write_dict_hdf5(filename, self.__dict__)

    def _escape_latex(self, txt):
        return txt.replace('_', r'\_').replace('%', r'\%')

    def _format(self, val):
        out = self._a_format % val
        if self._a_cdot:
            a1, a2 = out.split('e')
            if (self._a_filter is not None) and (int(a2) < self._a_filter):
                out = '0'

            else:
                out = r'%s \cdot 10^{%s}' % (a1, int(a2))

        return out

    def _write1d(self, fd, val):
        fd.write(r'  \begin{equation}')
        fd.write('\n')
        fd.write(r'    \left[')
        fd.write('\n')
        fd.write(', '.join([self._format(vv) for vv in val]))
        fd.write('\n')
        fd.write(r'    \right]')
        fd.write('\n')
        fd.write(r'  \end{equation}')
        fd.write('\n')

    def _write2d(self, fd, val):
        fd.write(r'  \begin{equation}')
        fd.write('\n')
        fd.write(r'    \left[\begin{array}{%s}' % ('c' * val.shape[0]))
        fd.write('\n')
        for ir in range(val.shape[1]):
            for ic in range(val.shape[0]):
                fd.write('    ' + self._format(val[ir,ic]))
                if ic < (val.shape[0] - 1):
                    fd.write(r' & ')
                elif ir < (val.shape[1] - 1):
                    fd.write(r' \\')
                    fd.write('\n')
        fd.write('\n')
        fd.write(r'    \end{array}\right]')
        fd.write('\n')
        fd.write(r'  \end{equation}')
        fd.write('\n')

    def _save_dict_latex(self, adict, fd, names, idx=None):
        fd.write(r'\begin{itemize}')
        fd.write('\n')
        for key, val in ordered_iteritems(adict):
            if key.startswith('_a_'): continue

            try:
                lname = names[key]
            except:
                lname = self._escape_latex(key)
            fd.write(r'\item %s:' % lname)
            fd.write('\n')

            if isinstance(val, list):
                if idx is not None:
                    val = val[idx]
                else:
                    raise NotImplementedError("'idx' must be set in the case "
                                              "of multi-coefficients!")

            if isinstance(val, dict):
                self._save_dict_latex(val, fd, names)

            elif isinstance(val, str):
                fd.write(self._escape_latex(val) + '\n')

            elif isinstance(val, float):
                fd.write('$' + self._format(val) + '$\n')

            elif isinstance(val, nm.ndarray):
                if val.ndim == 0:
                    fd.write('$' + self._format(val) + '$\n')

                elif val.ndim == 1:
                    self._write1d(fd, val)

                elif val.ndim == 2:
                    self._write2d(fd, val)

            else:
                fd.write('%s' % val)

        fd.write(r'\end{itemize}')
        fd.write('\n\n')


    def to_file_latex(self, filename, names, format='%.2e',
                      cdot=False, filter=None, idx=None):
        r"""
        Save the coefficients to a file in LaTeX format.

        Parameters
        ----------
        filename : str
            The name of the output file.
        names : dict
            Mapping of attribute names to LaTeX names.
        format : str
            Format string for numbers.
        cdot : bool
            For '%.e' formats only. If True, replace 'e'  by LaTeX '\cdot
            10^{exponent}' format.
        filter : int
            For '%.e' formats only. Typeset as 0, if exponent is less than
            `filter`.
        idx : int
            For multi-coefficients, set the coefficient index.
        """
        self._a_format = format
        self._a_cdot = cdot
        self._a_filter = filter

        fd = open(filename, 'w')
        self._save_dict_latex(self.__dict__, fd, names, idx)
        fd.close()

    def _save_dict(self, adict, fd, names, format):
        toremove = []
        adict_complex = {}
        for key, val in ordered_iteritems(adict):
            if (hasattr(val, 'dtype') and
                nm.issubdtype(val.dtype, nm.complexfloating)):
                adict_complex[key + '_real'] = val.real
                adict_complex[key + '_imag'] = val.imag
                toremove.append(key)
        for key in toremove:
            del(adict[key])

        adict.update(adict_complex)

        for key, val in ordered_iteritems(adict):
            try:
                lname = names[key]
            except:
                lname = key
            fd.write('%s:\n' % lname)

            if hasattr(val, 'to_file_txt'):
                if val.to_file_txt is not None:
                    val.to_file_txt(fd, format, val)

                else:
                    fd.write('--\n')

            elif isinstance(val, dict):
                self._save_dict(val, fd, names, format)
                fd.write('\n')

            elif isinstance(val, list):
                if isinstance(val[0], str):
                    fd.write('\n'.join(val) + '\n')

            elif isinstance(val, str):
                fd.write(val + '\n')

            elif isinstance(val, float):
                fd.write('%e\n' % val)

            elif isinstance(val, nm.ndarray):
                if val.ndim == 0:
                    fd.write(format % val)
                    fd.write('\n')

                elif val.ndim == 1:
                    for ic in range(val.shape[0]):
                        fd.write(format % val[ic])
                        if ic < (val.shape[0] - 1):
                            fd.write(', ')
                        else:
                            fd.write('\n')

                elif val.ndim == 2:
                    for ir in range(val.shape[0]):
                        for ic in range(val.shape[1]):
                            fd.write(format % val[ir,ic])
                            if ic < (val.shape[1] - 1):
                                fd.write(', ')
                            elif ir < (val.shape[0] - 1):
                                fd.write(';\n')
                    fd.write('\n')

                elif val.ndim == 3:
                    for ii in range(val.shape[0]):
                        fd.write('  step %d:\n' % ii)
                        for ir in range(val.shape[1]):
                            for ic in range(val.shape[2]):
                                fd.write('  ' + format % val[ii,ir,ic])
                                if ic < (val.shape[2] - 1):
                                    fd.write(', ')
                                elif ir < (val.shape[1] - 1):
                                    fd.write(';\n')
                        fd.write('\n')
                    fd.write('\n')

            else:
                fd.write('--\n')

            fd.write('\n')

    def to_file_txt(self, filename, names, format):

        fd = open(filename, 'w')
        self._save_dict(coef_arrays_to_dicts(self.__dict__), fd, names, format)
        fd.close()

    _table_vector = r"""
\begin{center}
\begin{tabular}{cc}
i & value \\
%s
\end{tabular}
\end{center}
    """

    _table_matrix_1 = r"""
\begin{center}
\begin{tabular}{cc}
ij & value \\
%s
\end{tabular}
\end{center}
    """

    _table_matrix_2 = r"""
\begin{center}
\begin{tabular}{cc}
ijkl & value \\
%s
\end{tabular}
\end{center}
    """

    _itemize = r"""
\begin{itemize}
%s
\end{itemize}
    """

    ##
    # c: 09.07.2008, r: 09.07.2008
    def _typeset(self, val, dim, style='table', format='%f', step=None):
        sym = (dim + 1) * dim // 2

        mode = None
        if val.ndim == 0:
            mode = 'scalar'

        elif val.ndim == 1:
            if val.shape[0] == 1:
                mode = 'scalar'
            elif val.shape[0] == dim:
                mode = 'vector'
            elif val.shape[0] == sym:
                mode = 'matrix_t1d'

        elif val.ndim == 2:
            if val.shape[0] == dim:
                mode = 'matrix_2D'
            elif val.shape[0] == sym:
                mode = 'matrix_t2d'

        out = ''
        if mode == 'scalar':
            out = format % val
        elif mode == 'vector':
            aux = ' \\\\\n'.join([r'$_%d$ & %s' % (ir + 1, format % val[ir])
                                  for ir in range(dim)])
            out = self._table_vector % aux
        elif mode == 'matrix_t1d':
            aux = ' \\\\\n'.join([r'$_{%d%d}$ & %s' % (ir + 1, ic + 1,
                                                       format % val[ii])
                                  for ii, (ir, ic)
                                  in enumerate(iter_sym(dim))])
            out = self._table_matrix_1 % aux
        elif mode == 'matrix_2D':
            aux = ' \\\\\n'.join([r'$_{%d%d}$ & %s' % (ir + 1, ic + 1,
                                                       format % val[ir,ic])
                                  for ir in range(dim)
                                  for ic in range(dim)])
            out = self._table_matrix_1 % aux
        elif mode == 'matrix_t2d':
            aux = ' \\\\\n'.join([r'$_{%d%d%d%d}$ & %s' % (irr + 1, irc + 1,
                                                           icr + 1, icc + 1,
                                                           format % val[ii,jj])
                                  for ii, (irr, irc)
                                  in enumerate(iter_sym(dim))
                                  for jj, (icr, icc)
                                  in enumerate(iter_sym(dim))])
            out = self._table_matrix_2 % aux
        return out

    def to_latex(self, attr_name, dim, style='table', format='%f', step=None):

        val = getattr(self, attr_name)
        if step is not None:
            val = val[step]

        if isinstance(val, dict):
            aux = ''
            for key, dval in val.items():
                aux2 = r'\item %s : %s' % (key,
                                           self._typeset(dval, dim, style,
                                                         format, step))
                aux = '\n'.join((aux, aux2))
            out = self._itemize % aux
        else:
            out = self._typeset(val, dim, style, format, step)

        return out
