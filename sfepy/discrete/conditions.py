"""
The Dirichlet, periodic and linear combination boundary condition
classes, as well as the initial condition class.
"""
import numpy as nm

from sfepy.base.base import Container, Struct, is_sequence
from sfepy.discrete.functions import Function

def get_condition_value(val, functions, kind, name):
    """
    Check a boundary/initial condition value type and return the value or
    corresponding function.
    """
    if type(val) == str:
        if functions is not None:
            try:
                fun = functions[val]

            except IndexError:
                raise ValueError('unknown function %s given for %s %s!'
                                 % (val, kind, name))

        else:
            raise ValueError('no functions given for %s %s!' % (kind, name))

    elif (isinstance(val, Function) or nm.isscalar(val)
          or isinstance(val, nm.ndarray)):
        fun = val

    elif is_sequence(val):
        fun = nm.array(val)

    else:
        raise ValueError('unknown value type for %s %s!'
                         % (kind, name))

    return fun

def _get_region(name, regions, bc_name):
    try:
        region = regions[name]
    except IndexError:
        msg = "no region '%s' used in condition %s!" % (name, bc_name)
        raise IndexError(msg)

    return region

class Conditions(Container):
    """
    Container for various conditions.
    """
    @staticmethod
    def from_conf(conf, regions):
        conds = []
        for key, cc in conf.items():
            times = cc.get('times', None)



            if key.startswith("ebc"):
                region = _get_region(cc.region, regions, cc.name)
                cond = EssentialBC(cc.name, region, cc.dofs, key=key,
                                   times=times)

            elif key.startswith("epbc"):
                rs = [_get_region(ii, regions, cc.name) for ii in cc.region]
                cond = PeriodicBC(cc.name, rs, cc.dofs, cc.match, key=key,
                                   times=times)

            elif key.startswith('lcbc'):
                if isinstance(cc.region, str):
                    rs = [_get_region(cc.region, regions, cc.name), None]

                else:
                    rs = [_get_region(ii, regions, cc.name)
                          for ii in cc.region]

                cond = LinearCombinationBC(cc.name, rs, cc.dofs,
                                           cc.dof_map_fun, cc.kind,
                                           key=key,
                                           times=times,
                                           arguments=cc.get('arguments', None))

            elif key.startswith('dgebc'):
                region = _get_region(cc.region, regions, cc.name)
                cond = DGEssentialBC(cc.name, region, cc.dofs, key=key,
                                     times=times)

            elif key.startswith('dgepbc'):
                rs = [_get_region(ii, regions, cc.name) for ii in cc.region]
                cond = DGPeriodicBC(cc.name, rs, cc.dofs, cc.match, key=key,
                                    times=times)

            elif 'ic' in key:
                region = _get_region(cc.region, regions, cc.name)
                cond = InitialCondition(cc.name, region, cc.dofs, key=key)

            else:
                raise ValueError('unknown condition type! (%s)' % key)

            conds.append(cond)

        obj = Conditions(conds)
        return obj

    def group_by_variables(self, groups=None):
        """
        Group boundary conditions of each variable. Each condition is a
        group is a single condition.

        Parameters
        ----------
        groups : dict, optional
            If present, update the `groups` dictionary.

        Returns
        -------
        out : dict
            The dictionary with variable names as keys and lists of
            single condition instances as values.
        """
        if groups is None:
            out = {}

        else:
            out = groups

        for cond in self:
            for single_cond in cond.iter_single():
                vname = single_cond.dofs[0].split('.')[0]
                out.setdefault(vname, Conditions()).append(single_cond)

        return out

    def canonize_dof_names(self, dofs):
        """
        Canonize the DOF names using the full list of DOFs of a
        variable.
        """
        for cond in self:
            cond.canonize_dof_names(dofs)

    def sort(self):
        """
        Sort boundary conditions by their key.
        """
        self._objs.sort(key=lambda a: a.key)
        self.update()

    def zero_dofs(self):
        """
        Set all boundary condition values to zero, if applicable.
        """
        for cond in self:
            if isinstance(cond, EssentialBC):
                cond.zero_dofs()

def _canonize(dofs, all_dofs):
    """
    Helper function.
    """
    vname, dd = dofs.split('.')

    if dd == 'all':
        cdofs = all_dofs

    elif dd[0] == '[':
        cdofs = [vname + '.' + ii.strip()
                 for ii in dd[1:-1].split(',')]

    else:
        cdofs = [dofs]

    return cdofs

class Condition(Struct):
    """
    Common boundary condition methods.
    """
    def __init__(self, name, **kwargs):
        Struct.__init__(self, name=name, **kwargs)
        self.is_single = False

    def iter_single(self):
        """
        Create a single condition instance for each item in self.dofs
        and yield it.
        """
        for dofs, val in self.dofs.items():
            single_cond = self.copy(name=self.name)
            single_cond.is_single = True
            if 'grad' in dofs:
                # extract variable name from grad.<var-name>.all dofs
                dofs = ".".join((dofs.split(".")[1:]))
                # mark condition as diff
                single_cond.diff = 1

            single_cond.dofs = [dofs, val]
            yield single_cond

    def canonize_dof_names(self, dofs):
        """
        Canonize the DOF names using the full list of DOFs of a
        variable.

        Assumes single condition instance.
        """
        self.dofs[0] = _canonize(self.dofs[0], dofs)

class EssentialBC(Condition):
    """
    Essential boundary condidion.

    Parameters
    ----------
    name : str
        The boundary condition name.
    region : Region instance
        The region where the boundary condition is applied.
    dofs : dict
        The boundary condition specification defining the constrained
        DOFs and their values.
    key : str, optional
        The sorting key.
    times : list or str, optional
        The list of time intervals or a function returning True at time
        steps, when the condition applies.
    """
    def __init__(self, name, region, dofs, key='', times=None):
        Condition.__init__(self, name=name, region=region, dofs=dofs, key=key,
                           times=times)

    def zero_dofs(self):
        """
        Set all essential boundary condition values to zero.
        """
        if self.is_single:
            self.dofs[1] = 0.0

        else:
            new_dofs = {}
            for key in self.dofs.keys():
                new_dofs[key] = 0.0

            self.dofs = new_dofs

class PeriodicBC(Condition):
    """
    Periodic boundary condidion.

    Parameters
    ----------
    name : str
        The boundary condition name.
    regions : list of two Region instances
        The master region and the slave region where the DOFs should match.
    dofs : dict
        The boundary condition specification defining the DOFs in the master
        region and the corresponding DOFs in the slave region.
    match : str
        The name of function for matching corresponding nodes in the
        two regions.
    key : str, optional
        The sorting key.
    times : list or str, optional
        The list of time intervals or a function returning True at time
        steps, when the condition applies.
    """
    def __init__(self, name, regions, dofs, match, key='', times=None):
        Condition.__init__(self, name=name, regions=regions, dofs=dofs,
                           match=match, key=key, times=times)

    def canonize_dof_names(self, dofs):
        """
        Canonize the DOF names using the full list of DOFs of a
        variable.

        Assumes single condition instance.
        """
        self.dofs[0] = _canonize(self.dofs[0], dofs)
        self.dofs[1] = _canonize(self.dofs[1], dofs)

class DGPeriodicBC(PeriodicBC):
    """
    This class is empty, it serves the same purpose
    as PeriodicBC, and is created only for branching in
    dof_info.py
    """
    pass

class DGEssentialBC(EssentialBC):
    """
    This class is empty, it serves the same purpose
    as EssentialBC, and is created only for branching in
    dof_info.py
    """

    def __init__(self, *args, diff=0, **kwargs):
        EssentialBC.__init__(self, *args, **kwargs)
        self.diff = diff

class LinearCombinationBC(Condition):
    """
    Linear combination boundary condidion.

    Parameters
    ----------
    name : str
        The boundary condition name.
    regions : list of two Region instances
        The constrained (master) DOFs region and the new (slave) DOFs
        region. The latter can be None if new DOFs are not field variable DOFs.
    dofs : dict
        The boundary condition specification defining the constrained
        DOFs and the new DOFs (can be None).
    dof_map_fun : str
        The name of function for mapping the constrained DOFs to new DOFs (can
        be None).
    kind : str
        The linear combination condition kind.
    key : str, optional
        The sorting key.
    times : list or str, optional
        The list of time intervals or a function returning True at time
        steps, when the condition applies.
    arguments: tuple, optional
        Additional arguments, depending on the condition kind.
    """
    def __init__(self, name, regions, dofs, dof_map_fun, kind, key='',
                 times=None, arguments=None):
        Condition.__init__(self, name=name, regions=regions, dofs=dofs,
                           dof_map_fun=dof_map_fun, kind=kind,
                           key=key, times=times, arguments=arguments)

    def get_var_names(self):
        """
        Get names of variables corresponding to the constrained and new DOFs.
        """
        names = [self.dofs[0].split('.')[0]]
        if self.dofs[1] is not None:
            names.append(self.dofs[1].split('.')[0])

        return names

    def canonize_dof_names(self, dofs0, dofs1=None):
        """
        Canonize the DOF names using the full list of DOFs of a
        variable.

        Assumes single condition instance.
        """
        self.dofs[0] = _canonize(self.dofs[0], dofs0)

        if self.dofs[1] is not None:
            self.dofs[1] = _canonize(self.dofs[1], dofs1)

class InitialCondition(Condition):
    """
    Initial condidion.

    Parameters
    ----------
    name : str
        The initial condition name.
    region : Region instance
        The region where the initial condition is applied.
    dofs : dict
        The initial condition specification defining the constrained
        DOFs and their values.
    key : str, optional
        The sorting key.
    """
    def __init__(self, name, region, dofs, key=''):
        Condition.__init__(self, name=name, region=region, dofs=dofs, key=key)


class ConstraintPlan(Struct):
    """
    Unified constraint plan for EBC, EPBC, and LCBC conditions.

    This class consolidates the constraint parsing, DOF elimination,
    matrix graph construction, and residual back-filling logic into a
    single plan that drives all downstream operations consistently.

    Attributes
    ----------
    name : str
        The plan name.
    var_name : str
        The variable name this plan applies to.
    ebc_dofs : array
        The EBC-constrained DOF indices.
    ebc_values : array
        The prescribed values for EBC-constrained DOFs.
    epbc_master : array
        The EPBC master DOF indices.
    epbc_slave : array
        The EPBC slave DOF indices.
    lcbc_ops : list
        The LCBC operators.
    eqi : array
        The mapping from full DOF indices to reduced (active) DOF indices.
    eq : array
        The mapping from reduced DOF indices to full DOF indices.
    n_eq : int
        The number of active (reduced) DOFs.
    """

    def __init__(self, name, var_name, n_dof):
        Struct.__init__(self, name=name, var_name=var_name, n_dof=n_dof)

        self.ebc_dofs = nm.array([], dtype=nm.int32)
        self.ebc_values = nm.array([], dtype=nm.float64)
        self.epbc_master = nm.array([], dtype=nm.int32)
        self.epbc_slave = nm.array([], dtype=nm.int32)
        self.lcbc_ops = []

        self.eqi = nm.arange(n_dof, dtype=nm.int32)
        self.eq = nm.arange(n_dof, dtype=nm.int32)
        self.n_eq = n_dof

        self._finalized = False

    def set_ebc(self, dofs, values):
        """
        Set EBC constraints.

        Parameters
        ----------
        dofs : array
            The DOF indices constrained by EBC.
        values : array
            The prescribed values for the constrained DOFs.
        """
        self.ebc_dofs = dofs
        self.ebc_values = values

    def set_epbc(self, master_dofs, slave_dofs):
        """
        Set EPBC constraints.

        Parameters
        ----------
        master_dofs : array
            The master DOF indices.
        slave_dofs : array
            The slave DOF indices.
        """
        self.epbc_master = master_dofs
        self.epbc_slave = slave_dofs

    def add_lcbc(self, op):
        """
        Add an LCBC operator to the plan.

        Parameters
        ----------
        op : LCBCOperator
            The LCBC operator instance.
        """
        self.lcbc_ops.append(op)

    def set_equation_mapping(self, eqi, eq, n_eq):
        """
        Set the full-to-reduced and reduced-to-full equation mappings.

        Parameters
        ----------
        eqi : array
            The mapping from full DOF indices to reduced DOF indices.
        eq : array
            The mapping from reduced DOF indices to full DOF indices.
        n_eq : int
            The number of active (reduced) DOFs.
        """
        self.eqi = eqi
        self.eq = eq
        self.n_eq = n_eq

    def finalize(self):
        """
        Finalize the constraint plan by resolving conflicts between
        EBC, EPBC, and LCBC constraints.

        The resolution priority is: EBC > EPBC > LCBC

        Returns
        -------
        plan : ConstraintPlan
            The finalized constraint plan.
        """
        ebc_set = set(self.ebc_dofs)
        epbc_master_set = set(self.epbc_master)
        epbc_slave_set = set(self.epbc_slave)

        if len(self.ebc_dofs) > 0:
            epbc_mask = nm.ones(len(self.epbc_master), dtype=bool)
            for ii, mdof in enumerate(self.epbc_master):
                if mdof in ebc_set:
                    epbc_mask[ii] = False
                sdof = self.epbc_slave[ii]
                if sdof in ebc_set:
                    epbc_mask[ii] = False

            if not nm.all(epbc_mask):
                self.epbc_master = self.epbc_master[epbc_mask]
                self.epbc_slave = self.epbc_slave[epbc_mask]

        if len(self.ebc_dofs) > 0 or len(self.epbc_master) > 0:
            constrained_set = ebc_set | epbc_master_set | epbc_slave_set

            new_ops = []
            for op in self.lcbc_ops:
                if hasattr(op, 'ameq') and len(op.ameq) > 0:
                    mask = nm.ones(len(op.ameq), dtype=bool)
                    for ii, dof in enumerate(op.ameq):
                        if dof in constrained_set:
                            mask[ii] = False

                    if not nm.all(mask):
                        op.ameq = op.ameq[mask]
                        if hasattr(op, 'aseq') and len(op.aseq) > 0:
                            op.aseq = op.aseq[mask]

                        if hasattr(op, 'mtx') and op.mtx is not None:
                            if hasattr(op.mtx, 'shape') and op.mtx.shape[0] > 0:
                                op.mtx = op.mtx[mask]
                                op.n_mdof = op.mtx.shape[0]

                if len(op.ameq) > 0:
                    new_ops.append(op)

            self.lcbc_ops = new_ops

        self._finalized = True
        return self

    def get_ebc_dofs(self):
        """
        Get EBC-constrained DOF indices.

        Returns
        -------
        dofs : array
            The EBC-constrained DOF indices.
        """
        return self.ebc_dofs

    def get_ebc_values(self):
        """
        Get EBC prescribed values.

        Returns
        -------
        values : array
            The EBC prescribed values.
        """
        return self.ebc_values

    def get_epbc_pairs(self):
        """
        Get EPBC master-slave DOF pairs.

        Returns
        -------
        master : array
            The master DOF indices.
        slave : array
            The slave DOF indices.
        """
        return self.epbc_master, self.epbc_slave

    def get_lcbc_ops(self):
        """
        Get LCBC operators.

        Returns
        -------
        ops : list
            The list of LCBC operators.
        """
        return self.lcbc_ops

    def get_constrained_dofs(self):
        """
        Get all constrained DOF indices (EBC + EPBC master).

        Returns
        -------
        dofs : array
            All constrained DOF indices.
        """
        dofs = nm.concatenate((self.ebc_dofs, self.epbc_master))
        return nm.unique(dofs)

    def has_ebc(self):
        """Check if there are any EBC constraints."""
        return len(self.ebc_dofs) > 0

    def has_epbc(self):
        """Check if there are any EPBC constraints."""
        return len(self.epbc_master) > 0

    def has_lcbc(self):
        """Check if there are any LCBC constraints."""
        return len(self.lcbc_ops) > 0

    def apply_ebc_to_vector(self, vec, offset=0):
        """
        Apply EBC constraints to a vector.

        Parameters
        ----------
        vec : array
            The vector to apply constraints to.
        offset : int
            The offset for the variable in the vector.
        """
        if self.has_ebc():
            ii = offset + self.ebc_dofs
            vec[ii] = self.ebc_values

    def apply_epbc_to_vector(self, vec, offset=0):
        """
        Apply EPBC constraints to a vector (copy slave values to master).

        Parameters
        ----------
        vec : array
            The vector to apply constraints to.
        offset : int
            The offset for the variable in the vector.
        """
        if self.has_epbc():
            master = offset + self.epbc_master
            slave = offset + self.epbc_slave
            vec[master] = vec[slave]

    def apply_to_vector(self, vec, offset=0):
        """
        Apply all constraints to a vector.

        Parameters
        ----------
        vec : array
            The vector to apply constraints to.
        offset : int
            The offset for the variable in the vector.
        """
        self.apply_ebc_to_vector(vec, offset)
        self.apply_epbc_to_vector(vec, offset)

    def get_reduced(self, vec, offset=0, follow_epbc=False):
        """
        Get the reduced DOF vector, with EBC and PBC DOFs removed.

        Parameters
        ----------
        vec : array
            The full DOF vector.
        offset : int
            The offset for the variable in the vector.
        follow_epbc : bool
            If True, values of EPBC master DOFs are added to the
            corresponding slave DOFs.

        Returns
        -------
        r_vec : array
            The reduced DOF vector.
        """
        ii = offset + self.eqi
        r_vec = vec[ii]

        if follow_epbc and self.has_epbc():
            master = offset + self.epbc_master
            slave = self.eq[self.epbc_slave]
            ii = slave >= 0
            if nm.any(ii):
                from sfepy.linalg import assemble1d
                assemble1d(r_vec, slave[ii], vec[master[ii]])

        return r_vec

    def get_full(self, r_vec, r_offset=0, force_value=None,
                 vec=None, offset=0):
        """
        Get the full DOF vector satisfying E(P)BCs from a reduced DOF
        vector.

        Parameters
        ----------
        r_vec : array
            The reduced DOF vector.
        r_offset : int
            The offset for the reduced vector.
        force_value : float, optional
            If given, overrides the EBC values.
        vec : array, optional
            If given, the buffer for storing the result.
        offset : int
            The offset for the variable in the full vector.

        Returns
        -------
        vec : array
            The full DOF vector.
        """
        if vec is None:
            vec = nm.empty(self.n_dof, dtype=r_vec.dtype)

        r_vec = r_vec[r_offset:r_offset+self.n_eq]

        vec[self.eqi] = r_vec

        if force_value is not None:
            vec[self.ebc_dofs] = force_value
        else:
            self.apply_ebc_to_vector(vec)

        self.apply_epbc_to_vector(vec)

        return vec

    def get_ebc_row_indices(self, offset=0):
        """
        Get row indices for EBC constraints in the full matrix.

        Parameters
        ----------
        offset : int
            The offset for the variable in the global matrix.

        Returns
        -------
        ebc_rows : array
            The row indices for EBC constraints.
        """
        if self.has_ebc():
            return offset + self.ebc_dofs
        return nm.array([], dtype=nm.int32)

    def get_epbc_row_indices(self, offset=0):
        """
        Get row indices for EPBC constraints in the full matrix.

        Parameters
        ----------
        offset : int
            The offset for the variable in the global matrix.

        Returns
        -------
        epbc_rows : tuple
            The (master, slave) row indices for EPBC constraints.
        """
        if self.has_epbc():
            return (offset + self.ebc_master, offset + self.ebc_slave)
        return (nm.array([], dtype=nm.int32), nm.array([], dtype=nm.int32))

    def get_lcbc_constrained_dofs(self, offset=0):
        """
        Get DOF indices constrained by LCBC operators.

        Parameters
        ----------
        offset : int
            The offset for the variable in the global matrix.

        Returns
        -------
        lcbc_dofs : array
            The DOF indices constrained by LCBC.
        """
        dofs = []
        for op in self.lcbc_ops:
            if hasattr(op, 'ameq'):
                dofs.extend(op.ameq)
        if len(dofs) > 0:
            return offset + nm.array(dofs, dtype=nm.int32)
        return nm.array([], dtype=nm.int32)

    def has_lcbc_rhs(self):
        """
        Check if any LCBC operator has a right-hand side.

        Returns
        -------
        has_rhs : bool
            True if any LCBC operator has a right-hand side.
        """
        for op in self.lcbc_ops:
            if op.get('rhs', None) is not None:
                return True
        return False

    def get_residual_backfill_order(self):
        """
        Get the order for residual back-filling.

        The order is: LCBC first, then EBC, then EPBC.
        This ensures that constraints are applied in the correct order.

        Returns
        -------
        order : list of tuples
            The order for residual back-filling, each tuple is
            (constraint_type, constraint_data).
        """
        order = []
        if self.has_lcbc():
            order.append(('lcbc', self.lcbc_ops))
        if self.has_ebc():
            order.append(('ebc', (self.ebc_dofs, self.ebc_values)))
        if self.has_epbc():
            order.append(('epbc', (self.ebc_master, self.ebc_slave)))
        return order

    def apply_residual_backfill(self, vec, offset=0, order=None):
        """
        Apply residual back-filling to a vector.

        Parameters
        ----------
        vec : array
            The vector to apply constraints to.
        offset : int
            The offset for the variable in the vector.
        order : list of tuples, optional
            The order for residual back-filling. If not given, the
            default order is used.
        """
        if order is None:
            order = self.get_residual_backfill_order()

        for constraint_type, constraint_data in order:
            if constraint_type == 'ebc':
                ebc_dofs, ebc_values = constraint_data
                ii = offset + ebc_dofs
                vec[ii] = ebc_values
            elif constraint_type == 'epbc':
                ebc_master, ebc_slave = constraint_data
                master = offset + ebc_master
                slave = offset + ebc_slave
                vec[master] = vec[slave]
            elif constraint_type == 'lcbc':
                # LCBC constraints are applied via matrix transformation
                pass

    def get_matrix_application_order(self):
        """
        Get the order for applying constraints to a matrix.

        The order is: LCBC first (matrix transformation), then EBC (row
        replacement), then EPBC (row replacement).

        Returns
        -------
        order : list of tuples
            The order for matrix constraint application, each tuple is
            (constraint_type, constraint_data).
        """
        order = []
        if self.has_lcbc():
            order.append(('lcbc', self.lcbc_ops))
        if self.has_ebc():
            order.append(('ebc', self.ebc_dofs))
        if self.has_epbc():
            order.append(('epbc', (self.ebc_master, self.ebc_slave)))
        return order

    def apply_constraints_to_matrix(self, mtx, offset=0, order=None):
        """
        Apply constraints to a matrix.

        Parameters
        ----------
        mtx : sparse matrix
            The matrix to apply constraints to.
        offset : int
            The offset for the variable in the matrix.
        order : list of tuples, optional
            The order for matrix constraint application. If not given,
            the default order is used.
        """
        from sfepy.discrete.evaluate import apply_ebc_to_matrix

        if order is None:
            order = self.get_matrix_application_order()

        for constraint_type, constraint_data in order:
            if constraint_type == 'ebc':
                ebc_rows = offset + constraint_data
                apply_ebc_to_matrix(mtx, ebc_rows)
            elif constraint_type == 'epbc':
                ebc_master, ebc_slave = constraint_data
                epbc_rows = (offset + ebc_master, offset + ebc_slave)
                apply_ebc_to_matrix(mtx, [], epbc_rows)
            elif constraint_type == 'lcbc':
                # LCBC constraints are applied via matrix transformation
                pass

    def get_lcbc_input(self):
        """
        Get input data for LCBC operator construction.

        Returns
        -------
        lcbc_input : dict
            The input data for LCBC operator construction, containing:
            - 'ebc_dofs': EBC-constrained DOF indices
            - 'epbc_master': EPBC master DOF indices
            - 'epbc_slave': EPBC slave DOF indices
            - 'eqi': Full-to-reduced DOF mapping
            - 'eq': Reduced-to-full DOF mapping
            - 'n_eq': Number of active (reduced) DOFs
        """
        return {
            'ebc_dofs': self.ebc_dofs,
            'epbc_master': self.epbc_master,
            'epbc_slave': self.epbc_slave,
            'eqi': self.eqi,
            'eq': self.eq,
            'n_eq': self.n_eq,
            'has_ebc': self.has_ebc(),
            'has_epbc': self.has_epbc(),
            'has_lcbc': self.has_lcbc(),
        }

    def get_constraint_summary(self):
        """
        Get a summary of all constraints.

        Returns
        -------
        summary : dict
            The constraint summary.
        """
        return {
            'var_name': self.var_name,
            'n_dof': self.n_dof,
            'n_eq': self.n_eq,
            'n_ebc': len(self.ebc_dofs),
            'n_epbc': len(self.epbc_master),
            'n_lcbc': len(self.lcbc_ops),
            'has_ebc': self.has_ebc(),
            'has_epbc': self.has_epbc(),
            'has_lcbc': self.has_lcbc(),
        }
