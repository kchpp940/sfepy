r"""
Thermo-elasticity with a given temperature distribution.

Uses `dw_biot` term with an isotropic coefficient for thermo-elastic coupling.

For given body temperature :math:`T` and background temperature
:math:`T_0` find :math:`\ul{u}` such that:

.. math::
    \int_{\Omega} D_{ijkl}\ e_{ij}(\ul{v}) e_{kl}(\ul{u})
    - \int_{\Omega}  (T - T_0)\ \alpha_{ij} e_{ij}(\ul{v})
    = 0
    \;, \quad \forall \ul{v} \;,

where

.. math::
    D_{ijkl} = \mu (\delta_{ik} \delta_{jl}+\delta_{il} \delta_{jk}) +
    \lambda \ \delta_{ij} \delta_{kl}
    \;, \\

    \alpha_{ij} = (3 \lambda + 2 \mu) \alpha \delta_{ij}

and :math:`\alpha` is the thermal expansion coefficient.

The result export is driven declaratively via the ``export_config`` option:
the derived quantities (strain, stresses, von Mises stress, physical
temperature) are evaluated automatically on save, the output format and file
name come from the same configuration.
"""
import numpy as np

from sfepy.base.base import Struct
from sfepy.mechanics.matcoefs import stiffness_from_lame
from sfepy.mechanics.tensors import get_von_mises_stress
from sfepy.discrete.export_config import DerivedQuantity
from sfepy import data_dir

# Material parameters.
lam = 10.0
mu = 5.0
thermal_expandability = 1.25e-5
T0 = 20.0 # Background temperature.

filename_mesh = data_dir + '/meshes/3d/block.mesh'

def get_temperature_load(ts, coors, region=None, **kwargs):
    """
    Temperature load depends on the `x` coordinate.
    """
    x = coors[:, 0]
    return (x - x.min())**2 - T0

def _von_mises_stress(data, problem, state, extend):
    """Transform the total stress tensor into the von Mises scalar."""
    vms = get_von_mises_stress(data.squeeze())
    return vms.reshape((vms.shape[0], 1, 1, 1))

def _physical_temperature(out, problem, state, extend):
    """Store the temperature variable with the background level added back."""
    val = problem.get_variables()['T']()
    val.shape = (val.shape[0], 1)
    out['T'] = Struct(name='output_data',
                      mode='vertex', data=val + T0,
                      dofs=None)
    return out

def _total_stress(data, problem, state, extend):
    """Sum the elastic and thermal contributions to the total stress."""
    t_stress = problem.evaluate(
        'ev_biot_stress.2.Omega( solid.alpha, T )', mode='el_avg'
    )
    return data + t_stress

options = {
    'nls' : 'newton',
    'ls' : 'ls',
    'export_config': {
        'variable_names': ['u'],
        'derived_quantities': [
            ('cauchy_strain', 'ev_cauchy_strain.2.Omega(u)'),
            ('elastic_stress',
             'ev_cauchy_stress.2.Omega(solid.D, u)'),
            ('thermal_stress',
             'ev_biot_stress.2.Omega(solid.alpha, T)'),
            ('total_stress',
             'ev_cauchy_stress.2.Omega(solid.D, u)',
             {'transform': _total_stress}),
            ('von_mises_stress',
             'ev_cauchy_stress.2.Omega(solid.D, u)',
             {'transform': _von_mises_stress}),
            DerivedQuantity('T', _physical_temperature),
        ],
    },
}

functions = {
    'get_temperature_load' : (get_temperature_load,),
}

regions = {
    'Omega' : 'all',
    'Left' : ('vertices in (x < -4.99)', 'facet'),
}

fields = {
    'displacement': ('real', 3, 'Omega', 1),
    'temperature': ('real', 1, 'Omega', 1),
}

variables = {
    'u' : ('unknown field', 'displacement', 0),
    'v' : ('test field', 'displacement', 'u'),
    'T' : ('parameter field', 'temperature',
           {'setter' : 'get_temperature_load'}),
}

ebcs = {
    'fix_u' : ('Left', {'u.all' : 0.0}),
}

eye_sym = np.array([[1], [1], [1], [0], [0], [0]], dtype=np.float64)
materials = {
    'solid' : ({
        'D' : stiffness_from_lame(3, lam=lam, mu=mu),
        'alpha' : (3.0 * lam + 2.0 * mu) * thermal_expandability * eye_sym
    },),
}

equations = {
    'balance_of_forces' :
    """dw_lin_elastic.2.Omega( solid.D, v, u )
     - dw_biot.2.Omega( solid.alpha, v, T )
     = 0""",
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max'      : 1,
        'eps_a'      : 1e-10,
    }),
}
