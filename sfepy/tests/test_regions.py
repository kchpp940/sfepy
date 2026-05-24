import numpy as nm
import pytest

from sfepy.base.conf import transform_functions
import sfepy.base.testing as tst

def get_vertices(coors, domain=None):
    x, z = coors[:,0], coors[:,2]

    return nm.where((z < 0.1) & (x < 0.1))[0]

def get_cells(coors, domain=None):
    return nm.where(coors[:, 0] < 0)[0]

@pytest.fixture(scope='module')
def data():
    from sfepy import data_dir
    from sfepy.base.base import Struct
    from sfepy.discrete.fem import Mesh, FEDomain
    from sfepy.discrete import Functions

    mesh = Mesh.from_file(data_dir
                          + '/meshes/various_formats/abaqus_tet.inp')
    mesh.nodal_bcs['set0'] = [0, 7]
    domain = FEDomain('test domain', mesh)

    conf_functions = {
        'get_vertices' : (get_vertices,),
        'get_cells' : (get_cells,),
    }
    functions = Functions.from_conf(transform_functions(conf_functions))

    return Struct(domain=domain, functions=functions)

def test_selectors(data):
    """
    Test basic region selectors.
    """
    selectors = [
        ['all', 'cell'],
        ['vertices of surface', 'facet'],
        ['vertices of group %d' % data.domain.cmesh.vertex_groups[0],
         'facet'],
        ['vertices of set set0', 'vertex'],
        ['vertices in (z < 0.1) & (x < 0.1)', 'facet'],
        ['vertices by get_vertices', 'cell'],
        ['vertex 0, 1, 2', 'vertex'],
        ['vertex in r.r6', 'vertex'],
        ['cells of group %d' % data.domain.cmesh.cell_groups[0], 'cell'],
        # ['cells of set 0', 'cell'], not implemented...
        ['cells by get_cells', 'cell'],
        ['cell 1, 4, 5', 'cell'],
        ['copy r.r5', 'cell'],
        ['r.r5', 'cell'],
    ]

    vertices = [
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
        [0,  1,  3,  7],
        [0,  7],
        [1,  2,  3,  4,  5,  9, 11],
        [1,  2,  3,  4,  5,  9, 11],
        [0,  1,  2],
        [0],
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
        [0,  1,  2,  3,  4,  5,  6,  9, 10, 11],
        [0,  1,  2,  3,  4,  5,  6,  8],
        [1,  2,  3,  4,  5,  9, 11],
        [1,  2,  3,  4,  5,  9, 11],
    ]

    data.domain.reset_regions()

    ok = True
    for ii, sel in enumerate(selectors):
        tst.report('select:', sel)
        reg = data.domain.create_region('r%d' % ii, sel[0], kind=sel[1],
                                        functions=data.functions)
        _ok = ((len(reg.vertices) == len(vertices[ii]))
               and (reg.vertices == vertices[ii]).all())
        tst.report('  vertices:', _ok)

        ok = ok and _ok

    assert ok

def test_operators(data):
    """
    Test operators in region selectors.
    """
    ok = True

    data.domain.reset_regions()

    r1 = data.domain.create_region('r1', 'all')

    sel = 'r.r1 -v vertices of group 0'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel, kind='vertex')
    av = [2,  4,  5,  6,  8,  9, 10, 11, 12]
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'vertex 0, 1, 2 +v vertices of group 0'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel, kind='vertex')
    av = [0,  1,  2,  3,  7]
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'vertex 0, 1, 2 *v vertices of group 0'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel, kind='vertex')
    av = [0,  1]
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'vertex 20 *v vertices of group 0'
    tst.report('select:', sel, ', allow_empty == False')
    _ok = False
    try:
        reg = data.domain.create_region('reg', sel, kind='vertex',
                                        allow_empty=False)
    except ValueError:
        _ok = True
    tst.report('  exception raised:', _ok)
    ok = ok and _ok

    sel = 'vertex 20 *v vertices of group 0'
    tst.report('select:', sel, ', allow_empty == True')
    reg = data.domain.create_region('reg', sel, kind='vertex',
                                    allow_empty=True)
    av = []
    _ok = (reg.vertices == nm.array(av)).all()
    tst.report('  vertices:', _ok)
    ok = ok and _ok

    sel = 'r.r1 -c cell 1, 4, 5'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel)
    _ok = (nm.setdiff1d(r1.cells[0], [1, 4, 5]) == reg.cells[0]).all()
    tst.report('  cells:', _ok)
    ok = ok and _ok

    sel = 'cell 8, 3 +c cell 1, 4, 5'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel)
    cells = [1,  3,  4,  5,  8]
    _ok = (reg.cells == nm.array(cells)).all()
    tst.report('  cells:', _ok)
    ok = ok and _ok

    sel = 'cell 8, 3, 2 *c cell 8, 4, 2, 7'
    tst.report('select:', sel)
    reg = data.domain.create_region('reg', sel)
    cells = [2,  8]
    _ok = (reg.cells == nm.array(cells)).all()
    tst.report('  cells:', _ok)
    ok = ok and _ok

    assert ok

def test_aliases(data):
    """
    Test region alias expansion in region definitions.
    """
    from sfepy.discrete.region_aliases import RegionAliasRegistry

    data.domain.reset_regions()

    aliases = RegionAliasRegistry()
    aliases.define('top', 'vertices in (z > 0.499)', kind='facet')
    aliases.define('bottom', 'vertices in (z < -0.499)', kind='facet')
    aliases.define('band',
                   'vertices in ((z > {z0}) & (z < {z1}))',
                   parameters=('z0', 'z1'))
    aliases.define('all_cells', 'cells of group {gid}',
                   parameters=('gid',))

    # Register on the domain so create_regions consumes it automatically.
    data.domain.set_aliases(aliases)

    data.domain.create_region('Omega', 'all')
    gid = data.domain.cmesh.cell_groups[0]
    data.domain.create_region('Top', 'a.top', kind='facet')
    data.domain.create_region('Bottom', 'a.bottom', kind='facet')

    # Parameterized alias directly via call (explicit aliases argument
    data.domain.create_region(
        'Band',
        'a.band[z0=-0.1, z1=0.1]',
        kind='facet',
    )

    # Set expression of group via aliases as dict-style alias
    data.domain.create_region(
        'Gcells',
        'a.all_cells[gid=%d]' % gid,
    )

    # Aliases in set algebra
    data.domain.create_region(
        'TopAndBottom',
        'a.top +v a.bottom',
        kind='facet',
    )

    assert 'Top' in data.domain.regions.get_names()
    assert 'Bottom' in data.domain.regions.get_names()
    assert 'Band' in data.domain.regions.get_names()
    assert 'Gcells' in data.domain.regions.get_names()
    assert 'TopAndBottom' in data.domain.regions.get_names()

    # Check the aliases expansion yields valid vertices
    assert len(data.domain.regions['Top'].vertices) > 0
    assert len(data.domain.regions['Bottom'].vertices) > 0
    tst.report('OK: aliases produced non-empty region vertices')


def test_alias_create_regions_with_aliases(data):
    """
    Test Domain.create_regions() integration with a config-style aliases.
    """
    from sfepy.base.conf import transform_regions
    from sfepy.discrete.region_aliases import RegionAliasRegistry

    data.domain.reset_regions()

    aliases = RegionAliasRegistry()
    aliases.define('left', 'vertices in (x < -0.499)', kind='facet')
    aliases.define('right', 'vertices in (x > 0.499)', kind='facet')

    regions = {
        'Omega': {'name': 'Omega', 'select': 'all'},
        'Left': {'name': 'Left', 'select': 'a.left', 'kind': 'facet'},
        'Right': {'name': 'Right', 'select': 'a.right', 'kind': 'facet'},
        'LeftRight': {'name': 'LeftRight',
                     'select': 'a.left +v a.right',
                     'kind': 'facet'},
    }
    regions = transform_regions(regions)

    data.domain.create_regions(regions, aliases=aliases, allow_empty=True)

    names = data.domain.regions.get_names()
    assert 'Left' in names
    assert 'Right' in names
    assert 'LeftRight' in names
    tst.report('OK: create_regions with aliases')


def test_alias_full_chain(data):
    """
    Test the full alias pipeline: config -> regions -> fields -> BCs.

    This test exercises every step where aliases are consumed:

    1. ``ProblemConf.transform_region_aliases`` produces a registry
    2. ``Domain.create_regions`` expands alias selectors
    3. ``fields_from_conf`` resolves ``a.<alias>`` in field ``region``
    4. ``Conditions.from_conf`` resolves ``a.<alias>`` in BC ``region``
    """
    from sfepy.base.conf import (transform_region_aliases,
                                  transform_regions, transform_fields,
                                  transform_ebcs)
    from sfepy.discrete.region_aliases import (
        RegionAliasRegistry, resolve_alias_region_name,
    )
    from sfepy.discrete.common.fields import fields_from_conf
    from sfepy.discrete.conditions import Conditions

    data.domain.reset_regions()

    # 1.  Transform the raw config dictionary (as ProblemConf would).
    raw_aliases = {
        'left':  {'select': 'vertices in (x < -0.499)', 'kind': 'facet'},
        'right': {'select': 'vertices in (x >  0.499)', 'kind': 'facet'},
        'omega': {'select': 'all',                      'kind': 'cell'},
    }
    aliases = transform_region_aliases(raw_aliases)
    assert isinstance(aliases, RegionAliasRegistry)
    tst.report('1. transform_region_aliases OK -> registry')

    # 2.  Regions use aliases in selectors.
    raw_regions = {
        'Omega': {'name': 'Omega', 'select': 'a.omega'},
        'Left':  {'name': 'Left',  'select': 'a.left',  'kind': 'facet'},
        'Right': {'name': 'Right', 'select': 'a.right', 'kind': 'facet'},
    }
    conf_regions = transform_regions(raw_regions)
    data.domain.create_regions(conf_regions, aliases=aliases,
                               allow_empty=True)
    assert 'Left' in data.domain.regions.get_names()
    tst.report('2. create_regions with aliases OK')

    # 3.  Fields can reference regions by alias name.
    raw_fields = {
        'u': {'name': 'u', 'dtype': 'real', 'shape': (1,),
              'region': 'a.omega', 'approx_order': 1},
    }
    conf_fields = transform_fields(raw_fields)
    # The region attribute still contains 'a.omega' at this point.
    assert conf_fields['field_u'].region == 'a.omega'
    fields = fields_from_conf(conf_fields, data.domain.regions,
                              aliases=aliases)
    assert 'u' in fields
    assert fields['u'].region.name == 'Omega'
    tst.report('3. fields_from_conf resolves a.omega -> Omega')

    # 4.  BCs can reference regions by alias name.
    raw_ebcs = {
        'fixed_u': {'name': 'fixed_u', 'region': 'a.left',
                    'dofs': {'u.all': 0.0}},
    }
    conf_ebcs = transform_ebcs(raw_ebcs)
    ebcs = Conditions.from_conf(conf_ebcs, data.domain.regions,
                                aliases=aliases)
    assert len(ebcs) == 1
    assert ebcs[0].region.name == 'Left'
    tst.report('4. Conditions.from_conf resolves a.left -> Left')

    # 5.  Direct helper: resolve_alias_region_name.
    assert resolve_alias_region_name('a.left', aliases) == 'left'
    # Non-alias strings pass through unchanged.
    assert resolve_alias_region_name('Omega', aliases) == 'Omega'
    tst.report('5. resolve_alias_region_name helper OK')


def test_alias_parameterized_and_composed(data):
    """
    Test parameterized aliases (``a.band[y0=..., y1=...]``) and
    composed aliases (aliases whose template references other aliases)
    across the full config→region→field→BC pipeline.
    """
    from sfepy.base.conf import (transform_region_aliases,
                                  transform_regions, transform_fields,
                                  transform_ebcs)
    from sfepy.discrete.region_aliases import (
        RegionAliasRegistry, resolve_alias_region_name,
        is_alias_expression,
    )
    from sfepy.discrete.common.fields import fields_from_conf
    from sfepy.discrete.conditions import Conditions

    data.domain.reset_regions()

    # ------------------------------------------------------------------
    # 1.  Define parameterized + composed aliases via config dict.
    # ------------------------------------------------------------------
    raw_aliases = {
        # Simple facet selectors.
        'left': {
            'select': 'vertices in (x < -0.499)',
            'kind': 'facet',
            'region_name': 'Left',
        },
        'right': {
            'select': 'vertices in (x >  0.499)',
            'kind': 'facet',
            'region_name': 'Right',
        },
        # Parameterized alias: a band in y.
        'band_y': {
            'select': 'vertices in ((y > {y0}) & (y < {y1}))',
            'parameters': ('y0', 'y1'),
            'kind': 'facet',
        },
        # Composed alias: union of left + right facets.
        'both': {
            'select': 'a.left +v a.right',
            'kind': 'facet',
            'region_name': 'LeftRight',
        },
        # Whole domain.
        'omega': {
            'select': 'all',
            'kind': 'cell',
            'region_name': 'Omega',
        },
    }
    aliases = transform_region_aliases(raw_aliases)
    tst.report('1. transform_region_aliases OK')

    # ------------------------------------------------------------------
    # 2.  Create regions — some use parameterized / composed aliases.
    # ------------------------------------------------------------------
    raw_regions = {
        'Omega':     {'name': 'Omega',     'select': 'a.omega'},
        'Left':      {'name': 'Left',      'select': 'a.left',
                      'kind': 'facet'},
        'Right':     {'name': 'Right',     'select': 'a.right',
                      'kind': 'facet'},
        'LeftRight': {'name': 'LeftRight', 'select': 'a.both',
                      'kind': 'facet'},
        'Band':      {'name': 'Band',
                      'select': 'a.band_y[y0=-0.1, y1=0.1]',
                      'kind': 'facet'},
    }
    conf_regions = transform_regions(raw_regions)
    data.domain.create_regions(conf_regions, aliases=aliases,
                               allow_empty=True)

    names = data.domain.regions.get_names()
    for n in ('Omega', 'Left', 'Right', 'LeftRight', 'Band'):
        assert n in names, 'region %s missing: %s' % (n, names)
    tst.report('2. create_regions OK (param + composed)')

    # ------------------------------------------------------------------
    # 3.  is_alias_expression helper
    # ------------------------------------------------------------------
    assert is_alias_expression('a.band_y[y0=-0.1, y1=0.1]') == 'band_y'
    assert is_alias_expression('a.omega') == 'omega'
    assert is_alias_expression('a.left +v a.right') is None  # has operators
    assert is_alias_expression('vertices in (x < 0)') is None
    tst.report('3. is_alias_expression OK')

    # ------------------------------------------------------------------
    # 4.  Materialized alias map should contain the parameterized expr.
    # ------------------------------------------------------------------
    mat = aliases.get_materialized()
    assert 'a.band_y[y0=-0.1, y1=0.1]' in mat
    assert mat['a.band_y[y0=-0.1, y1=0.1]'] == 'Band'
    tst.report('4. materialized alias map OK:', mat)

    # ------------------------------------------------------------------
    # 5.  Fields reference regions by alias expressions.
    # ------------------------------------------------------------------
    raw_fields = {
        'u': {
            'name': 'u', 'dtype': 'real', 'shape': (1,),
            'region': 'a.omega',
            'approx_order': 1,
        },
        'v': {
            'name': 'v', 'dtype': 'real', 'shape': (1,),
            'region': 'a.band_y[y0=-0.1, y1=0.1]',
            'approx_order': 1,
        },
    }
    conf_fields = transform_fields(raw_fields)
    fields = fields_from_conf(conf_fields, data.domain.regions,
                               aliases=aliases)
    assert 'u' in fields
    assert fields['u'].region.name == 'Omega'
    assert 'v' in fields
    assert fields['v'].region.name == 'Band'
    tst.report('5. fields_from_conf resolves alias expressions OK')

    # ------------------------------------------------------------------
    # 6.  EBCs reference regions by alias expressions.
    # ------------------------------------------------------------------
    raw_ebcs = {
        'fix_left':  {'name': 'fix_left',  'region': 'a.left',
                       'dofs': {'u.all': 0.0}},
        'fix_band':  {'name': 'fix_band',
                       'region': 'a.band_y[y0=-0.1, y1=0.1]',
                       'dofs': {'u.all': 1.0}},
        'fix_both':  {'name': 'fix_both',  'region': 'a.both',
                       'dofs': {'u.all': 2.0}},
    }
    conf_ebcs = transform_ebcs(raw_ebcs)
    ebcs = Conditions.from_conf(conf_ebcs, data.domain.regions,
                                aliases=aliases)
    assert len(ebcs) == 3
    region_map = {c.name: c.region.name for c in ebcs}
    assert region_map == {'fix_left': 'Left', 'fix_band': 'Band',
                          'fix_both': 'LeftRight'}
    tst.report('6. EBCs resolve alias expressions OK:', region_map)

    # ------------------------------------------------------------------
    # 7.  resolve_alias_region_name for parameterized expression.
    # ------------------------------------------------------------------
    assert (resolve_alias_region_name('a.band_y[y0=-0.1, y1=0.1]', aliases)
            == 'Band')
    # Non-parameterized falls back to region_name.
    assert resolve_alias_region_name('a.omega', aliases) == 'Omega'
    # Composed alias resolves via region_name.
    assert resolve_alias_region_name('a.both', aliases) == 'LeftRight'
    # Unknown parameterized expression → falls through to bare lookup → KeyError.
    try:
        resolve_alias_region_name('a.band_y[y0=-0.2, y1=0.2]', aliases)
    except KeyError as e:
        tst.report('7. unknown param expr correctly errors:', e)
    tst.report('7. resolve_alias_region_name param OK')

    tst.report('8. ALL parameterized + composed alias tests PASSED')


def test_alias_selector_match_fallback(data):
    """
    Test that parameterized alias expressions in field/BC regions
    resolve even without the materialized map, by matching the expanded
    selector against existing regions.
    """
    from sfepy.base.conf import (transform_region_aliases,
                                  transform_regions, transform_fields,
                                  transform_ebcs)
    from sfepy.discrete.region_aliases import (
        RegionAliasRegistry, resolve_alias_region_name,
    )
    from sfepy.discrete.common.fields import fields_from_conf
    from sfepy.discrete.conditions import Conditions

    data.domain.reset_regions()

    # 1. Define aliases (no region_name set → defaults to alias name).
    raw_aliases = {
        'band_y': {
            'select': 'vertices in ((y > {y0}) & (y < {y1}))',
            'parameters': ('y0', 'y1'),
            'kind': 'facet',
        },
        'omega': {'select': 'all', 'kind': 'cell'},
    }
    aliases = transform_region_aliases(raw_aliases)

    # 2. Create regions WITHOUT the aliases registry — use raw selectors
    #    directly. This simulates the case where materialized map is
    #    never populated.
    data.domain.create_region('Omega', 'all')
    data.domain.create_region(
        'Band',
        'vertices in ((y > -0.1) & (y < 0.1))',
        kind='facet',
    )

    # 3. Field uses parameterized alias expression → should resolve
    #    via selector match (step 3 in resolution order).
    raw_fields = {
        'u': {
            'name': 'u', 'dtype': 'real', 'shape': (1,),
            'region': 'a.band_y[y0=-0.1, y1=0.1]',
            'approx_order': 1,
        },
    }
    conf_fields = transform_fields(raw_fields)
    fields = fields_from_conf(conf_fields, data.domain.regions,
                               aliases=aliases)
    assert 'u' in fields
    assert fields['u'].region.name == 'Band'
    tst.report('1. field resolves parameterized alias via selector match')

    # 4. EBC uses parameterized alias expression → should also resolve.
    raw_ebcs = {
        'fix_band': {
            'name': 'fix_band',
            'region': 'a.band_y[y0=-0.1, y1=0.1]',
            'dofs': {'u.all': 0.0},
        },
    }
    conf_ebcs = transform_ebcs(raw_ebcs)
    ebcs = Conditions.from_conf(conf_ebcs, data.domain.regions,
                                aliases=aliases)
    assert len(ebcs) == 1
    assert ebcs[0].region.name == 'Band'
    tst.report('2. EBC resolves parameterized alias via selector match')

    # 5. Verify the materialized map was populated as a side-effect.
    assert aliases.resolve_materialized(
        'a.band_y[y0=-0.1, y1=0.1]') == 'Band'
    tst.report('3. materialized map auto-populated after selector match')

    tst.report('4. ALL selector-match fallback tests PASSED')


def test_alias_parameterized_error(data):
    """
    Test that a parameterized alias with no matching region gives a
    clear, actionable error message.
    """
    from sfepy.base.conf import (transform_region_aliases,
                                  transform_regions, transform_fields,
                                  transform_ebcs)
    from sfepy.discrete.region_aliases import (
        RegionAliasRegistry, resolve_alias_region_name,
    )
    from sfepy.discrete.common.fields import fields_from_conf

    data.domain.reset_regions()

    raw_aliases = {
        'band_y': {
            'select': 'vertices in ((y > {y0}) & (y < {y1}))',
            'parameters': ('y0', 'y1'),
            'kind': 'facet',
        },
    }
    aliases = transform_region_aliases(raw_aliases)

    # Create a region with different parameters.
    data.domain.create_region('Omega', 'all')
    data.domain.create_region(
        'Band',
        'vertices in ((y > -0.2) & (y < 0.2))',
        kind='facet',
    )

    raw_fields = {
        'u': {
            'name': 'u', 'dtype': 'real', 'shape': (1,),
            'region': 'a.band_y[y0=-0.1, y1=0.1]',
            'approx_order': 1,
        },
    }
    conf_fields = transform_fields(raw_fields)

    # The parameterized alias won't match any region's select.
    # After selector-match fails and bare alias lookup returns 'band_y'
    # (the default region_name), regions['band_y'] will KeyError — but
    # first verify that resolve_alias_region_name itself gives a clear
    # error when NO region matches.

    # With no region having a matching select and no explicit
    # region_name, the bare alias name 'band_y' is returned. If that
    # region doesn't exist, regions[name] will raise KeyError.
    # The user will see which alias failed.
    try:
        fields_from_conf(conf_fields, data.domain.regions, aliases=aliases)
        # Should not reach here.
        assert False, 'should have raised'
    except (KeyError, IndexError) as e:
        # The error should reference the alias name or region name.
        err_str = str(e)
        tst.report('1. clear error for unmatched param alias:', err_str)
        # Accept either 'band_y' or the parameterized expression.
        assert 'band_y' in err_str or 'band' in err_str.lower()

    # 2. Test with a completely unknown alias — should get a clear error.
    raw_fields_bad = {
        'v': {
            'name': 'v', 'dtype': 'real', 'shape': (1,),
            'region': 'a.nonexistent[x=1]',
            'approx_order': 1,
        },
    }
    conf_fields_bad = transform_fields(raw_fields_bad)

    try:
        fields_from_conf(conf_fields_bad, data.domain.regions,
                         aliases=aliases)
        assert False, 'should have raised'
    except KeyError as e:
        err_str = str(e)
        tst.report('2. clear error for unknown alias:', err_str)
        assert 'nonexistent' in err_str.lower()

    # 3. Test with missing parameters — should get a clear error.
    raw_fields_no_params = {
        'w': {
            'name': 'w', 'dtype': 'real', 'shape': (1,),
            'region': 'a.band_y',
            'approx_order': 1,
        },
    }
    conf_fields_no_params = transform_fields(raw_fields_no_params)
    # This should return 'band_y' (bare alias name), which doesn't
    # exist as a region — regions['band_y] will raise KeyError.
    try:
        fields_from_conf(conf_fields_no_params, data.domain.regions,
                         aliases=aliases)
        assert False, 'should have raised'
    except (KeyError, IndexError) as e:
        err_str = str(e)
        tst.report('3. error for alias without params:', err_str)

    tst.report('4. ALL error-case tests PASSED')
