"""
Region alias registry and expansion utilities.

Region aliases allow users to define reusable, parameterized selectors
that can be referenced from region definitions, boundary conditions, and
other configuration points using the ``a.<name>`` syntax.

Examples
--------
>>> from sfepy.discrete.region_aliases import RegionAliasRegistry
>>> aliases = RegionAliasRegistry()
>>> aliases.define('left', 'vertices in (x < -0.499)')
>>> aliases.define('right', 'vertices in (x > 0.499)')
>>> aliases.define('band',
...     'vertices in ((y > {y0}) & (y < {y1}))',
...     parameters=('y0', 'y1'))
>>> aliases.expand('a.left +v a.right')
'(vertices in (x < -0.499)) +v (vertices in (x > 0.499))'
>>> aliases.expand('a.band[y0=-0.1, y1=0.1]')
'(vertices in ((y > -0.1) & (y < 0.1)))'
"""

import re
from collections import OrderedDict

_alias_ref = re.compile(
    r'a\.([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]*)\])?'
)


class RegionAliasRegistry(object):
    """
    Registry of reusable region selector aliases.

    Each alias has:

    - a *name*: identifier used in ``a.<name>`` references;
    - a *template*: a region selector string (or callable returning one)
      that may contain ``{param}`` placeholders;
    - an optional *parameters* tuple listing the placeholder names
      expected by the template;
    - an optional *kind* (``'cell'``, ``'facet'``, ...) that downstream
      consumers (e.g. :func:`Domain.create_regions`) can use when the
      alias is the entire selector.
    """

    def __init__(self, aliases=None):
        self._aliases = OrderedDict()
        self._materialized = OrderedDict()
        if aliases is not None:
            self.update(aliases)

    # -- dict-like interface --------------------------------------------------

    def __contains__(self, name):
        return name in self._aliases

    def __iter__(self):
        return iter(self._aliases)

    def __len__(self):
        return len(self._aliases)

    def __getitem__(self, name):
        return self._aliases[name]

    def keys(self):
        return self._aliases.keys()

    def values(self):
        return self._aliases.values()

    def items(self):
        return self._aliases.items()

    def get(self, name, default=None):
        return self._aliases.get(name, default)

    # -- definition API -------------------------------------------------------

    def define(self, name, template, parameters=(), kind=None, parent=None,
               region_name=None):
        """
        Register an alias.

        Parameters
        ----------
        name : str
            Alias name.  Must be a valid Python identifier.
        template : str or callable
            Either a selector string with optional ``{name}`` placeholders,
            or a callable ``(**kwargs) -> str`` that produces the selector.
        parameters : tuple of str, optional
            Names of the expected placeholder / keyword arguments.
        kind : str, optional
            Suggested region kind (``'cell'``, ``'facet'``, ...).
        parent : str, optional
            Suggested parent region name.
        region_name : str, optional
            The region name that this alias will be used to create.  When
            given, downstream code (fields, boundary conditions) can
            reference the alias via ``a.<name>`` and resolve it to this
            region name.  Defaults to *name* itself.
        """
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            raise ValueError('invalid alias name: %r' % name)
        if name in self._aliases:
            raise ValueError('region alias %r already defined' % name)
        if region_name is None:
            region_name = name
        self._aliases[name] = {
            'name': name,
            'template': template,
            'parameters': tuple(parameters),
            'kind': kind,
            'parent': parent,
            'region_name': region_name,
        }
        return self

    def update(self, aliases):
        """
        Bulk-register aliases.

        Accepts either a ``dict`` of ``{name: template}``, a list of
        tuples, or another :class:`RegionAliasRegistry`.
        """
        if isinstance(aliases, RegionAliasRegistry):
            for name, spec in aliases.items():
                self._aliases[name] = dict(spec)
        elif isinstance(aliases, dict):
            for name, spec in aliases.items():
                if isinstance(spec, str):
                    self.define(name, spec)
                elif isinstance(spec, dict):
                    self.define(name,
                                spec.get('template', spec.get('select')),
                                parameters=spec.get('parameters', ()),
                                kind=spec.get('kind'),
                                parent=spec.get('parent'),
                                region_name=spec.get('region_name'))
                else:
                    raise ValueError('bad alias spec for %s: %r'
                                     % (name, spec))
        else:
            for entry in aliases:
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    self.define(entry[0], entry[1],
                                parameters=entry[2]
                                if len(entry) > 2 else (),
                                kind=entry[3] if len(entry) > 3 else None,
                                region_name=entry[4]
                                if len(entry) > 4 else None)
                else:
                    raise ValueError('bad alias entry: %r' % (entry,))
        return self

    def remove(self, name):
        """Remove a previously defined alias."""
        if name not in self._aliases:
            raise KeyError(name)
        del self._aliases[name]

    def clear(self):
        """Remove all aliases and materialized entries."""
        self._aliases.clear()
        self._materialized.clear()

    # -- materialized alias tracking ------------------------------------------

    def record_materialized(self, alias_expr, region_name):
        """
        Record that the alias expression *alias_expr* (e.g.
        ``'a.band[y0=-0.1, y1=0.1]'``) was used to create the region
        with name *region_name*.

        This enables downstream consumers (fields, boundary conditions)
        to reference the region using the same alias expression.
        """
        self._materialized[alias_expr] = region_name

    def resolve_materialized(self, alias_expr):
        """
        Look up the region name that was recorded for the given alias
        expression.  Returns ``None`` if not found.
        """
        return self._materialized.get(alias_expr)

    def get_materialized(self):
        """Return a copy of the materialized-alias mapping."""
        return dict(self._materialized)

    # -- introspection --------------------------------------------------------

    def get_names(self):
        return list(self._aliases.keys())

    def get_template(self, name):
        return self._aliases[name]['template']

    def get_parameters(self, name):
        return self._aliases[name]['parameters']

    def get_kind(self, name):
        return self._aliases[name].get('kind')

    def get_parent(self, name):
        return self._aliases[name].get('parent')

    def get_region_name(self, name):
        """Return the region name that alias *name* maps to."""
        return self._aliases[name].get('region_name', name)

    # -- expansion ------------------------------------------------------------

    def instantiate(self, name, kwargs=None):
        """
        Render the template of alias *name* with the given keyword
        arguments.  Returns a plain region selector string.
        """
        spec = self._aliases.get(name)
        if spec is None:
            raise KeyError('unknown region alias: %r' % name)
        template = spec['template']
        params = spec['parameters'] or ()
        if kwargs is None:
            kwargs = {}
        missing = [p for p in params if p not in kwargs]
        if missing:
            raise ValueError(
                'region alias %r requires parameters: %s'
                % (name, ', '.join(missing)))
        if callable(template):
            return str(template(**kwargs))
        if params:
            return template.format(**kwargs)
        return template

    def expand(self, selector, seen=None):
        """
        Replace every ``a.<name>`` (or ``a.<name>[k=v, ...]``)
        reference in *selector* with its fully expanded form.

        Expansion recurses until the selector is free of alias
        references.  Unknown aliases raise :class:`KeyError`.
        """
        if seen is None:
            seen = set()
        changed = True
        current = selector
        while changed:
            changed = False
            matches = list(_alias_ref.finditer(current))
            if not matches:
                break
            parts = []
            cursor = 0
            for m in matches:
                name = m.group(1)
                argstr = m.group(2)
                if name in seen:
                    raise ValueError(
                        'circular region alias reference involving %r'
                        % name)
                kwargs = {}
                if argstr:
                    for piece in _split_args(argstr):
                        k, v = piece.split('=', 1)
                        kwargs[k.strip()] = _coerce(v.strip())
                expanded = self.instantiate(name, kwargs)
                if _alias_ref.search(expanded):
                    expanded = self.expand(expanded, seen | {name})
                parts.append(current[cursor:m.start()])
                parts.append('(%s)' % expanded)
                cursor = m.end()
                changed = True
            parts.append(current[cursor:])
            current = ''.join(parts)
        return current

    # -- dependency helpers ---------------------------------------------------

    def referenced_by(self, selector):
        """
        Return the list of alias names referenced by *selector*
        (in order of first appearance, with duplicates removed).
        """
        seen = OrderedDict()
        for m in _alias_ref.finditer(selector):
            seen.setdefault(m.group(1), None)
        return list(seen.keys())


def _split_args(argstr):
    """Split ``k=v, k=v`` allowing balanced parentheses/brackets in values."""
    parts = []
    depth = 0
    buf = []
    for ch in argstr:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf))
    return parts


def _coerce(val):
    """Try to convert a string argument to int/float; fallback to str."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def find_region_aliases(selector):
    """
    Return alias names referenced in *selector*, preserving order and
    uniqueness.  This is a module-level helper so that the dependency
    analyzer in :mod:`sfepy.discrete.common.region` can use it without
    requiring a registry instance.
    """
    seen = OrderedDict()
    for m in _alias_ref.finditer(selector):
        seen.setdefault(m.group(1), None)
    return list(seen.keys())


def expand_region_aliases(selector, aliases):
    """
    Module-level wrapper around :meth:`RegionAliasRegistry.expand` that
    gracefully handles ``None`` / empty registries.
    """
    if aliases is None:
        return selector
    if isinstance(aliases, RegionAliasRegistry):
        return aliases.expand(selector)
    # plain dict fallback
    reg = RegionAliasRegistry(aliases)
    return reg.expand(selector)


def resolve_alias_region_name(name, aliases, regions=None):
    """
    Resolve a single region-name reference that may contain an alias
    (``a.<alias>`` or ``a.<alias>[k=v, ...]``) to the actual region name.

    Resolution order:

    1. If the input is not a string or has no alias reference, return as-is.
    2. Check the *materialized-alias* map for a full expression match
       (handles parameterized expressions like ``a.band[y0=-0.1]``).
    3. If *regions* is given, expand the alias and search for a region
       whose ``select`` matches the expanded selector (handles the case
       where the materialized map is not populated).
    4. Fall back to looking up the bare alias name's ``region_name``.

    If *aliases* is ``None``, *name* is returned unchanged.
    """
    if aliases is None or not isinstance(name, str):
        return name
    stripped = name.strip()
    m = _alias_ref.match(stripped)
    if m is None:
        return name
    if isinstance(aliases, RegionAliasRegistry):
        reg = aliases
    else:
        reg = RegionAliasRegistry(aliases)

    # 1. Check materialized map (handles parameterized expressions).
    materialized = reg.resolve_materialized(stripped)
    if materialized is not None:
        return materialized

    alias_name = m.group(1)

    # 2. If regions are provided, try to find a match by expanded selector.
    if regions is not None and m.group(2) is not None:
        try:
            expanded = reg.expand(stripped)
        except Exception:
            expanded = None
        if expanded is not None:
            # Search through regions for one whose select matches.
            matched = _find_region_by_selector(regions, expanded)
            if matched is not None:
                reg.record_materialized(stripped, matched)
                return matched

    # 3. Fall back to bare alias name lookup.
    try:
        return reg.get_region_name(alias_name)
    except KeyError:
        pass

    # 4. Alias doesn't exist — provide a helpful error.
    if alias_name in reg:
        params = reg.get_parameters(alias_name)
        if params:
            raise KeyError(
                "parameterized region alias %r (params: %s) referenced as "
                "region name %r, but no matching region was found. Create a "
                "region with this selector first, or check parameter values."
                % (alias_name, ', '.join(params), stripped))
    raise KeyError(
        'unknown region alias %r referenced as region name' % alias_name)


def _find_region_by_selector(regions, selector):
    """
    Search *regions* (a dict or OneTypeList) for a region whose
    ``select``
    attribute matches *selector* (after parenthesis stripping).

    Returns the region name, or ``None`` if not found.
    """
    target = _strip_wrapping_parens(selector.strip())

    if hasattr(regions, 'get'):
        items = regions.values()
    else:
        items = regions

    for region in items:
        sel = getattr(region, 'select', None)
        if sel is None:
            continue
        region_sel = _strip_wrapping_parens(sel.strip())
        if region_sel == target:
            return region.name
    return None


def _strip_wrapping_parens(s):
    """
    Remove outermost parentheses that are purely wrap the entire expression
    (i.e., the inner content is a balanced expression).
    """
    while len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        inner = s[1:-1].strip()
        depth = 0
        balanced = True
        for ch in inner:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth < 0:
                balanced = False
                break
        if balanced and depth == 0:
            s = inner
        else:
            break
    return s


def resolve_alias_region_names(region_attr, aliases, regions=None):
    """
    Resolve alias references in a region-name attribute, which may be a
    single string or a list/tuple of strings (e.g. for EPBC / LCBC).
    """
    if aliases is None:
        return region_attr
    if isinstance(region_attr, (list, tuple)):
        return type(region_attr)(
            resolve_alias_region_name(r, aliases, regions=regions)
            for r in region_attr)
    return resolve_alias_region_name(region_attr, aliases, regions=regions)


def is_alias_expression(selector):
    """
    Return ``True`` if *selector* consists entirely of a single alias
    reference (optionally with parameters), with no other operators or
    text around it.

    Returns the matched alias name if it *is* an alias expression, or
    ``None`` otherwise.
    """
    if not isinstance(selector, str):
        return None
    stripped = selector.strip()
    m = _alias_ref.match(stripped)
    if m is None:
        return None
    if m.end() != len(stripped):
        return None
    return m.group(1)
