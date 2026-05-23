"""
Helper functions related to mesh facets and Lagrange FE approximation.

Line: ori - iter:

0 - iter0
1 - iter1

Triangle: ori - iter:

0 - iter21
1 - iter12
3 - iter02
4 - iter20
6 - iter10
7 - iter01

Possible couples:

1, 4, 7 <-> 0, 3, 6

Square: ori - iter:

 0 - iter10x01y
 7 - iter10y01x
11 - iter01y01x
30 - iter01x10y
33 - iter10x10y
52 - iter01y10x
56 - iter10y10x
63 - iter01x01y

Possible couples:

7, 33, 52, 63 <-> 0, 11, 30, 56

_quad_ori_groups:

i < j < k < l

all faces are permuted to

l --- k
|     |
|     |
i --- j

ijkl

which is the same as

l --- j
|     |
|     |
i --- k

ikjl

k --- l
|     |
|     |
i --- j

ijlk

- start at one vertex and go around clock-wise or anticlock-wise
-> 8 groups of 3
-> same face nodes order in
ijkl (63), ikjl (59), ijlk (31)
ilkj (11), iklj (15), iljk (43)
jkli ( 7), jlki ( 3), kjli ( 6)
kjil (56), jkil (57), ljik (48)
lijk (52), likj (20), kijl (60)
lkji ( 0), ljki ( 4), klji ( 1)
klij (33), lkij (32), jlik (41)
jilk (30), kilj (22), jikl (62)
"""
from itertools import permutations

import numpy as nm

_quad_ori_groups = {
    0 : 0,
    1 : 0,
    3 : 7,
    4 : 0,
    6 : 7,
    7 : 7,
    11 : 11,
    15 : 11,
    20 : 52,
    22 : 30,
    30 : 30,
    31 : 63,
    32 : 33,
    33 : 33,
    41 : 33,
    43 : 11,
    48 : 56,
    52 : 52,
    56 : 56,
    57 : 56,
    59 : 63,
    60 : 52,
    62 : 30,
    63 : 63,
}

def build_orientation_map(n_fp):
    """
    The keys are binary masks of the lexicographical ordering of facet
    vertices. A bit i set to one means `v[i] < v[i+1]`.

    The values are `[original_order, permutation]`, where `permutation` can be
    used to sort facet vertices lexicographically. Hence `permuted_facet =
    facet[permutation]`.
    """
    indices = list(range(n_fp))

    cmps = [(i1, i2) for i2 in indices for i1 in indices[:i2]]
    powers = [2**ii for ii in range(len(cmps))]

    ori_map = {}
    for indx in permutations(indices):
        key = 0
        sign = 1
        for ip, power in enumerate(powers):
            i1, i2 = cmps[ip]
            less = (indx[i1] < indx[i2])
            key += power * less
            if not less:
                sign *= -1

        isort = nm.argsort(indx)
        ori_map[key] = [indx, isort]

    return ori_map, cmps, powers


def get_facet_cmps_powers(n_fp):
    """
    Return the lexicographical comparison pairs and bit powers used to encode
    the facet orientation key. These are the same (cmps, powers) as computed
    by :func:`build_orientation_map`, so callers can derive the same key
    without rebuilding the full permutation map.
    """
    indices = list(range(n_fp))
    cmps = [(i1, i2) for i2 in indices for i1 in indices[:i2]]
    powers = [2**ii for ii in range(len(cmps))]
    return cmps, powers


def get_facet_orientation_key(vertices, cmps=None, powers=None):
    """
    Compute the lexicographical orientation key for a single facet given its
    (locally numbered) vertex ids.

    The key matches the encoding used by :func:`build_orientation_map`: bit
    ``powers[ip]`` is set iff ``vertices[cmps[ip][0]] < vertices[cmps[ip][1]]``.

    Parameters
    ----------
    vertices : array_like
        The (global) vertex indices of the facet, in the order induced by the
        reference element's local numbering.
    cmps, powers : optional
        Comparison pairs and bit powers as returned by
        :func:`get_facet_cmps_powers`. If not given, they are derived from
        ``len(vertices)``.

    Returns
    -------
    key : int
        The orientation key.
    """
    n_fp = len(vertices)
    if cmps is None or powers is None:
        cmps, powers = get_facet_cmps_powers(n_fp)

    key = 0
    for ip, power in enumerate(powers):
        i1, i2 = cmps[ip]
        if vertices[i1] < vertices[i2]:
            key += power

    return key


def get_facet_orientation_array(ori_map_dict, n_fp, dtype=nm.int32):
    """
    Convert a (sparse) dictionary mapping ``key -> permutation`` into a dense
    ``(max_key + 1, n_fp)`` integer array suitable for indexing with an
    orientation integer. Missing keys are left as zero rows.
    """
    if not len(ori_map_dict):
        return nm.zeros((0, n_fp), dtype=dtype)

    keys = list(ori_map_dict.keys())
    max_key = max(keys)
    out = nm.zeros((max_key + 1, n_fp), dtype=dtype)
    for key, val in ori_map_dict.items():
        out[key] = nm.asarray(val[1], dtype=dtype)
    return out


def get_facet_orientations(cmesh, dim):
    """
    Unified accessor for facet orientation arrays from a :class:`CMesh`
    instance. Returns the raw orientation integers of entities of dimension
    ``dim`` (1 for edges, 2 for faces).
    """
    return cmesh.get_orientations(dim)

def iter0(num):
    for ir in range(num - 1, -1, -1):
        yield ir

def iter1(num):
    for ir in range(num):
        yield ir

ori_line_to_iter = {
    0 : iter0,
    1 : iter1,
}

def make_line_matrix(order):
    if (order < 2):
        return nm.zeros((0, 0), dtype=nm.int32)

    oo = order - 1
    mtx = nm.arange(oo, dtype=nm.int32)

    return mtx

def iter01(num):
    for ir in range(num - 1, -1, -1):
        for ic in range(ir + 1):
            yield ir, ic

def iter10(num):
    for ir in range(num - 1, -1, -1):
        for ic in range(ir, -1, -1):
            yield ir, ic

def iter02(num):
    for ic in range(num):
        for ir in range(num - 1, ic - 1, -1):
            yield ir, ic

def iter20(num):
    for ic in range(num):
        for ir in range(ic, num):
            yield ir, ic

def iter12(num):
    for idiag in range(num):
        irs, ics = nm.diag_indices(num - idiag)
        for ii in range(irs.shape[0] - 1, -1, -1):
            yield irs[ii] + idiag, ics[ii]

def iter21(num):
    for idiag in range(num):
        irs, ics = nm.diag_indices(num - idiag)
        for ii in range(irs.shape[0]):
            yield irs[ii] + idiag, ics[ii]

ori_triangle_to_iter = {
    0 : iter21,
    1 : iter12,
    3 : iter02,
    4 : iter20,
    6 : iter10,
    7 : iter01,
}

def make_triangle_matrix(order):
    if (order < 3):
        return nm.zeros((0, 0), dtype=nm.int32)

    oo = order - 2
    mtx = nm.zeros((oo, oo), dtype=nm.int32)
    for ii, (ir, ic) in enumerate(iter01(oo)):
        mtx[ir, ic] = ii

    return mtx

def iter01x01y(num):
    for ir in range(num):
        for ic in range(num):
            yield ir, ic

def iter01y01x(num):
    for ir, ic in iter01x01y(num):
        yield ic, ir

def iter10x01y(num):
    for ir in range(num - 1, -1, -1):
        for ic in range(num):
            yield ir, ic

def iter10y01x(num):
    for ir, ic in iter10x01y(num):
        yield ic, ir

def iter01x10y(num):
    for ir in range(num):
        for ic in range(num - 1, -1, -1):
            yield ir, ic

def iter01y10x(num):
    for ir, ic in iter01x10y(num):
        yield ic, ir

def iter10x10y(num):
    for ir in range(num - 1, -1, -1):
        for ic in range(num - 1, -1, -1):
            yield ir, ic

def iter10y10x(num):
    for ir, ic in iter10x10y(num):
        yield ic, ir

ori_square_to_iter = {
    0 : iter10x01y,
    7 : iter10y01x,
    11 : iter01y01x,
    30 : iter01x10y,
    33 : iter10x10y,
    52 : iter01y10x,
    56 : iter10y10x,
    63 : iter01x01y,
}

def make_square_matrix(order):
    if (order < 2):
        return nm.zeros((0, 0), dtype=nm.int32)

    oo = order - 1
    mtx = nm.arange(oo * oo, dtype=nm.int32)
    mtx.shape = (oo, oo)

    return mtx

def get_facet_dof_permutations(n_fp, order, node_desc=None):
    """
    Prepare the DOF permutation vector for each possible facet orientation.

    The returned table is bound to both the facet topology (``n_fp`` = 2/3/4
    for edges/triangles/quadrilaterals) and the approximation ``order``.
    For each orientation key (as returned by
    :func:`get_facet_orientation_key`) the table contains the permutation
    that maps the reference element's canonical DOF ordering on the facet
    to the DOF ordering of the lexicographically-sorted facet vertices.

    If ``node_desc`` (a :class:`NodeDescription` instance, or any object
    exposing ``edge_nodes`` / ``face_nodes``) is provided, the permutation
    is restricted to the interior facet DOFs only, matching the slots
    described by ``node_desc``. Otherwise all facet DOFs of the
    :class:`LagrangeSimplexPolySpace` / :class:`LagrangeTensorProductPolySpace`
    node descriptions are used.

    Parameters
    ----------
    n_fp : int
        Number of vertices per facet (2 = edge, 3 = triangle, 4 = quad).
    order : int
        Polynomial approximation order.
    node_desc : Struct, optional
        A node description providing ``edge_nodes`` or ``face_nodes``.
        If given, the permutation length matches ``len(edge_nodes[0])``
        (or ``face_nodes[0]``), including corner vertex DOFs when present.

    Returns
    -------
    dof_perms : ndarray, shape ``(n_ori, n_dof_per_facet)``
        Dense integer permutation table. Entry ``dof_perms[ori, i]`` is the
        local facet-DOF index (relative to the reference element's canonical
        ordering of DOFs on that facet) that should stand at position ``i``
        after the facet vertices have been sorted lexicographically.
    """
    from sfepy.base.base import dict_to_array

    if n_fp == 2:
        mtx = make_line_matrix(order)
        ori_map = ori_line_to_iter
        fo = order - 1

    elif n_fp == 3:
        mtx = make_triangle_matrix(order)
        ori_map = ori_triangle_to_iter
        fo = order - 2

    elif n_fp == 4:
        mtx = make_square_matrix(order)
        ori_map = {}
        for key, val in _quad_ori_groups.items():
            ori_map[key] = ori_square_to_iter[val]
        fo = order - 1

    else:
        raise ValueError('unsupported number of facet points! (%d)' % n_fp)

    dof_perms = {}
    for key, itfun in ori_map.items():
        dof_perms[key] = [mtx[ii] for ii in itfun(fo)]

    if node_desc is not None and n_fp == 2 and node_desc.edge_nodes is not None:
        n_per = len(node_desc.edge_nodes[0])
        for key in dof_perms:
            per = dof_perms[key]
            if len(per) == (n_per - 2):
                dof_perms[key] = [0, 1] + [p + 2 for p in per]
            elif len(per) == n_per:
                pass
            else:
                raise ValueError(
                    'edge node_desc size %d does not match order %d'
                    % (n_per, order))

    elif node_desc is not None and n_fp == 3 and node_desc.face_nodes is not None:
        n_per = len(node_desc.face_nodes[0])
        for key in dof_perms:
            per = dof_perms[key]
            if len(per) == (n_per - 3):
                dof_perms[key] = [0, 1, 2] + [p + 3 for p in per]
            elif len(per) == n_per:
                pass
            else:
                raise ValueError(
                    'tri face node_desc size %d does not match order %d'
                    % (n_per, order))

    elif node_desc is not None and n_fp == 4 and node_desc.face_nodes is not None:
        n_per = len(node_desc.face_nodes[0])
        for key in dof_perms:
            per = dof_perms[key]
            if len(per) == (n_per - 4):
                dof_perms[key] = [0, 1, 2, 3] + [p + 4 for p in per]
            elif len(per) == n_per:
                pass
            else:
                raise ValueError(
                    'quad face node_desc size %d does not match order %d'
                    % (n_per, order))

    dof_perms = dict_to_array(dof_perms)

    return dof_perms


def get_surface_facet_permutations(order, gel, node_desc):
    """
    Build DOF permutation tables for every surface-facet orientation of a
    reference element ``gel`` at approximation ``order``.

    This is a convenience wrapper around :func:`get_facet_dof_permutations`
    that returns the permutation tables for the edge and (if applicable)
    face facets of ``gel``. The tables can be indexed directly with the
    orientation integers from :func:`get_facet_orientation_key` or
    :func:`get_facet_orientations` to obtain the local DOF reordering on
    the facet.

    Parameters
    ----------
    order : int
        Polynomial approximation order.
    gel : GeometryElement
        The reference geometry element.
    node_desc : NodeDescription
        The node description produced by ``gel.poly_space.describe_nodes()``.

    Returns
    -------
    edge_dof_perms : ndarray or None
        Dense edge-DOF permutation table (``n_ori_edge x n_edge_dof``).
    face_dof_perms : ndarray or None
        Dense face-DOF permutation table (``n_ori_face x n_face_dof``).
    """
    edge_dof_perms = face_dof_perms = None

    if node_desc.edge is not None:
        n_fp = gel.edges.shape[1]
        edge_dof_perms = get_facet_dof_permutations(n_fp, order,
                                                    node_desc=node_desc)

    if node_desc.face is not None:
        n_fp = gel.faces.shape[1]
        face_dof_perms = get_facet_dof_permutations(n_fp, order,
                                                    node_desc=node_desc)

    return edge_dof_perms, face_dof_perms


def get_facet_dof_signs(n_fp, order, node_desc=None):
    """
    Prepare the per-DOF sign table for each facet orientation of a hierarchical
    (Lobatto) basis.

    For edges (``n_fp == 2``): the sign of a DOF at position ``p`` on the facet
    is ``+1`` when the edge orientation is canonical (``ori == 0``) or the
    polynomial order is even, and ``-1`` when the edge is flipped
    (``ori == 1``) and the order is odd. This matches the convention
    ``sign = (-1) ** (ori * (order_p % 2))``.

    For quad faces (``n_fp == 4``): the sign encodes the 3-bit face orientation
    used by :class:`LobattoTensorProductPolySpace`. Bit 0 is the axis-swap
    flag; bits 1-2 are multiplied by the parity of the respective axial
    polynomial orders.

    The returned table is in the **canonical** DOF order (matching the
    reference element), so callers must apply the same permutation from
    :func:`get_facet_dof_permutations` before assigning values into the
    orientation-dependent slots.

    Parameters
    ----------
    n_fp : int
        Number of vertices per facet (2 = edge, 4 = quad face).
    order : int
        Polynomial approximation order.
    node_desc : Struct, optional
        Node description providing ``edge_nodes`` / ``face_nodes``.  When
        given, the returned tables have ``len(edge_nodes[0])`` /
        ``len(face_nodes[0])`` columns including corner DOF slots (which are
        always sign-0, as corners carry no orientation sign).

    Returns
    -------
    dof_signs : ndarray, shape ``(n_ori, n_dof_per_facet)``
        Dense sign table.  ``dof_signs[ori, i]`` is the sign value (``0`` or
        ``1`` for edges; 3-bit code for quad faces) for the DOF at canonical
        position ``i`` when the facet orientation is ``ori``.
    """
    if n_fp == 2:
        if order < 2:
            return nm.zeros((2, 2), dtype=nm.int32)

        n_interior = order - 1
        n_ori = 2

        if node_desc is not None and node_desc.edge_nodes is not None:
            n_per = len(node_desc.edge_nodes[0])
            n_dof_per_facet = n_per
            if n_per == n_interior + 2:
                signs = nm.zeros((n_ori, n_dof_per_facet), dtype=nm.int32)
                signs[1, 2:] = 1
            elif n_per == n_interior:
                signs = nm.zeros((n_ori, n_dof_per_facet), dtype=nm.int32)
                signs[1, :] = 1
            else:
                raise ValueError(
                    'edge node_desc size %d does not match order %d'
                    % (n_per, order))
        else:
            signs = nm.zeros((n_ori, n_interior + 2), dtype=nm.int32)
            signs[1, 2:] = 1

        return signs

    elif n_fp == 4:
        if order < 2:
            return nm.zeros((8, 4), dtype=nm.int32)

        n_interior_side = order - 1
        n_interior = n_interior_side * n_interior_side
        n_ori = 8

        translate = nm.zeros(64, dtype=nm.int32)
        new = nm.repeat(nm.arange(8, dtype=nm.int32), 3)
        old = nm.array([31, 59, 63,
                        0, 1, 4,
                        22, 30, 62,
                        32, 33, 41,
                        11, 15, 43,
                        3, 6, 7,
                        20, 52, 60,
                        48, 56, 57], dtype=nm.int32)
        translate[old] = new

        orders = nm.arange(n_interior_side, dtype=nm.int32)
        eoo0 = (orders % 2)[:, None].repeat(n_interior_side, axis=1).ravel()
        eoo1 = (orders % 2)[None, :].repeat(n_interior_side, axis=0).ravel()

        if node_desc is not None and node_desc.face_nodes is not None:
            n_per = len(node_desc.face_nodes[0])
            n_dof_per_facet = n_per
            if n_per == n_interior + 4:
                _signs = nm.zeros((n_ori, n_dof_per_facet), dtype=nm.int32)
                interior_slice = slice(4, None)
            elif n_per == n_interior:
                _signs = nm.zeros((n_ori, n_dof_per_facet), dtype=nm.int32)
                interior_slice = slice(None)
            else:
                raise ValueError(
                    'quad face node_desc size %d does not match order %d'
                    % (n_per, order))
        else:
            _signs = nm.zeros((n_ori, n_interior + 4), dtype=nm.int32)
            interior_slice = slice(4, None)

        for ori in range(n_ori):
            ori_tiled = nm.tile(ori, n_interior)
            bits = nm.where(ori_tiled < 4,
                            nm.bitwise_and(
                                nm.bitwise_and(ori_tiled,
                                               2 * eoo0 + 5),
                                eoo1 + 6),
                            nm.bitwise_and(
                                nm.bitwise_and(ori_tiled,
                                               eoo0 + 6),
                                2 * eoo1 + 5))
            _signs[ori, interior_slice] = bits

        signs = nm.zeros((64, _signs.shape[1]), dtype=nm.int32)
        for old_ori in range(64):
            signs[old_ori] = _signs[translate[old_ori]]

        return signs

    else:
        raise ValueError('unsupported number of facet points! (%d)' % n_fp)


def get_surface_facet_signs(order, gel, node_desc):
    """
    Build sign tables for every surface-facet orientation of a reference
    element ``gel`` at approximation ``order`` for hierarchical (Lobatto)
    bases.

    Parameters
    ----------
    order : int
        Polynomial approximation order.
    gel : GeometryElement
        The reference geometry element.
    node_desc : NodeDescription
        The node description produced by ``gel.poly_space.describe_nodes()``.

    Returns
    -------
    edge_dof_signs : ndarray or None
        Dense edge-DOF sign table (``n_ori_edge x n_edge_dof``).
    face_dof_signs : ndarray or None
        Dense face-DOF sign table (``n_ori_face x n_face_dof``).
    """
    edge_dof_signs = face_dof_signs = None

    if node_desc.edge is not None:
        n_fp = gel.edges.shape[1]
        edge_dof_signs = get_facet_dof_signs(n_fp, order,
                                             node_desc=node_desc)

    if node_desc.face is not None:
        n_fp = gel.faces.shape[1]
        face_dof_signs = get_facet_dof_signs(n_fp, order,
                                             node_desc=node_desc)

    return edge_dof_signs, face_dof_signs


if __name__ == '__main__':
    order = 5
    mtx = make_triangle_matrix(order)
    print(mtx)

    oo = order - 2
    print([mtx[ir, ic] for ir, ic in ori_triangle_to_iter[0](oo)])
    print([mtx[ir, ic] for ir, ic in ori_triangle_to_iter[1](oo)])
    print([mtx[ir, ic] for ir, ic in ori_triangle_to_iter[3](oo)])
    print([mtx[ir, ic] for ir, ic in ori_triangle_to_iter[4](oo)])
    print([mtx[ir, ic] for ir, ic in ori_triangle_to_iter[6](oo)])
    print([mtx[ir, ic] for ir, ic in ori_triangle_to_iter[7](oo)])

    order = 4
    mtx = make_square_matrix(order)
    print(mtx)

    oo = order - 1
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[0](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[7](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[11](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[30](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[33](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[52](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[56](oo)])
    print([mtx[ir, ic] for ir, ic in ori_square_to_iter[63](oo)])
