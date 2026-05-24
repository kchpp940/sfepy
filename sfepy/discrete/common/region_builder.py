"""
Region entity builder.

Constructs Region instances by parsing selectors and building entity
collections (vertices, faces, cells, etc.).
"""
import numpy as nm

from sfepy.base.base import output, assert_
from sfepy.discrete.common.region import Region


class RegionBuilder:
    """封装区域实体集合构造逻辑。

    负责根据解析结果构建 Region 实例的实体集合（顶点、面、单元等）。
    """

    def __init__(self, domain, regions, functions=None):
        """初始化 RegionBuilder。

        Parameters
        ----------
        domain : Domain
            域实例
        regions : OneTypeList
            已存在的区域列表
        functions : dict, optional
            可用的函数字典
        """
        self.domain = domain
        self.regions = regions
        self.functions = functions or {}

    def build_leaf(self, rdef, tdim):
        """创建叶子区域的构建函数。

        Parameters
        ----------
        rdef : str
            区域定义字符串
        tdim : int
            拓扑维度

        Returns
        -------
        callable
            叶子节点访问器函数
        """
        domain = self.domain
        regions = self.regions
        functions = self.functions

        n_coor = domain.shape.n_nod
        dim = domain.shape.dim
        cmesh = domain.cmesh_tdim[tdim]

        def _region_leaf(level, op):
            token, details = op['token'], op['orig']

            if token != 'KW_Region':
                parse_def = token + '<' + ' '.join(details) + '>'
                if token != 'E_COG':
                    region = Region('leaf', rdef, domain, parse_def=parse_def,
                                    tdim=tdim)

            if token == 'KW_Region':
                details = details[1][2:]
                aux = regions.find(details)
                if not aux:
                    raise ValueError('region %s does not exist' % details)
                else:
                    if rdef[:4] == 'copy':
                        region = aux.copy()
                    else:
                        region = aux

            elif token == 'KW_All':
                region.vertices = nm.arange(n_coor, dtype=nm.uint32)

            elif token == 'E_VIR':
                where = details[2]

                if where[0] == '[':
                    vertices = nm.array(eval(where), dtype=nm.uint32)
                    assert_(nm.amin(vertices) >= 0)
                    assert_(nm.amax(vertices) < n_coor)
                else:
                    coors = cmesh.coors
                    y = z = None
                    x = coors[:, 0]

                    if dim > 1:
                        y = coors[:, 1]

                    if dim > 2:
                        z = coors[:, 2]

                    coor_dict = {'x': x, 'y': y, 'z': z}

                    vertices = nm.where(eval(where, {}, coor_dict))[0]

                region.vertices = vertices

            elif token == 'E_VOS':
                facets = cmesh.get_surface_facets()

                region.set_kind('facet')
                region.facets = facets

            elif token == 'E_VBF':
                where = details[2]

                coors = cmesh.coors

                fun = functions[where]
                vertices = fun(coors, domain=domain)

                region.vertices = vertices

            elif token == 'E_CBF':
                where = details[2]

                coors = domain.get_centroids(dim)

                fun = functions[where]
                cells = fun(coors, domain=domain)

                region.cells = cells

            elif token == 'E_COG':
                group = int(details[3])
                td = 0
                for k in range(4):
                    if domain.cmesh_tdim[k] is not None:
                        cg = domain.cmesh_tdim[k].cell_groups
                        if nm.any(cg == group):
                            td = k
                            break

                if td == 0:
                    raise ValueError('cell group %s does not exist' % group)

                region = Region('leaf', rdef, domain, parse_def=parse_def, tdim=td)
                region.cells = \
                    nm.where(domain.cmesh_tdim[td].cell_groups == group)[0]

            elif token == 'E_COSET':
                raise NotImplementedError('element sets not implemented!')

            elif token == 'E_VOG':
                group = int(details[3])

                region.vertices = nm.where(cmesh.vertex_groups == group)[0]

            elif token == 'E_VOSET':
                try:
                    vertices = domain.vertex_set_bcs[details[3]]

                except KeyError:
                    msg = 'undefined vertex set! (%s)' % details[3]
                    raise ValueError(msg)

                region.vertices = vertices

            elif token == 'E_OVIR':
                aux = regions[details[3][2:]]
                region.vertices = aux.vertices[0:1]

            elif token == 'E_VI':
                region.vertices = nm.array([int(ii) for ii in details[1:]],
                                           dtype=nm.uint32)

            elif token == 'E_CI':
                region.cells = nm.array([int(ii) for ii in details[1:]],
                                        dtype=nm.uint32)

            else:
                output('token "%s" unkown - check regions!' % token)
                raise NotImplementedError
            return region

        return _region_leaf

    @staticmethod
    def apply_op(level, op_code, item1, item2):
        """应用区域集合操作。

        Parameters
        ----------
        level : int
            嵌套层级
        op_code : dict
            操作码
        item1 : Region
            第一个操作数
        item2 : Region
            第二个操作数

        Returns
        -------
        Region
            操作结果区域
        """
        token = op_code['token']
        op = {'S': '-', 'A': '+', 'I': '*'}[token[3]]

        if token[-1] == 'V':
            return item1.eval_op_vertices(item2, op)

        elif token[-1] == 'E':
            return item1.eval_op_edges(item2, op)

        elif token[-1] == 'F':
            return item1.eval_op_faces(item2, op)

        elif token[-1] == 'S':
            return item1.eval_op_facets(item2, op)

        elif token[-1] == 'C':
            return item1.eval_op_cells(item2, op)

        else:
            raise ValueError('unknown region operator token! (%s)' % token)


def region_leaf(domain, regions, rdef, functions, tdim):
    """
    Create/setup a region instance according to rdef.
    """
    builder = RegionBuilder(domain, regions, functions)
    return builder.build_leaf(rdef, tdim)


def region_op(level, op_code, item1, item2):
    return RegionBuilder.apply_op(level, op_code, item1, item2)
