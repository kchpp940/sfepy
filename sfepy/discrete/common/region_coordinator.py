"""
Region creation coordinator.

Coordinates the parsing, dependency analysis, and entity construction
layers to create regions from definitions.
"""
from sfepy.base.base import output
from sfepy.base.timing import Timer
from sfepy.discrete.common.region import RegionDependencyAnalyzer, get_parents
from sfepy.discrete.parse_regions import RegionParser
from sfepy.discrete.common.region_builder import RegionBuilder, region_op


class RegionCoordinator:
    """编排区域创建完整流程。

    协调解析、依赖分析和实体构造三个层次，
    负责从区域定义到最终 Region 实例的完整流程。
    """

    def __init__(self, domain):
        """初始化 RegionCoordinator。

        Parameters
        ----------
        domain : Domain
            域实例
        """
        self.domain = domain
        self.parser = RegionParser()

    def parse_and_build(self, select, regions, functions, tdim):
        """解析选择器并构建区域实体。

        Parameters
        ----------
        select : str
            区域选择器
        regions : OneTypeList
            已存在的区域列表
        functions : dict or None
            可用的函数字典
        tdim : int
            拓扑维度

        Returns
        -------
        Region
            构建的区域实例
        """
        self.parser.parse(select)
        builder = RegionBuilder(self.domain, regions, functions)
        return self.parser.visit(region_op,
                                 builder.build_leaf(select, tdim))

    def _check_parents(self, name, select, regions):
        """检查区域依赖的父区域是否存在。

        Parameters
        ----------
        name : str
            区域名称
        select : str
            区域选择器
        regions : OneTypeList
            已存在的区域列表
        """
        parents = get_parents(select)
        for p in parents:
            if p not in [r.name for r in regions]:
                msg = 'parent region %s of %s not found!' % (p, name)
                raise ValueError(msg)

    def _resolve_tdim(self, parent, extra_options, regions):
        """解析区域的拓扑维度。

        Parameters
        ----------
        parent : str or None
            父区域名称
        extra_options : dict or None
            额外选项
        regions : OneTypeList
            已存在的区域列表

        Returns
        -------
        int
            拓扑维度
        """
        tdim = self.domain.shape.tdim if parent is None else regions[parent].tdim
        if extra_options is not None:
            tdim = extra_options.get('cell_tdim', tdim)
        return tdim

    def _finalize_region(self, region, name, select, kind, parent,
                         extra_options, allow_empty, regions, functions):
        """完成区域的最终设置。

        Parameters
        ----------
        region : Region
            待完成的区域实例
        name : str
            区域名称
        select : str
            区域选择器
        kind : str
            区域类型
        parent : str or None
            父区域名称
        extra_options : dict or None
            额外选项
        allow_empty : bool
            是否允许空区域
        regions : OneTypeList
            已存在的区域列表
        functions : dict or None
            可用的函数字典

        Returns
        -------
        Region
            完成设置的区域实例
        """
        eopts = extra_options
        finalize = True

        if eopts is not None:
            if 'mesh_dim' in eopts:
                region = self.domain.create_extra_tdim_region(
                    region, functions, eopts['mesh_dim'])
            if not (eopts.get('finalize', True)):
                finalize = False

            if 'vertices_from' in eopts:
                vreg = eopts['vertices_from']
                region.entities[0] = regions[vreg].vertices.copy()
                finalize = False

        region.name = name
        region.definition = select
        region.set_kind(kind)
        if finalize:
            region.finalize(allow_empty=allow_empty)
        region.parent = parent
        region.extra_options = extra_options
        region.update_shape()

        return region

    def create_region(self, name, select, kind='cell', parent=None,
                      check_parents=True, extra_options=None, functions=None,
                      add_to_regions=True, allow_empty=False):
        """创建单个区域。

        Parameters
        ----------
        name : str
            区域名称
        select : str
            区域选择器
        kind : str
            区域类型
        parent : str or None
            父区域名称
        check_parents : bool
            是否检查父区域存在
        extra_options : dict or None
            额外选项
        functions : dict or None
            可用的函数字典
        add_to_regions : bool
            是否添加到区域列表
        allow_empty : bool
            是否允许空区域

        Returns
        -------
        Region
            创建的区域实例
        """
        regions = self.domain.regions

        if check_parents:
            self._check_parents(name, select, regions)

        tdim = self._resolve_tdim(parent, extra_options, regions)

        region = self.parse_and_build(select, regions, functions, tdim)

        region = self._finalize_region(
            region, name, select, kind, parent, extra_options,
            allow_empty, regions, functions)

        if add_to_regions:
            regions.append(region)

        return region

    def create_regions(self, region_defs, functions=None, allow_empty=False):
        """批量创建区域。

        Parameters
        ----------
        region_defs : dict
            区域定义字典
        functions : dict or None
            可用的函数字典
        allow_empty : bool
            是否允许空区域

        Returns
        -------
        OneTypeList
            创建的区域列表
        """
        output('creating regions...')
        timer = Timer(start=True)

        self.domain.reset_regions()
        regions = self.domain.regions

        ##
        # 依赖分析层
        sorted_regions, name_to_sort_name = \
            RegionDependencyAnalyzer.analyze(region_defs)

        ##
        # 解析和构造层
        for name in sorted_regions:
            sort_name = name_to_sort_name[name]
            rdef = region_defs[sort_name]

            self.create_region(name, rdef.select,
                               kind=rdef.get('kind', 'cell'),
                               parent=rdef.get('parent', None),
                               check_parents=False,
                               extra_options=rdef.get('extra_options', None),
                               functions=functions,
                               allow_empty=allow_empty)
            output(' ', name)

        output('...done in %.2f s' % timer.stop())

        return regions
