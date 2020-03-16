# -*- coding: utf-8 -*-

# python_matplotlib.py
# @author yeyuc
# @description
# @created 2020-01-16T18:15:19.130Z+08:00
# @last-modified 2020-01-26T11:25:09.645Z+08:00

"""
概要：基于Matplotlib.pyplot，打包各类图形成类；
方法论：绘图要素（画笔，模板）
主要内容：图形，框架, 配色模板

数据预处理
基础图形: line, scatter, bar
复杂图形: contourf, imshow(灰度矩阵图), Axes3D
简单图形框架: figure, lim, label, ticks, spines, legend, ticklabels
复杂图形框架: 多图(subplot, subplots, subplot2grid, gridspec), 画中画(add_axes), 次坐标轴(twinx)

二期图形
1. 热力图
2. 箱线图
3. 直方图
4. 饼图

二期框架：
1. 双对数坐标轴
"""

import matplotlib.pyplot as plt
import numpy as np

from data.config import DIR_dict
from lib.yeyuc_read_write import ReadWrite


class Preprocessing(object):

    def __init__(self, **kwargs):
        pass

    def import_data(self, **kwargs):
        if kwargs.get("simple_line"):
            """
            绘制简单曲线
            """
            x = np.linspace(-10, 10, 50)
            y = 2 * x + 1
            return x, y
        elif kwargs.get("simple_lines"):
            """
            绘制多条简单曲线(在笛卡尔坐标系上)
            """
            x = np.linspace(-3, 3, 50)
            y1 = 2 * x + 1
            y2 = x ** 2
            return x, y1, y2
        elif kwargs.get("simple_scatter"):
            """
            绘制散点图
            """
            n = 1024  # data size
            x = np.random.normal(0, 1, n)
            y = np.random.normal(0, 1, n)
            t = np.arctan2(y, x)  # for color later on
            return x, y, t
        elif kwargs.get("simple_bars"):
            """
            绘制柱状图(Bar)
            """
            n = 12
            x = np.arange(n)
            y1 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
            y2 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
            return x, y1, y2
        elif kwargs.get("simple_contour"):
            """
            绘制等高线图
            """
            n = 256
            x = np.linspace(-3, 3, n)
            y = np.linspace(-3, 3, n)
            x, y = np.meshgrid(x, y)
            return x, y
        elif kwargs.get("simple_imshow"):
            """
            绘制图片(灰度矩阵)
            """
            a = np.array([
                0.313660827978, 0.365348418405, 0.423733120134, 0.365348418405,
                0.439599930621, 0.525083754405, 0.423733120134, 0.525083754405,
                0.651536351379
            ]).reshape(3, 3)
            return a
        elif kwargs.get("simple_3D"):
            """
            绘制3D图像
            """
            # x, y value
            x = np.arange(-4, 4, 0.25)
            y = np.arange(-4, 4, 0.25)
            x, y = np.meshgrid(x, y)
            r = np.sqrt(x ** 2 + y ** 2)
            # height value
            z = np.sin(r)
            return x, y, z
        elif kwargs.get("simple_subfigs"):
            """
            绘制画中画
            """
            x = [1, 2, 3, 4, 5, 6, 7]
            y = [1, 3, 4, 2, 5, 8, 6]
            return x, y
        elif kwargs.get("simple_twinx"):
            """
            绘制画中画
            """
            x = np.arange(0, 10, 0.1)
            y1 = 0.05 * x ** 2
            y2 = -1 * y1
            return x, y1, y2


class Template(object):

    def __init__(self, **kwargs):
        self.axes = ""
        self.handles = []

    def common_template(self, **kwargs):

        # 配置坐标轴与原点位置(spines): 是否使用笛卡尔坐标系
        if kwargs.get("cartesian"):
            # gca = 'get current axis'
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            ax.xaxis.set_ticks_position('bottom')
            # ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

            ax.spines['bottom'].set_position(('data', 0))
            # the 1st is in 'outward' | 'axes' | 'data'
            # axes: percentage of y axis
            # data: depend on y data

            ax.yaxis.set_ticks_position('left')
            # ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

            ax.spines['left'].set_position(('data', 0))

        # 设置图像有效范围(lim)
        self.axes.set_xlim(kwargs.get("xlim"))
        self.axes.set_ylim(kwargs.get("ylim"))
        if kwargs.get("zlim"):
            self.axes.set_zlim(kwargs.get("zlim"))

        # 设置坐标轴名称(label)
        if kwargs.get("xlabel"):
            self.axes.set_xlabel(kwargs.get("xlabel"))
        if kwargs.get("ylabel"):
            self.axes.set_ylabel(kwargs.get("ylabel"))

        # 设置坐标轴刻度(ticks)和标签(tick_labels)
        if type(kwargs.get("xticks")) == np.ndarray or kwargs.get(
                "xticks") == [] or kwargs.get("xticks"):
            self.axes.set_xticks(kwargs.get("xticks"))
        if type(kwargs.get("yticks")) == np.ndarray or kwargs.get(
                "yticks") == [] or kwargs.get("yticks"):
            self.axes.set_yticks(kwargs.get("yticks"))
        if kwargs.get("xtick_labels"):
            self.axes.set_xticklabels(kwargs.get("xtick_labels"))
        if kwargs.get("ytick_labels"):
            self.axes.set_yticklabels(kwargs.get("ytick_labels"))

        # 设置图例(legend)
        if kwargs.get("show_legend"):
            plt.legend(loc=kwargs.get("loc"))
        elif kwargs.get("legend_labels"):
            plt.legend(handles=self.handles[0]
            if len(self.handles) == 1 else self.handles,
                       labels=kwargs.get("legend_labels"),
                       loc=kwargs.get("loc", "best"))

            # the "," is very important in here l1, = plt... and l2, = plt... for this step
            """legend( handles=(line1, line2, line3),
                    labels=('label1', 'label2', 'label3'),
                    'upper right')
                The *loc* location codes are::

                    'best' : 0,          (currently not supported for figure legends)
                    'upper right'  : 1,
                    'upper left'   : 2,
                    'lower left'   : 3,
                    'lower right'  : 4,
                    'right'        : 5,
                    'center left'  : 6,
                    'center right' : 7,
                    'lower center' : 8,
                    'upper center' : 9,
                    'center'       : 10,"""

        # 设置标题
        if kwargs.get("title"):
            self.axes.set_title(kwargs.get("title"),
                                fontsize=12,
                                fontname="Times New Roman")

        # 对数坐标
        if kwargs.get("xlog"):
            self.axes.set_xscale('log')
        if kwargs.get("ylog"):
            self.axes.set_yscale('log')

        # # 设置坐标轴刻度的字体
        if kwargs.get("tick_font"):
            labels = self.axes.get_xticklabels() + self.axes.get_yticklabels()
            for label in labels:
                label.set_fontname('Times New Roman')
                label.set_fontsize(kwargs.get("tick_font"))
                label.set_bbox(
                    dict(facecolor=kwargs.get("facecolor", "white"),
                         edgecolor=kwargs.get("edgecolor", "none"),
                         alpha=kwargs.get("alpha", 0.8),
                         zorder=kwargs.get("zorder", 2)))

        # 设置色标(colorbar)
        if kwargs.get("colorbar"):
            plt.colorbar(shrink=kwargs.get("shrink", .92))
        return

    def subplots_example(self, **kwargs):
        """
        使用gridspec绘制多图
        """
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(3, 3)
        # use index from 0
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, :2])
        ax3 = plt.subplot(gs[1:, 2])
        ax4 = plt.subplot(gs[-1, 0])
        ax5 = plt.subplot(gs[-1, -2])
        return ax1, ax2, ax3, ax4, ax5


class Axes(object):
    def __init__(self, **kwargs):
        self.axes = ""
        self.fig = ""
        pass

    def draw_line(self, x, y, **kwargs):
        # the "," is very important in here l1, = plt... and l2, = plt... for this step
        l, = self.axes.plot(x,
                            y,
                            color=kwargs.get('color'),
                            marker=kwargs.get('marker'),
                            markersize=kwargs.get('markersize', 4),
                            markerfacecolor=kwargs.get('markerfacecolor'),
                            alpha=kwargs.get('alpha', 1),
                            linewidth=2,
                            linestyle='dashed')
        return l

    def draw_stackplot(self, x, y, **kwargs):
        s = self.axes.stackplot(x, y, baseline=kwargs.get("baseline", "zero"))
        return s

    def draw_scatter(self, x, y, **kwargs):
        s = self.axes.scatter(x,
                              y,
                              c=kwargs.get("c"),
                              s=kwargs.get("s"),
                              alpha=kwargs.get("alpha"))
        return s

    def draw_bar(self, x, y, **kwargs):
        b = self.axes.bar(x,
                          y,
                          width=kwargs.get("width", 0.8),
                          facecolor=kwargs.get("facecolor"),
                          edgecolor=kwargs.get("edgecolor"),
                          label=kwargs.get("label"))

        for x1, y1 in zip(x, y):
            # ha: horizontal alignment
            # va: vertical alignment
            self.axes.text(x1,
                           y1 + kwargs.get("ybias"),
                           '%.2f' % y1,
                           ha=kwargs.get("ha", "center"),
                           va=kwargs.get("va", "bottom"))
        return b

    def draw_barh(self, x, y, **kwargs):
        b = self.axes.barh(x,
                           y,
                           xerr=kwargs.get("xerr"),
                           error_kw=kwargs.get("error_kw", {
                               'ecolor': '0.1',
                               'capsize': 6
                           }),
                           color=kwargs.get("color"),
                           alpha=kwargs.get("alpha", 0.7),
                           label=kwargs.get("label"))
        return b

    def draw_barh1(self, x, y, **kwargs):
        b = self.axes.bar(x=0,
                          bottom=x,
                          width=y,
                          height=kwargs.get("height", 0.5),
                          color=kwargs.get("color"),
                          alpha=kwargs.get("alpha"),
                          orientation='horizontal',
                          label=kwargs.get("label"))
        return b

    def draw_pie(self, sizes, **kwargs):

        wedges, texts, autotexts = self.axes.pie(
            sizes,
            explode=kwargs.get("explode"),
            labels=kwargs.get("labels"),
            autopct=kwargs.get("autopct", '%1.1f%%'),
            shadow=kwargs.get("shadow", True),
            startangle=kwargs.get("startangle", 90))

        return wedges

    def draw_hist(self, x, bins, **kwargs):
        n, bins, patches = self.axes.hist(
            x=x,
            bins=bins,
            density=kwargs.get("density", True),
            histtype=kwargs.get("histtype", "bar"),
            cumulative=kwargs.get("cumulative"),
            label=kwargs.get("label"),
        )
        return n, bins, patches

    def draw_hist2d(self, x, y, bins, **kwargs):
        from matplotlib import colors

        h = self.axes.hist2d(
            x,
            y,
            bins=bins,
            norm=colors.LogNorm(),
        )
        return h

    def draw_hexbin(self, x, y, **kwargs):
        hb = self.axes.hexbin(x,
                              y,
                              gridsize=kwargs.get("gridsize", 50),
                              bins=kwargs.get("bins"),
                              cmap=kwargs.get("cmap", "inferno"))
        cb = plt.colorbar(hb, ax=self.axes)
        return hb

    def draw_errorbar(self, x, y, **kwargs):
        err = self.axes.errorbar(
            x,
            y,
            xerr=kwargs.get("xerr"),
            yerr=kwargs.get("yerr"),
            marker=kwargs.get("marker"),
            markersize=kwargs.get("markersize"),
            linestyle=kwargs.get("linestyle"),
        )
        return err

    def draw_boxplot(self, x, **kwargs):
        """
        x: data
        labels: labels of data
        showmeans: green triangle
        meanline: green line
        showbox: show box
        showcaps: show bottom and top
        notch: shape of "S"
        showfliers: show point

        boxprops: settings of box
        flierprops: settings of point
        medianprops: settings of mean
        meanprops: settings of mean
        meanlineprops: settings of mean

        # 常用属性配置
        boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
        flierprops = dict(marker='o',
                        markerfacecolor='green',
                        markersize=12,
                        linestyle='none')
        medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
        meanpointprops = dict(marker='D',
                            markeredgecolor='black',
                            markerfacecolor='firebrick')
        meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
        """
        box = self.axes.boxplot(
            x,
            labels=kwargs.get("labels"),
            showmeans=kwargs.get("showmeans"),
            meanline=kwargs.get("meanline"),
            showbox=kwargs.get("showbox", True),
            showcaps=kwargs.get("showcaps", True),
            notch=kwargs.get("notch"),
            bootstrap=kwargs.get("bootstrap"),
            showfliers=kwargs.get("showfliers", True),
            boxprops=kwargs.get("boxprops"),
            flierprops=kwargs.get("flierprops"),
            medianprops=kwargs.get("medianprops"),
            meanprops=kwargs.get("meanpointprops",
                                 kwargs.get("meanlineprops")),
        )
        return box

    def draw_violinplot(self, x, **kwargs):
        if kwargs.get("simple"):
            box = self.axes.violinplot(
                x,
                showmeans=kwargs.get("showmeans", True),
                showmedians=kwargs.get("meanline", True),
                showextrema=kwargs.get("showbox", True),
            )
        elif kwargs.get("complex"):

            def adjacent_values(vals, q1, q3):
                upper_adjacent_value = q3 + (q3 - q1) * 1.5
                upper_adjacent_value = np.clip(upper_adjacent_value, q3,
                                               vals[-1])

                lower_adjacent_value = q1 - (q3 - q1) * 1.5
                lower_adjacent_value = np.clip(lower_adjacent_value, vals[0],
                                               q1)
                return lower_adjacent_value, upper_adjacent_value

            box = self.axes.violinplot(
                x,
                showmeans=kwargs.get("showmeans", False),
                showmedians=kwargs.get("meanline", False),
                showextrema=kwargs.get("showbox", False),
            )

            for pc in box['bodies']:
                pc.set_facecolor('#D43F3A')
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            quartile1, medians, quartile3 = np.percentile(x, [25, 50, 75],
                                                          axis=1)
            whiskers = np.array([
                adjacent_values(sorted_array, q1, q3)
                for sorted_array, q1, q3 in zip(x, quartile1, quartile3)
            ])
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self.axes.scatter(inds,
                              medians,
                              marker='o',
                              color='white',
                              s=30,
                              zorder=3)
            self.axes.vlines(inds,
                             quartile1,
                             quartile3,
                             color='k',
                             linestyle='-',
                             lw=5)
            self.axes.vlines(inds,
                             whiskers_min,
                             whiskers_max,
                             color='k',
                             linestyle='-',
                             lw=1)
        return box

    def draw_contour(self, x, y, h, **kwargs):
        # use plt.contourf to filling contours
        # x, y and value for (x,y) point
        self.axes.contourf(x, y, h, 8, alpha=.75, cmap=plt.cm.hot)

        # use plt.contour to add contour lines
        C = self.axes.contour(x, y, h, 8, colors='black', linewidths=.5)
        # adding label
        self.axes.clabel(C, inline=True, fontsize=10)

    def draw_imshow(self, a, **kwargs):
        """
        for the value of "interpolation", check this:
        http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
        for the value of "origin"= ['upper', 'lower'], check this:
        http://matplotlib.org/examples/pylab_examples/image_origin.html
        """
        im = plt.imshow(a,
                        interpolation=kwargs.get("interpolation", "nearest"),
                        cmap=kwargs.get("cmap", "bone"),
                        origin=kwargs.get("origin", "lower"))
        return im

    def draw_3D(self, x, y, z, **kwargs):
        from mpl_toolkits.mplot3d import Axes3D

        self.axes = Axes3D(fig=self.fig)
        self.axes.plot_surface(x,
                               y,
                               z,
                               rstride=1,
                               cstride=1,
                               cmap=plt.get_cmap('rainbow'))
        """
        ============= ================================================
        Argument      Description
        ============= ================================================
        *x*, *y*, *z* Data values as 2D arrays
        *rstride*     Array row stride (step size), defaults to 10
        *cstride*     Array column stride (step size), defaults to 10
        *color*       Color of the surface patches
        *cmap*        A colormap for the surface patches.
        *facecolors*  Face colors for the individual patches
        *norm*        An instance of Normalize to map values to colors
        *vmin*        Minimum value to map
        *vmax*        Maximum value to map
        *shade*       Whether to shade the facecolors
        ============= ================================================
        """

        # I think this is different from plt12_contours
        self.axes.contourf(x,
                           y,
                           z,
                           zdir='z',
                           offset=-2,
                           cmap=plt.get_cmap('rainbow'))
        """
        ==========  ================================================
        Argument    Description
        ==========  ================================================
        *x*, *y*,   Data values as numpy.arrays
        *z*
        *zdir*      The direction to use: x, y or z (default)
        *offset*    If specified plot a projection of the filled contour
                    on this position in plane normal to zdir
        ==========  ================================================
        """

        return


class PythonMatplotlib(Preprocessing, Template, Axes, ReadWrite):

    def __init__(self, **kwargs):
        Preprocessing.__init__(self)
        Axes.__init__(self)
        Template.__init__(self)

        self.fig, self.axes = plt.subplots(num=kwargs.get("num"),
                                           figsize=kwargs.get("figsize"))
        self.handles = []

        self.ALPHA = [1, 1, 1, 1, 1, 1]
        self.COLOR = [plt.get_cmap('tab20c').colors[i] for i in [0, 4, 8, 12, 16, 18]]
        self.MARKER = ['^', 'o', 's', '*', '+', 'D']
        self.MARKER_COLOR = [plt.get_cmap('tab20c').colors[i] for i in [1, 5, 8, 12, 16, 18]]

    #####控制台：绘图的通用模式#####

    def fig_show(self, **kwargs):
        plt.tight_layout()
        self.fig.show()
        plt.pause(3)
        plt.close()
        return

    def fig_save(self, **kwargs):
        save_name = kwargs.get("save_name", "untitled")
        self.fig.savefig(DIR_dict.get("PDF_DIR") + '\\' + save_name + '.pdf',
                         dpi=500,
                         bbox_inches='tight')
        return

    ##########一期：莫烦示例改写##########
    def simple_line(self, **kwargs):
        # 数据预处理
        x, y = self.import_data(simple_line=True)

        # 绘制图形
        self.draw_line(x=x, y=y, color='red', linewidth=1.0, linestyle='--')

        # 使用模板
        self.common_template(xlim=(-1, 2),
                             ylim=(-2, 3),
                             xlabel="I am x",
                             ylabel="I am y",
                             xticks=np.linspace(-1, 2, 5),
                             yticks=[-2, -1.8, -1, 1.22, 3],
                             ytick_labels=[
                                 r'$really\ bad$', r'$bad$', r'$normal$',
                                 r'$good$', r'$really\ good$'
                             ])

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_line"))

    def simple_lines(self, **kwargs):
        # 数据预处理
        x, y1, y2 = self.import_data(simple_lines=True)

        # 绘制图形
        self.handles.append(self.draw_line(x=x, y=y1))
        self.handles.append(
            self.draw_line(x=x,
                           y=y2,
                           color='red',
                           linewidth=1.0,
                           linestyle='--'))

        # 使用模板
        self.common_template(
            xlim=(-1, 2),
            ylim=(-2, 3),
            #  xlabel="I am x",
            #  ylabel="I am y",
            xticks=np.linspace(-1, 2, 5),
            yticks=[-2, -1.8, -1, 1.22, 3],
            ytick_labels=[
                r'$really\ bad$', r'$bad$', r'$normal$', r'$good$',
                r'$really\ good$'
            ],
            tick_font=12,
            cartesian=True,
            # legend_labels=["up", "down"]
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_lines"))

    def simple_scatters(self, **kwargs):
        # 数据预处理
        x, y, T = self.import_data(simple_scatter=True)

        # 绘制图形
        self.draw_scatter(x=x, y=y, s=75, c=T, alpha=0.5)

        # 使用模板
        self.common_template(
            xlim=(-1.5, 1.5),
            ylim=(-1.5, 1.5),
            xticks=[],
            yticks=[],
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_scatters"))

    def simple_bars(self, **kwargs):
        # 数据预处理
        x, y1, y2 = self.import_data(simple_bars=True)

        # 绘制图形
        self.handles.append(
            self.draw_bar(
                x=x,
                y=y1,
                ybias=0.05,
                facecolor='#9999ff',
                edgecolor='white',
            ))
        self.handles.append(
            self.draw_bar(x=x,
                          y=-y2,
                          ybias=-0.05,
                          facecolor='#ff9999',
                          edgecolor='white',
                          va="top"))

        # 使用模板
        self.common_template(
            xlim=(-.5, 12),
            ylim=(-1.25, 1.25),
            xticks=[],
            yticks=[],
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_bars"))

    def simple_contour(self, **kwargs):
        # 数据预处理

        def f(x, y):
            # the height function
            return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

        x, y = self.import_data(simple_contour=True)
        h = f(x, y)

        # 绘制图形
        self.draw_contour(x=x, y=y, h=h)

        # 使用模板
        self.common_template(
            xticks=[],
            yticks=[],
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_contour"))

    def simple_imshow(self, **kwargs):
        # 数据预处理
        a = self.import_data(simple_imshow=True)

        # 绘制图形
        self.handles.append(self.draw_imshow(a=a))

        # 使用模板
        self.common_template(
            xticks=[],
            yticks=[],
            colorbar=True,
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_imshow"))

    def simple_3D(self, **kwargs):
        # 数据预处理
        x, y, z = self.import_data(simple_3D=True)

        # 绘制图形
        self.handles.append(self.draw_3D(x=x, y=y, z=z))

        # 使用模板
        self.common_template(zlim=(-2, 2))

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_3D"))

    def simple_subplots(self, **kwargs):
        ax1, ax2, ax3, ax4, ax5 = self.subplots_example()

        self.axes = ax1
        self.simple_line(fig_show=False)

        self.axes = ax2
        self.simple_lines(fig_show=False)

        self.axes = ax3
        self.simple_scatters(fig_show=False)

        self.axes = ax4
        self.simple_bars(fig_show=False)

        self.axes = ax5
        self.simple_contour(fig_show=False)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_subplots"))

    def simple_subfigs(self, **kwargs):

        self.common_template(xticks=[], yticks=[])  # 清楚原坐标值

        # 数据预处理
        x, y = self.import_data(simple_subfigs=True)

        # 绘制图形, 使用模板
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        self.axes = self.fig.add_axes([left, bottom, width,
                                       height])  # main axes
        self.draw_line(x=x, y=y, color='red')
        self.common_template(xlabel="x", ylabel="y", title="title")

        self.axes = self.fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
        self.draw_line(x=y, y=x, color='blue')
        self.common_template(xlabel="x", ylabel="y", title="title inside 1")

        self.axes = self.fig.add_axes([0.6, 0.2, 0.25, 0.25])  # inside axes
        self.draw_line(x=y[::-1], y=x, color='green')
        self.common_template(xlabel="x", ylabel="y", title="title inside 2")

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_subfigs"))

    def simple_twinx(self, **kwargs):
        # self.common_template(xticks=[], yticks=[])  # 清楚原坐标值

        # 数据预处理
        x, y1, y2 = self.import_data(simple_twinx=True)

        # 绘制图形, 使用模板
        self.draw_line(x=x, y=y1, color='green')
        self.axes = self.axes.twinx()  # mirror the ax1
        self.draw_line(x=x, y=y2, color='blue')

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_twinx"))

    ##########二期：https://gitee.com/LeetCode_player/matplotlib/tree/master/examples##########
    ##########line_stack, Pie, barh, hist, hexbin, boxplot, heatmap##########
    def stack_plot(self, **kwargs):
        """
        ==============
        Stackplot Demo
        ==============

        How to create stackplots with Matplotlib.

        Stackplots are generated by plotting different datasets vertically on
        top of one another rather than overlapping with one another. Below we
        show some examples to accomplish this with Matplotlib.
        """

        # 数据预处理
        x = [1, 2, 3, 4, 5]
        y1 = [1, 1, 2, 3, 5]
        y2 = [0, 4, 2, 6, 8]
        y3 = [1, 3, 5, 7, 9]
        y = np.vstack([y1, y2, y3])

        # 绘制图形
        self.handles.append(self.draw_stackplot(x=x, y=y))

        # 使用模板
        self.common_template(legend_labels=["Fibonacci ", "Evens", "Odds"],
                             loc="upper left")

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "stack_plot"))

    def simple_pie(self, **kwargs):
        """
        ===============
        Basic pie chart
        ===============

        Demo of a basic pie chart plus a few additional features.

        In addition to the basic pie chart, this demo shows a few optional features:

            * slice labels
            * auto-labeling the percentage
            * offsetting a slice with "explode"
            * drop-shadow
            * custom start angle

        Note about the custom start angle:

        The default ``startangle`` is 0, which would start the "Frogs" slice on the
        positive x-axis. This example sets ``startangle = 90`` such that everything is
        rotated counter-clockwise by 90 degrees, and the frog slice starts on the
        positive y-axis.
        """

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # 数据预处理
        labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
        sizes = [15, 30, 45, 10]
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        # 绘制图形
        self.handles.append(self.draw_pie(sizes=sizes, explode=explode))
        # self.handles.append(
        #     self.draw_pie(sizes=sizes, labels=labels, explode=explode))

        self.axes.axis(
            'equal'
        )  # Equal aspect ratio ensures that pie is drawn as a circle.

        # 使用模板
        self.common_template(
            title="Matplotlib bakery: A pie",
            legend_labels=labels,
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_pie"))

    def simple_bars_combine(self, **kwargs):
        # 数据预处理
        labels = ['G1', 'G2', 'G3', 'G4', 'G5']
        x = np.arange(len(labels))
        y1 = [20, 34, 30, 35, 27]
        y2 = [25, 32, 34, 20, 25]
        width = 0.35

        # 绘制图形
        self.handles.append(
            self.draw_bar(x=x - (width + 0.01) / 2,
                          y=y1,
                          width=width,
                          ybias=0.05,
                          facecolor='#9999ff',
                          edgecolor='white',
                          label='Men'))

        self.handles.append(
            self.draw_bar(x=x + (width + 0.01) / 2,
                          y=y2,
                          width=width,
                          ybias=0.05,
                          facecolor='#ff9999',
                          edgecolor='white',
                          label='Women'))

        # 使用模板
        self.common_template(ylabel="Scores",
                             title="Scores by group and gender",
                             xticks=x,
                             xtick_labels=labels,
                             show_legend=True)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name",
                                               "simple_bars_combine"))

    def horizontal_bar(self, **kwargs):
        # 数据预处理
        x = np.arange(5)
        labels = ['A', 'B', 'C', 'D', 'E']
        y = [5, 7, 3, 4, 6]
        std = [0.8, 1, 0.4, 0.9, 1.3]

        # 绘制图形
        if kwargs.get("fun1", True):
            self.handles.append(
                self.draw_barh(x=x, y=y, color='b', alpha=0.7, label='First'))
        else:
            self.handles.append(
                self.draw_barh1(x=x, y=y, alpha=0.7, color='b', label='First'))

        # 使用模板
        self.common_template(
            yticks=x,
            ytick_labels=labels,
            show_legend=True,
            loc=5,
            title="geek-docs.com",
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "horizontal_bar"))

    def simple_hist(self, **kwargs):
        """
        =========================================================
        Demo of the histogram (hist) function with a few features
        =========================================================

        In addition to the basic histogram, this demo shows a few optional features:

        * Setting the number of data bins.
        * The ``normed`` flag, which normalizes bin heights so that the integral of
        the histogram is 1. The resulting histogram is an approximation of the
        probability density function.
        * Setting the face color of the bars.
        * Setting the opacity (alpha value).

        Selecting different bin counts and sizes can significantly affect the shape
        of a histogram. The Astropy docs have a great section_ on how to select these
        parameters.

        .. _section: http://docs.astropy.org/en/stable/visualization/histogram.html
        """

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # 数据预处理
        np.random.seed(19680801)

        # example data
        mu = 100  # mean of distribution
        sigma = 15  # standard deviation of distribution
        x = mu + sigma * np.random.randn(437)
        num_bins = 50

        # 绘制图形
        n, bins, patches = self.draw_hist(x=x, bins=num_bins)

        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        self.draw_line(x=bins, y=y, linestyle="--")

        # 使用模板
        self.common_template(
            xlabel="Smarts",
            ylabel="Probability density",
            title=r'Histogram of IQ: $\mu=100$, $\sigma=15$',
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name",
                                               "simple_histogram"))

    def cumulative_hist(self, **kwargs):
        """
        ==================================================
        Using histograms to plot a cumulative distribution
        ==================================================

        This shows how to plot a cumulative, normalized histogram as a
        step function in order to visualize the empirical cumulative
        distribution function (CDF) of a sample. We also show the theoretical CDF.

        A couple of other options to the ``hist`` function are demonstrated.
        Namely, we use the ``normed`` parameter to normalize the histogram and
        a couple of different options to the ``cumulative`` parameter.
        The ``normed`` parameter takes a boolean value. When ``True``, the bin
        heights are scaled such that the total area of the histogram is 1. The
        ``cumulative`` kwarg is a little more nuanced. Like ``normed``, you
        can pass it True or False, but you can also pass it -1 to reverse the
        distribution.

        Since we're showing a normalized and cumulative histogram, these curves
        are effectively the cumulative distribution functions (CDFs) of the
        samples. In engineering, empirical CDFs are sometimes called
        "non-exceedance" curves. In other words, you can look at the
        y-value for a given-x-value to get the probability of and observation
        from the sample not exceeding that x-value. For example, the value of
        225 on the x-axis corresponds to about 0.85 on the y-axis, so there's an
        85% chance that an observation in the sample does not exceed 225.
        Conversely, setting, ``cumulative`` to -1 as is done in the
        last series for this example, creates a "exceedance" curve.

        Selecting different bin counts and sizes can significantly affect the
        shape of a histogram. The Astropy docs have a great section on how to
        select these parameters:
        http://docs.astropy.org/en/stable/visualization/histogram.html

        """

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

        # 数据预处理
        mu = 200
        sigma = 25
        num_bins = 50
        x = np.random.normal(mu, sigma, size=100)

        # 绘制图形
        n, bins, patches = self.draw_hist(x=x,
                                          bins=num_bins,
                                          density=True,
                                          histtype="step",
                                          cumulative=True,
                                          label='Empirical')

        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        y = y.cumsum()
        y /= y[-1]
        self.draw_line(x=bins, y=y, linestyle="--")

        # 使用模板
        self.common_template(
            xlabel="Annual rainfall (mm)",
            ylabel="Likelihood of occurrence",
            title="Cumulative step histograms",
            show_legend=True,
            loc="right",
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "cumulative_hist"))

    def hist_2d(self, **kwargs):
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        ###############################################################################
        # Generate data and plot a simple histogram
        # -----------------------------------------
        #
        # To generate a 1D histogram we only need a single vector of numbers. For a 2D
        # histogram we'll need a second vector. We'll generate both below, and show
        # the histogram for each vector.

        # 数据预处理
        # Generate a normal distribution, center at x=0 and y=5
        x = np.random.randn(100000)
        y = .4 * x + np.random.randn(100000) + 5
        # n_bins = (20, 20)
        n_bins = 40

        # 绘制图形
        self.handles.append(self.draw_hist2d(x=x, y=y, bins=n_bins))

        # 使用模板
        self.common_template()

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "hist_2d"))

    def simple_hexbin(self, **kwargs):
        """
        ===========
        Hexbin Demo
        ===========

        Plotting hexbins with Matplotlib.

        Hexbin is an axes method or pyplot function that is essentially
        a pcolor of a 2-D histogram with hexagonal cells.  It can be
        much more informative than a scatter plot. In the first plot
        below, try substituting 'scatter' for 'hexbin'.
        """

        # 数据预处理
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        n = 100000
        x = np.random.standard_normal(n)
        y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        # 绘制图形
        self.handles.append(
            self.draw_hexbin(x=x, y=y, gridsize=50, cmap='inferno',
                             bins="log"))

        # 使用模板
        self.common_template(xlim=(xmin, xmax), ylim=(ymin, ymax))

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "hist_2d"))

    def simple_boxplot(self, **kwargs):
        """
        =================================
        Artist customization in box plots
        =================================

        This example demonstrates how to use the various kwargs
        to fully customize box plots. The first figure demonstrates
        how to remove and add individual components (note that the
        mean is the only value not shown by default). The second
        figure demonstrates how the styles of the artists can
        be customized. It also demonstrates how to set the limit
        of the whiskers to specific percentiles (lower right axes)

        A good general reference on boxplots and their history can be found
        here: http://vita.had.co.nz/papers/boxplots.pdf

        """

        # 数据预处理
        np.random.seed(19680801)
        data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
        labels = list('ABCD')
        fs = 10  # fontsize

        # 绘制图形
        self.handles.append(
            self.draw_boxplot(x=data,
                              labels=labels,
                              showmeans=True,
                              meanline=True,
                              notch=True,
                              bootstrap=10000,
                              showfliers=False))

        # 使用模板
        self.common_template(title="simple_boxplot", )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_boxplot"))

    def simple_violinplot(self, **kwargs):
        """
        =========================
        Violin plot customization
        =========================

        This example demonstrates how to fully customize violin plots.
        The first plot shows the default style by providing only
        the data. The second plot first limits what matplotlib draws
        with additional kwargs. Then a simplified representation of
        a box plot is drawn on top. Lastly, the styles of the artists
        of the violins are modified.

        For more information on violin plots, the scikit-learn docs have a great
        section: http://scikit-learn.org/stable/modules/density.html
        """

        # 数据预处理
        np.random.seed(19680801)
        data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]
        labels = ['A', 'B', 'C', 'D']

        # 绘制图形
        if kwargs.get("simple"):
            self.handles.append(self.draw_violinplot(simple=True, x=data))
        elif kwargs.get("complex"):
            self.handles.append(self.draw_violinplot(complex=True, x=data))

        # 使用模板
        self.common_template(title="Customized violin plot",
                             ylabel="Observed values",
                             xticks=np.arange(1,
                                              len(labels) + 1),
                             xtick_labels=labels)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name",
                                               "simple_violinplot"))

    def simple_heatmap(self, **kwargs):
        # 数据预处理

        x = np.random.rand(100).reshape(10, 10)
        # x = np.random.rand(16).reshape(4, 4)
        # import pandas as pd
        # attr = ['a', 'b', 'c', 'd']
        # x = pd.DataFrame(x, columns=attr, index=attr)

        # 绘制图形, 使用模板
        if kwargs.get("imshow"):
            plt.imshow(x, cmap=plt.cm.hot, vmin=0, vmax=1)
            save_name = "simple_heatmap_imshow"
            self.common_template(colorbar=True)
        elif kwargs.get("matshow"):
            plt.matshow(x, cmap=plt.cm.cool, vmin=0, vmax=1)
            save_name = "simple_heatmap_matshow"
            self.common_template(colorbar=True)
        elif kwargs.get("seaborn"):
            import seaborn as sns
            sns.heatmap(x, vmin=0, vmax=1, center=0)
            save_name = "simple_heatmap_seaborn"

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", save_name))

    ##########心形曲线##########
    def heart_fun(self, **kwargs):
        def heart(x):
            y = (x ** 2) ** (1 / 3) + (4 - (
                    (np.abs(x + 2) - np.abs(x - 2)) / 2) ** 2) ** (1 / 2) * np.sin(
                10 * np.pi * x)
            return y

        # 导入数据
        x = np.linspace(-3, 3, 10000)
        y = heart(x)

        save_name = "Surpise, maybe!"

        # 绘制图形
        self.handles.append(self.draw_line(x=x, y=y))

        # 使用模板
        self.common_template(title="My heart for you", xlim=(-3, 3))
        #  cartesian=True)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", save_name))

    def heatmap1(self, df, **kwargs):

        # 绘制图形, 使用模板
        import seaborn as sns
        sns.heatmap(df, annot=True, vmax=1, square=True, linewidths=0.05, cmap='YlGnBu', xticklabels=False, cbar=False)

        self.common_template(
            tick_font=16,
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", 'untitled'))

    def lines1(self, x, data, **kwargs):
        # 绘制图形
        for i, y in enumerate(data):
            self.handles.append(
                self.draw_line(x=x,
                               y=y,
                               color=self.COLOR[i],
                               marker=self.MARKER[i],
                               markersize=4,
                               markerfacecolor=self.MARKER_COLOR[i],
                               alpha=self.ALPHA[i],
                               linewidth=2,
                               linestyle='dashed'))

        # 使用模板
        self.common_template(
            xlabel=kwargs.get('xlabel'),
            ylabel=kwargs.get('ylabel'),
            xlog=kwargs.get('xlog'),
            ylog=kwargs.get('ylog'),
            legend_labels=kwargs.get('legend_labels'),
        )

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "untitled"))

    def subplots1(self, **kwargs):
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(1, 2)
        # use index from 0
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        self.axes = ax1
        self.simple_line(fig_show=False)

        self.axes = ax2
        self.simple_lines(fig_show=False)

        # 输出与保存(PDF)
        if kwargs.get("fig_show", True):
            self.fig_show()
        if kwargs.get("save_as_pdf"):
            self.fig_save(save_as_pdf=kwargs.get("save_as_pdf"),
                          save_name=kwargs.get("save_name", "simple_subplots"))


if __name__ == '__main__':
    client = PythonMatplotlib(figsize=(12, 6))
    # client = PythonMatplotlib(figsize=(16, 6))
    client.subplots1()

    # client.subplots1(save_as_pdf=False)

    # x = [1, 2, 3]
    # y = [2, 4, 5]
    # draw_line(x, y)

    # client = PythonMatplotlib()
    # client.simple_line(save_as_pdf=True)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_lines(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_scatters(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_bars(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_contour(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_imshow(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_3D(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_subplots(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_subfigs(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_twinx(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.stack_plot(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_pie(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_bars_combine(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.horizontal_bar(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_hist(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()

    # client.cumulative_hist(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()

    # client.hist_2d(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_hexbin(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()

    # client.simple_boxplot(save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()

    # client.simple_violinplot(simple=True, save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_violinplot(complex=True, save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_heatmap(imshow=True, save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_heatmap(matshow=True, save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    #
    # client.simple_heatmap(seaborn=True, save_as_pdf=False)
    # client.fig, client.axes = plt.subplots()
    pass
