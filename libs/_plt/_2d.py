# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://matplotlib.org/stable/gallery/index.html

from .._types import *
sns.reset_orig()
# sns.set()
#
__all__ = [
    "ArrayLike", "Color", "Cmap", "Marker", "LineStyle",
    #
    "config_plt", "config_ax",
    "get_figure_2d", "save_and_show",
    #
    "plot", "scatter", "imshow", "hist"
]
ArrayLike = Union[Tensor, ndarray, List[float]]
Color = Literal[
    "r", "g", "b", "w", "k", "y", "grey",
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
    "#000000",
    "#333",
    None
]
# https://matplotlib.org/stable/gallery/color/colormap_reference.html
Cmap = Literal[
    None, "viridis",  # None 即 "viridis"
    "gray", "gray_r",
    "hot", "Blues"
]
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
Marker = Literal[".", "o", "*", None]
LineStyle = Literal["-", "--"]


def config_plt(
    backend: Literal["Agg", "TkAgg", None] = None,  # 非交互式, 交互式, 不变
    chinese: bool = False
) -> None:
    if chinese:
        plt.rcParams['font.sans-serif'].insert(0, 'SimSun')
        plt.rcParams['axes.unicode_minus'] = False
    if backend is not None:
        matplotlib.use(backend)


def get_figure_2d(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 200,
    #
    squeeze: bool = True,
) -> Tuple[Figure, Union[Axes, ndarray]]:
    """
    figsize: W,H
    nrows: 行数
    ncols: 列数
    return: 对axes会自动压缩(if squeeze=True). 
    """
    if figsize is None:
        figsize = (8, 5)
        if nrows > 1 or ncols > 1:
            figsize = ncols * 4, nrows * 4
    #
    fig, ax = plt.subplots(nrows, ncols, squeeze=squeeze,
                           figsize=figsize, dpi=dpi)
    return fig, ax


def config_fig(
    fig: Figure,
    title: Optional[str] = None,
    hspace: float = 0.2,
    wspace: float = 0.2,
) -> None:
    title_size = 18
    #
    if title is not None:
        fig.suptitle(title, fontsize=title_size)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)


# if __name__ == "__main__":
#     fig, axs = get_figure_2d(2, 3)
#     config_fig(fig, "AAA")
#     plt.show()


def config_ax(
    ax: Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    #
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    #
    axis: Literal["off", "on", None] = None,
    grid: Literal["both", "y", "off", None] = None,
    legend: bool = False,
    #
    xscale: Literal["linear", "log", None] = None,
    yscale: Literal["linear", "log", None] = None,
    #
    xticks: Optional[ArrayLike] = None,
    xticks_labels: Optional[List[str]] = None,
    yticks: Optional[ArrayLike] = None,
    yticks_labels: Optional[List[str]] = None,
    xticks_rotate90: bool = False
) -> None:
    """
    xticks_labels: 必须提供xticks
        xticks: [N]
        xticks_labels: [N]
    """
    title_size = 17
    label_size = 14
    tick_size = 11
    pad = 7
    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=pad, weight="bold")  # pad 默认: 6
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=pad, fontsize=label_size, weight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=pad, fontsize=label_size, weight="bold")
    #
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    #
    if grid == "off":
        ax.grid(False)
    elif grid is not None:
        ax.grid(True, axis=grid)
    #
    if axis is not None:
        ax.axis(axis)
    if legend:
        ax.legend()
    #
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    #
    if xticks is not None:
        ax.set_xticks(xticks, xticks_labels)
    if yticks is not None:
        ax.set_yticks(yticks, yticks_labels)
    #
    if xticks_rotate90:
        ax.tick_params(axis='x', which='major', rotation=90)
    #
    ax.tick_params(axis="both", which="major", labelsize=tick_size)


def save_and_show(
    fpath: Optional[str] = None,
    #
    dpi: int = 200,
    show: bool = True,
    close: bool = True
) -> None:
    if fpath is not None:
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()


# if __name__ == "__main__":
#     fig, ax = get_figure_2d()
#     config_ax(ax, "aaa", "bbb", "ccc", (0, None), (0, None),
#               axis="on", grid="y",
#               xticks_rotate90=True)
#     save_and_show()


def plot(
    ax: Axes,
    x: Optional[ArrayLike],
    y: ArrayLike,
    #
    color: Color = None,
    marker: Marker = None,
    linestyle: LineStyle = "-",
    #
    markersize: int = 6,
    markeredgecolor: Color = "k",
    markerfacecolor: Color = None,
    linewidth: float = 1.5,
    alpha: float = 1,  # 0.7, 0.5
    zorder: Optional[int] = None,
    #
    label: Optional[str] = None,
) -> None:
    """
    x: [N]. 默认np.arange(length)
    y: [N]
    zorder: 多个直线的覆盖的先后顺序, 越大越上面, 越不容易被覆盖. 
    """
    if x is None:
        N = len(y) if isinstance(y, list) else y.shape[0]
        x = np.arange(N)
    ax.plot(  # sns.lineplot
        x=x, y=y, marker=marker, linestyle=linestyle, color=color,
        markersize=markersize, markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor, linewidth=linewidth,
        alpha=alpha, zorder=zorder, label=label,  # ax=ax
    )


# if __name__ == "__main__":
#     y = np.sin(np.linspace(0, 10, 10))
#     fig, ax = get_figure_2d()
#     plot(ax, None, y, marker="o", linestyle="-")
#     save_and_show()
#     #
#     fig, ax = get_figure_2d()
#     plot(ax, None, y, "k", "*", "--", 16, "k", "y")
#     save_and_show()


def hist(
    ax: Axes,
    x: ArrayLike,
    bins: Union[int, Literal["auto"]] = "auto",
    color: Color = None,
    #
    stat: Literal["density", "count"] = "density",
    kde: bool = False,
    edgecolor: Color = "w",
    #
    label: Optional[str] = None,
    alpha: float = 1.
) -> None:
    if label is not None and color is None:
        # bug in sns: https://github.com/mwaskom/seaborn/issues/3115
        raise ValueError(f"label: {label}, color: {color}")
    #
    sns.histplot(x, bins=bins, color=color, stat=stat, kde=kde, edgecolor=edgecolor,
                 label=label, alpha=alpha, ax=ax)
    if stat == "density":
        config_ax(ax, ylabel="Density")


# if __name__ == "__main__":
#     fig, ax = get_figure_2d()
#     x = np.random.randn(100000)
#     hist(ax, x, 1000, color="C0", kde=True, label="AAA", alpha=0.5, edgecolor=None)
#     x = np.random.rand(100000)
#     hist(ax, x, 1000, color="C1", kde=True, label="BBB", alpha=0.5, edgecolor=None)
#     config_ax(ax, legend=True)
#     save_and_show("./asset/1.png")

def text(
    ax: Axes,
    x: float, y: float,  # 坐标(文字左下角与该点对其)
    s: str,
    #
    fontsize: int = 11,
    color: Color = "k",
) -> None:
    ax.text(x, y, s, fontsize=fontsize, color=color)


# if __name__ == "__main__":
#     fig, ax = get_figure_2d()
#     text(ax, 0, 1, "123aaa")
#     save_and_show()


def bar(
    ax: Axes,
    x: ArrayLike,
    height: ArrayLike,
    #
    width: float = 0.8,
    bottom: float = 0.,
    color: Color = None,
    #
    edgecolor: Color = None,
    linewidth: Optional[float] = None,
    label: Optional[str] = None,
    alpha: float = 1.
) -> None:
    """
    x: [N]
    height: [N]
    """
    ax.bar(
        x, height, width, bottom,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        label=label,
        alpha=alpha
    )

# if __name__ == "__main__":
#     fig, ax = get_figure_2d()
#     x = np.arange(5)
#     y = np.random.rand(5)
#     y2 = np.random.rand(5)
#     bar(ax, x, y, 0.6, 0.1, label="AAA", alpha=0.5)
#     bar(ax, x, y2, label="BBB", alpha=0.5)
#     config_ax(ax, legend=True, xticks=x, xticks_labels=["aaa", "bbb", "ccc", "ddd", "eee"])
#     save_and_show()


def scatter(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    #
    s: int = 30,  # default: 36
    color: Color = None,
    edgecolor: Color = "#333",
    label: Optional[str] = None,
) -> None:
    """
    x: [N]
    y: [N]
    """
    ax.scatter(
        x=x, y=y, s=s, color=color, edgecolor=edgecolor, label=label,  # ax=ax
    )  # or sns.scatterplot


# if __name__ == "__main__":
#     fig, ax = get_figure_2d()
#     x = np.random.randn(1000, 2)
#     scatter(ax, x[:, 0], x[:, 1])
#     save_and_show("asset/3.png")


def imshow(
    ax: Axes,
    x: ndarray,
    #
    cmap: Cmap = None,
    vmin: Union[None, int, float] = None,  # 0
    vmax: Union[None, int, float] = None,  # 1, 255
    origin: Literal["lower", "upper"] = "upper",
    extent: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """
    x: [H, W]. [0..1] float or [0..255] uint8. 
        or [H, W, 3] (cmap无效了), 一般伴随着axis_off
    origin: 表示0,0点在哪.
    extent: 表示下标范围. 
        含义: (x_min(left), x_max(right), y_min(bottom), y_max(top))
    """
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    if extent is None:
        extent = 0, x.shape[1], 0, x.shape[0]
        if origin == "upper":
            extent = 0, x.shape[1], x.shape[0], 0
    #
    ax.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin, extent=extent)


# if __name__ == "__main__":
#     x, y = np.arange(200), np.arange(100)
#     x, y = np.meshgrid(x, y, indexing="xy")
#     fig, ax = get_figure_2d()
#     imshow(ax, x, origin="lower")
#     save_and_show()
#     #
#     fig, ax = get_figure_2d()
#     imshow(ax, y)
#     save_and_show()


def contour(
    ax: Axes,
    x: ArrayLike, y: ArrayLike, z: ArrayLike,
    levels: List[float],
    #

) -> None:
    ax.contour(x, y, z, levels=levels)


# if __name__ == "__main__":
#     x, y = np.arange(200), np.arange(200)
#     x, y = np.meshgrid(x, y, indexing="xy")
#     z = np.sin(x + y)
#     fig, ax = get_figure_2d()
#     contour(ax, x, y, z, [0])
#     save_and_show()

def fill_between(
    ax: Axes,
    x: ArrayLike,
    y1: Union[float, ArrayLike],
    y2: Union[float, ArrayLike] = 0,  # between y1..y2(y1, y2谁大谁小无所谓)
    #
    color: Color = None,
    alpha: float = 0.2,
) -> None:
    ax.fill_between(x, y1, y2, color=color, alpha=alpha)


if __name__ == "__main__":
    x = np.arange(200)
    y = np.sin(x) + 2
    y2 = np.ones_like(y)
    fig, ax = get_figure_2d()
    fill_between(ax, x, 0, 1, alpha=0.2)
    save_and_show()
