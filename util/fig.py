from pathlib import Path
from matplotlib import pyplot as plt

plt.rcParams["axes.axisbelow"] = True

def graph_plot(
    *plots,
    figsize=None,
    title=None,
    xlabel=None,
    ylabel=None,
    tick=True,
    grid=True,
    xlim=None,
    ylim=None,
    legend=None,
    legend_loc="best",
    savefig_path=None,
    close=False,
    show=False,
):
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)


    for p in plots:
        p()
    
    graph_settings(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        tick=tick,
        grid=grid,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        legend_loc=legend_loc,
        savefig_path=savefig_path,
        close=close,
        show=show,
    )


def graph_settings(
    title=None,
    xlabel=None,
    ylabel=None,
    tick=True,
    grid=True,
    xlim=None,
    ylim=None,
    legend=None,
    legend_loc="best",
    savefig_path=None,
    close=False,
    show=False,
):
    if title is not None:
        plt.title(title, fontsize=16)
    
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=16)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)

    if tick:
        plt.tick_params(labelsize=14)

    if grid:
        plt.grid()

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    if legend is not None:
        plt.legend(legend, loc=legend_loc, fontsize=14)

    if savefig_path is not None:
        Path.mkdir(Path(savefig_path).parent, parents=True, exist_ok=True)
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0.2)
    
    if show:
        plt.show()
        
    if close:
        plt.close()