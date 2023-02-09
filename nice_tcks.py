__author__ = 'roman'
from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from matplotlib.transforms import Bbox

def nice_ticks():
    ax = subplot(111)

    ax.get_xaxis().set_tick_params(direction='in', bottom=1, top=1)
    ax.get_yaxis().set_tick_params(direction='in', left=1, right=1)
    #ax.xaxis.set_major_locator(MultipleLocator(MLs[fig][0]))
    #ax.xaxis.set_minor_locator(MultipleLocator(MLs[fig][1]))

    #ax.yaxis.set_major_locator(MultipleLocator(MLs[fig][2]))
    #ax.yaxis.set_minor_locator(MultipleLocator(MLs[fig][3]))
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(10)
        l.set_markeredgewidth(2.0)
    for l in ax.yaxis.get_minorticklines() + ax.xaxis.get_minorticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(1.5)

    ax.set_position(Bbox([[0.16, 0.16], [0.95, 0.95]]))