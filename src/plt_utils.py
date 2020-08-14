import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np

data_size = 0

def div_10(x, *args):
    x = float(x)/data_size
    return "{:.1f}".format(x)


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def add_plots(fig, counter):
    ax_inv = fig.add_subplot(111, frameon=False)
    ax_inv.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_inv.set_xticks([])
    ax_inv.set_yticks([])
    if counter == 4:
        return fig.add_subplot(141), fig.add_subplot(142), fig.add_subplot(143), fig.add_subplot(144)
    elif counter == 3:
        return fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
    elif counter == 2:
        return fig.add_subplot(121), fig.add_subplot(122)
    else:
        return ax_inv


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def add_scatter(ax, gt_len, data, color, limit):
    pos = len(([1 for i in data if i < limit]))
    if (pos/gt_len > 0.15):
        ax.scatter(limit, pos, marker='.', color=color, zorder=3) 


def add_limit_2(ax, limit, methods):
    font_size_y = 0.015
    y_offset = -font_size_y / 2
    xmin, xmax = ax.get_xlim()
    text_data = {}
    x_offset = xmax * 0.03
    for l in limit:
        ax.axvline(x=l, color='#CCCCCC', linestyle='dashed', linewidth=0.5, zorder=2)
    counter = 0
    for h in methods: 
        for v in h:
            if v not in text_data:
                text_data[v] = {}
            text_data[v][h[v]+y_offset] = [v+x_offset, '{:.2f}'.format(h[v])]
        counter += 1
    for line in text_data:
        last_y = 0
        for y in sorted(text_data[line]):
            y_pos = y
            distance = y_pos - last_y
            if (distance < 0.06):
                y_pos = last_y + 0.06
            last_y = y_pos
            ax.text(text_data[line][y][0], y_pos, text_data[line][y][1], fontsize=9, bbox=dict(boxstyle='square,pad=0',edgecolor='white', facecolor='white', alpha=0.5))


def add_limit(ax, len_gt, limit, methods, data, min_limit, y_max=1):
    max_y = len_gt*y_max
    font_size_y = max_y * 0.015
    y_offset = -font_size_y / 2
    xmin, xmax = ax.get_xlim()
    ax.axvline(x=limit, color='#CCCCCC', linestyle='dashed', linewidth=0.5, zorder=2)
    x_offset = xmax * 0.03 
    text_data = {}
    for method in methods:
        hs_len = len([1 for i in methods[method][data] if i < limit])
        if (hs_len/len_gt > min_limit):
            text_data[hs_len+y_offset] = [limit+x_offset, '{:.2f}'.format(hs_len/len_gt)]
    last_y = 0
    for text in sorted(text_data):
        fs = font_size_y * 4
        y_pos = text
        distance = y_pos - last_y
        if (distance < fs):
            y_pos = last_y + fs
        last_y = y_pos
        t = ax.text(text_data[text][0], y_pos, text_data[text][1], fontsize=9, bbox=dict(boxstyle='square,pad=0',edgecolor='white', facecolor='white', alpha=0.5))
