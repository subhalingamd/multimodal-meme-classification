''' Set of functions for plotting results of using classes in
    radial_split.py & color_density.py '''

import numpy as np
import matplotlib.pyplot as plt


def plot_radial_splits(img_file, rs, ax, color='r'):
    '''Displays an image with the radial split overlay

        Parameters
        ----------
        img_file: str
            Path to image file
        rs: RadialSplitter object
            Instance of the RadialSplitter class from radial_split.py
        ax: matplotlib Axes
            Axes in which to display the image
        color: str or tuple
            Color for overlay lines (argument for matplotlib functions)
    '''

    img = plt.imread(img_file)  # read image file
    _ = rs.split(img)  # apply RadialSplitter to generate vertices
    vs = rs.vertices[:, :, ::-1].copy()  # get vertices of splits

    for y in range(rs.nqslices * 4 + 1):  # loop that will cover every segment
        for r in range(rs.nrings):
            ax.plot(*[vs[r:r + 2, y, i]
                      for i in [0, 1]], color=color, linewidth=2)
            ax.plot(*[vs[r, y:y + 2, i]
                      for i in [0, 1]], color=color, linewidth=2)

    for a in ['bottom', 'top', 'left', 'right']:  # color borders
        ax.spines[a].set_color(color)
        ax.spines[a].set_linewidth(2)

    ax.set_xticks([])
    ax.set_yticks([])  # remove ticks
    ax.imshow(img)  # add image


def plot_color_histogram(img_file, cd, ax, color='r', annot=None, annot_size=10):
    '''Displays color histogram from an image

        Parameters
        ----------
        img_file: str
            Path to image file
        cd: ColorDensity object
            Instance of the ColorDensity class from color_density.py
        ax: matplotlib Axes
            Axes in which to display the image
        color: str or tuple
            Color for vertical lines and annot
            (argument for matplotlib functions)
        annot: None or iterable
            If not None, must be an iterable of equal length of
            the nsegs attribute of cd.
        annot_size: int
            Fontsize for annotation
    '''

    hist = cd.fit_transform([img_file])[0]  # get histogram
    bps = int(len(hist) / cd.nsegs)  # bars per segment
    cmap = bar_color_map(cd.n_bins)  # colormap for bars

    bar_cmap = []  # need to create cmap that matches hist length
    for n in range(cd.nsegs):
        bar_cmap.extend(cmap)  # duplicate cmap for each segment
        bar_cmap.append([0, 0, 0])  # add dummy zero for spacing

    y_val = []  # need to extend hist values to allow spaceing
    for i in range(cd.nsegs):
        y_val.extend(hist[i * bps:(i + 1) * bps])  # this segment of hist
        y_val.append(0)  # add dummy zero for spacing

    # extend by nsegs to account for spaces we added
    x_val = range(len(hist) + cd.nsegs)

    bars = ax.bar(x=x_val, height=y_val)  #  plot bars
    for i in range(1, cd.nsegs):  # add vertical lines at spacing
        ax.axvline((i * bps) + ((i - 1) * 1), linestyle=':', color=color)

    for idx, b in enumerate(bars):  # for each bar
        b.set_color(bar_cmap[idx])  # apply correct color
        b.set_edgecolor('k')  # add edge color

    # remove ticks, set limit, add axis labels
    ax.set_xticks([])
    ax.set_xlim(-1, len(x_val))
    ax.set_xlabel('Color bin')
    ylab = 'Density' if cd.nsegs == 1 else 'Density within segment'
    ax.set_ylabel(ylab)

    if cd.nsegs > 1:
        y = ax.get_ylim()[1] * 0.95  # where to add annotation
        offset = ax.get_xlim()[1] / (2 * cd.nsegs)
        for i in range(cd.nsegs):
            x = (i * bps) + ((i - 1) * 1) + offset  # where to add annotation
            s = 'Segment {}'.format(i + 1)  # label
            if annot is not None:  # if extra description supplied
                s += '\n' + annot[i]
            ax.text(x=x, y=y, s=s, color=color, fontsize=annot_size,
                    va='top', ha='center')


def bar_color_map(nbins=3):
    ''' Get list of RGB color shades for color histogram plot

        Parameters
        ----------
        nbins: int
            the number of bins used to generate histograms
            (i.e. n_bins argument into the ColorDensity object used)

        Returns
        -------
        cmap: numpy array
            Array of shape (nbins**3, 3), where each row contains the
            numeric definition of RGB color shade
    '''
    step = 1 / nbins  # steps to take
    # we start by defining the midpoint of each bin
    perc = np.arange(step / 2, 1 + (step / 2), step=step)
    # perform loop to get the RGB values for the nbins**3 shades
    cmap = np.array([[a, b, c] for a in perc for b in perc for c in perc])
    return cmap
