'''The dspac_viz_density module allows plotting of density plots for different settings.

Functions:
    prep_density: Returns the df and limits that will be used in the plotting.
    plot_single_density: Plots a single density plot.
    multiplot_stp: Returns the ax to be used for the construction of the multiplot.
    overlay_cv_hull_dataframe: Plots the convex hull of a set of points.
    multiplot_kde_cvh: Creates a multiplot of convex hulls/density plots of a set of data.
'''

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.colors import to_rgba, LinearSegmentedColormap, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.cm
from design_space.dist_matrix import *
from scipy import stats, integrate


def prep_density(df, embed):
    """Returns the df and limits that will be used in the plotting.

    Args:
        df (dataframe): Main dataframe with the dump from the spreadsheet.
        embed (matrix): Full embedding with x,y points coordinates.

    Returns:
        df_kde: Dataframe for the KDE plots.
        lim_x: Limits for the x axis.
        lim_y: Limits for the y axis.
    """
    # make df_kde out of the df,embed
    df_kde = df[['ParticipantID', 'GroupID', 'PrePost', 'SolutionID', 'result', 'type']].copy()
    df_kde['x_emb'] = embed[:,0]
    df_kde['y_emb'] = embed[:,1]

    #* plot scatter plot to get limits of x, y
    fig = sns.scatterplot(data=df_kde, x="x_emb", y="y_emb", hue='ParticipantID')
    xmin, xmax = fig.get_xlim()
    ymin, ymax = fig.get_ylim()
    plt.close
    lim_x = [xmin, xmax]
    lim_y = [ymin, ymax]

    return df_kde, lim_x, lim_y

def plot_single_density(dir_data, df_dens, x_lim, y_lim, lvls=None, cmapn=None, figtitle=None, fn=None):
    """Plots a single density plot.

    Args:
        dir_data (path): Path to the directory with the data file.
        df_dens (dataframe): Dataframe with the KDE plot data.
        x_lim (list): Limits for the x axis.
        y_lim (list): Limits for the y axis.
        lvls (integer, optional): Number of levels for the KDE plot. Defaults to 100.
        cmapn (string, optional): Colormap to be used. Defaults to inferno.
        figtitle (string, optional): Title of the figure. Defaults to None.
        fn (string, optional): Name of the file that will be saved. Defaults to None.
    """
    dir_exviz = Path(f'{dir_data.parent}'+r'/experimental/viz')

    # arguments defaults
    if lvls is None:
        lvls = 100

    if cmapn is None:
        cmapn = 'inferno'

    if figtitle is None:
        figtitle = 'Density plot for the full DS'

    if fn is None:
        fn = f'EX_Density_all_{cmapn}.png'

    fig, axes = plt.subplots(layout='constrained', figsize=(15, 15))

    plot = sns.kdeplot(data=df_dens, x="x_emb", y="y_emb", fill=True, levels=lvls, cmap='inferno')
    sns.scatterplot(data=df_dens, x="x_emb", y="y_emb", hue='ParticipantID')
    plot.set_xlim(x_lim)
    plot.set_ylim(y_lim)
    plt.legend([],[], frameon=False)
    plt.title(f'{figtitle}', fontsize=14, fontweight='bold')
    axes.set(xlabel=None, ylabel=None)
    plt.savefig(f'{dir_exviz}/{fn}', dpi=300, bbox_inches='tight')
    plt.close()

def multiplot_stp(x_lim, y_lim):
    """Returns the ax to be used for the construction of the multiplot.

    Args:
        x_lim (list): Limits for the x axis.
        y_lim (list): Limits for the y axis.

    Returns:
        axbase: Axis that wil be used to construct the multiplot.
    """
    fig1, axbase = plt.subplots(layout='constrained')
    axbase.set_xlim(x_lim)
    axbase.set_ylim(y_lim)
    axbase.set_xticks([])
    axbase.set_yticks([])
    return axbase

def overlay_cv_hull_dataframe(x, y, color, splot, alpha, vtx=False):
    """Plots the convex hull of a set of points.

    Args:
        x (list): List of x coordinates.
        y (list): List of y coordinates.
        color (string): Name/hex code of the color the convex hull should take.
        splot (sns plot object): Plot in which the convex hull should be drawn.
        alpha (float): Transparency of the convex hull.
        vtx (bool, optional): Argument to especify if vertices of the convex hull should be returned. Defaults to False.

    Returns:
        x_vtx: List of x coordinates for the convex hull vertices.
        y_vtx: List of y coordinates for the convex hull vertices.
    """
    points = np.array(list(zip(x,y)))
    hull = ConvexHull(points)
    splot.fill(points[hull.vertices, 0], points[hull.vertices, 1],
    facecolor=to_rgba(color, alpha),
    edgecolor=color)

    x_vtx = points[hull.vertices, 0]
    y_vtx = points[hull.vertices, 1]

    if vtx == True:
        return x_vtx, y_vtx

def multiplot_kde_cvh(dir_data, df_dens, df_color, x_lim, y_lim, nrows, ncols, axbase, param=None, lvls=None, cmapn=None, kde=True, cvxh=True, precvh=False, postcvh=False, allscatter=False, figtitle=None, fn=None):
    """Creates a multiplot of convex hulls/density plots of a set of data.

    Args:
        dir_data (path): Path to the directory with the data file.
        df_dens (dataframe): Dataframe with the KDE plot data.
        df_color (dataframe): Dataframe with color scheme for the participants.
        x_lim (list): Limits for the x axis.
        y_lim (list): Limits for the y axis.
        nrows (integer): Number of rows for the multiplot.
        ncols (integer): Number of columns for the multiplot.
        axbase (axis object): Axis to format each cell of the
        param (string, optional): WHat is the data to be used for the plot. Defaults to ParticipantID.
        lvls (integer, optional): Number of levels for the KDE plot. Defaults to 100.
        cmapn (string, optional): Colormap to be used. Defaults to inferno.
        kde (bool, optional): Argument to determine if kde plot should be created. Defaults to True.
        cvxh (bool, optional): Argument to determine if convex hull plots should be created. Defaults to True.
        precvh (bool, optional): Argument to determine if convex hull pre intervention should be created. Defaults to False.
        postcvh (bool, optional): Argument to determine if convex hull post intervention should be created. Defaults to False.
        allscatter (bool, optional): Argument to determine if all points should be added to the plots. Defaults to False.
        figtitle (string, optional): Title of the figure. Defaults to None.
        fn (string, optional): Name of the file that will be saved. Defaults to None.
    """
    dir_exviz = Path(f'{dir_data.parent}'+r'/experimental/viz')
    sns.set_theme(style="white")

    # arguments defaults
    if param is None:
        param='ParticipantID'

    if lvls is None:
        lvls = 100

    if cmapn is None:
        cmapn = 'inferno'

    if figtitle is None:
        figtitle = 'Multiplot for the full DS'

    if fn is None:
        fn = f'EX_Density_all_{cmapn}.png'

    # multiplot regions
    if nrows <= 1:
        fig, axes = plt.subplots(ncols=ncols, layout='constrained', figsize=(30, 13), sharey=axbase, sharex=axbase)
    else:
        fig, axes = plt.subplots(nrows, ncols, layout='constrained', figsize=(30, 26), sharey=axbase, sharex=axbase)

    # get ids list for the parameter that will be plotted
    ids = list(df_dens[f'{param}'].unique())
    nplots = nrows * ncols
    if len(ids) > nplots:
        ids = ids[:nplots]


    # plot kde
    print('Multiplot started...')
    for (n, axi) in zip(range(len(ids)), axes.flat):
        dfsubset = df_dens[df_dens[f'{param}'] == ids[n]].copy()

        # colors if participant focus/others
        if param == 'ParticipantID':
            pt_color = df_color.loc[n,'HEX-Win']
        else:
            pt_color = 'red'

        # create kde
        if kde == False:
            plot = sns.scatterplot(data=dfsubset, x="x_emb", y="y_emb", ax=axi, color=pt_color)
        else:
            plot = sns.kdeplot(data=dfsubset, x="x_emb", y="y_emb", fill=True, levels=lvls, cmap=cmapn, ax=axi)


        # adding convex hull
        if cvxh == True:
            # participant convex hull
            overlay_cv_hull_dataframe(dfsubset["x_emb"], dfsubset["y_emb"], pt_color, plot, 0.25)

            # full convex hull
            x_vtx, y_vtx = overlay_cv_hull_dataframe(df_dens["x_emb"], df_dens["y_emb"], 'k', plot, 0.05, vtx=True)
            df_vtx = pd.DataFrame(np.array(list(zip(x_vtx,y_vtx))), columns=['x_vt', 'y_vt'])
            sns.scatterplot(data=df_vtx, x="x_vt", y="y_vt", ax=axi, color='k')

            if precvh == True:
                dfpre = dfsubset[dfsubset['PrePost'] == 'Pre'].copy()
                if len(dfpre) > 3:
                    x_pre = dfpre['x_emb']
                    y_pre = dfpre['y_emb']
                    overlay_cv_hull_dataframe(x_pre, y_pre, pt_color, plot, 0.15)

            if postcvh == True:
                dfpost = dfsubset[dfsubset['PrePost'] == 'Pst'].copy()
                if len(dfpost) > 3:
                    x_post = dfpost['x_emb']
                    y_post = dfpost['y_emb']
                    overlay_cv_hull_dataframe(x_post, y_post, pt_color, plot, 0.15)

        # adding scatter points for the subset
        sns.scatterplot(data=dfsubset, x="x_emb", y="y_emb", ax=axi, color=pt_color)

        # adding scatter for all points
        if allscatter == True:
            sns.scatterplot(data=df_dens, x="x_emb", y="y_emb", ax=axi, color='.8', marker='+')

        plot.set_xlim(x_lim)
        plot.set_ylim(y_lim)

        axi.title.set_text(f'{ids[n]}')
        axi.set(xlabel=None, ylabel=None, adjustable='box', aspect='equal')

    fig.suptitle(f'{figtitle}', fontsize=14, fontweight='bold')
    plt.savefig(f'{dir_exviz}/{fn}', dpi=300, bbox_inches='tight')
    plt.close()
    print('Multiplot done!')

def create_cmap(colorname, rev=True):
    """Returns colormap based on colorname [color --> white].

    Args:
        colorname (string): Color name or string with hex code.
        rev (bool, optional): If color map should be reversed. Defaults to True.

    Returns:
        newcmp: Colormap that gets returned (color --> white)
    """
    clr = mcolors.to_rgb(colorname)
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(clr[0], 1, N)
    vals[:, 1] = np.linspace(clr[1], 1, N)
    vals[:, 2] = np.linspace(clr[2], 1, N)

    cmp = ListedColormap(vals)

    if rev == True:
        newcmp = ListedColormap(cmp.colors[::-1])
    else:
        newcmp = cmp

    return newcmp

def bl_cmp(colorname, rev=True):
    """Returns a colormap based on colorname [color --> black].

    Args:
        colorname (string): Color name or string with hex code.
        rev (bool, optional): If color map should be reversed. Defaults to True.

    Returns:
        newcmp: Colormap that gets returned (color --> black)
    """
    clr = mcolors.to_rgb(colorname)

    if rev == True:
        colors = [(0, 0, 0), clr]
    else:
        colors = [clr,(0, 0, 0)]

    cm = LinearSegmentedColormap.from_list('test', colors, N=256)

    return cm

def gauplot(centers, radiuses, axs, cmp, bg='white', xr=None, yr=None):
    """Plots the gradient discs on the axis of the multiplot.

    Args:
        centers (list): List of x,y values for the centers of the discs.
        radiuses (list): List with with radius lenght for each circle (avg dist between solutions).
        axs (axis): Axis of matplotlib object where discs should be plotted.
        cmp (colormap): Colormap for the gradient.
        bg (str, optional): Background of the dubplots/figure. Defaults to 'white'.
        xr (list, optional): Limits of the x axis. Defaults to None.
        yr (_type_, optional): Limits of the y axis. Defaults to None.
    """
    # creates grid of 1000 x 1000 from the axis limits
    nx, ny = 1000.,1000.
    xgrid, ygrid = np.mgrid[xr[0]:xr[1]:(xr[1]-xr[0])/nx,yr[0]:yr[1]:(yr[1]-yr[0])/ny]

    # initiate image/vectors
    im = xgrid*0 + np.nan
    xs = np.array([np.nan])
    ys = np.array([np.nan])
    fis = np.concatenate((np.linspace(-np.pi,np.pi,100), [np.nan]))

    # colormap settigns
    cmap = cmp
    cmap.set_bad(f'{bg}')
    thresh = 3

    # render discs on centers, with radiuses
    for curcen,currad in zip(centers,radiuses):
            curim=(((xgrid-curcen[0])**2+(ygrid-curcen[1])**2)**.5)/currad*thresh
            im[curim<thresh]=np.exp(-.5*curim**2)[curim<thresh]
            xs = np.append(xs, curcen[0] + currad * np.cos(fis))
            ys = np.append(ys, curcen[1] + currad * np.sin(fis))

    # add to axis
    axs.imshow(im.T, cmap=cmap, extent=xr+yr, origin='lower', aspect=10)
    axs.plot(xs, ys,lw=0)
    axs.autoscale(False)

def multiplot_discs(dir_data, df_dens, df_distmetric, df_color, x_lim, y_lim, nrows, ncols, axbase, bg=None, cvxh=True, precvh=False, postcvh=False, allscatter=False, figtitle=None, fn=None):
    """Creates a multiplot adding gradient discs to each point of the participant.

    Args:
        dir_data (path): Path to the directory with the data file.
        df_dens (dataframe): Dataframe with the KDE plot data.
        df_distmetric (dataframe): Dataframe with distmetrics for each participant.
        df_color (dataframe): Dataframe with color scheme for the participants.
        x_lim (list): Limits for the x axis.
        y_lim (list): Limits for the y axis.
        nrows (integer): Number of rows for the multiplot.
        ncols (integer): Number of columns for the multiplot.
        axbase (axis object): Axis to format each cell of the
        bg (str, optional): Background of the dubplots/figure. Defaults to 'white'.
        cvxh (bool, optional): Argument to determine if convex hull plots should be created. Defaults to True.
        precvh (bool, optional): Argument to determine if convex hull pre intervention should be created. Defaults to False.
        postcvh (bool, optional): Argument to determine if convex hull post intervention should be created. Defaults to False.
        allscatter (bool, optional): Argument to determine if all points should be added to the plots. Defaults to False.
        figtitle (string, optional): Title of the figure. Defaults to None.
        fn (string, optional): Name of the file that will be saved. Defaults to None.
    """

    dir_exviz = Path(f'{dir_data.parent}'+r'/experimental/viz')
    sns.set_theme(style="white")

    if bg is None:
        bg = 'white'

    if figtitle is None:
        figtitle = 'Multiplot with Discs (r=avg dist) on Points'

    if fn is None:
        fn = f'EX_Discs.png'

    print('Multiplot Discs...')
    # multiplot regions
    fig, axes = plt.subplots(nrows, ncols, layout='constrained', figsize=(30, 26), sharey=axbase, sharex=axbase)

    # get ids list for the parameter that will be plotted
    ids = list(df_dens['ParticipantID'].unique())
    nplots = nrows * ncols
    if len(ids) > nplots:
        ids = ids[:nplots]

    for (n, axi) in zip(range(len(ids)), axes.flat):
        if n < len(ids)-1:
            # x & y (center of)
            dfsubset = df_dens[df_dens['ParticipantID'] == ids[n]].copy()
            x_pt = dfsubset['x_emb'].tolist()
            y_pt = dfsubset['y_emb'].tolist()

            # color of points
            pt_color = df_color.loc[n,'HEX-Win']
            cmp_col = df_color.loc[n,'HEX-Fail']


            print(df_distmetric)

            # radius
            avg_dist = df_distmetric.loc[n,'avgdist_FS']
            n_sol = len(x_pt)
            radis = [avg_dist] * n_sol

            # ptcmap = bl_cmp(cmp_col) #(black cmap)
            ptcmap = create_cmap(cmp_col)

            gauplot(np.array(list(zip(x_pt,y_pt))), radis, axi, ptcmap, bg=bg, xr=x_lim, yr=y_lim)

            splot = sns.scatterplot(data=dfsubset, x="x_emb", y="y_emb", ax=axi, color=pt_color)

            if cvxh == True:
                overlay_cv_hull_dataframe(x_pt, y_pt, pt_color, splot, 0.25)

                # full convex hull
                x_vtx, y_vtx = overlay_cv_hull_dataframe(df_dens["x_emb"], df_dens["y_emb"], 'k', splot, 0.05, vtx=True)
                df_vtx = pd.DataFrame(np.array(list(zip(x_vtx,y_vtx))), columns=['x_vt', 'y_vt'])
                sns.scatterplot(data=df_vtx, x="x_vt", y="y_vt", ax=axi, color='k')

                if precvh == True:
                    dfpre = dfsubset[dfsubset['PrePost'] == 'Pre'].copy()
                    if len(dfpre) > 3:
                        x_pre = dfpre['x_emb']
                        y_pre = dfpre['y_emb']
                        overlay_cv_hull_dataframe(x_pre, y_pre, pt_color, splot, 0.15)

                if postcvh == True:
                    dfpost = dfsubset[dfsubset['PrePost'] == 'Pst'].copy()
                    if len(dfpost) > 3:
                        x_post = dfpost['x_emb']
                        y_post = dfpost['y_emb']
                        overlay_cv_hull_dataframe(x_post, y_post, pt_color, splot, 0.15)

            if allscatter == True:
                sns.scatterplot(data=df_dens, x="x_emb", y="y_emb", ax=axi, color='.75', marker='+')

            axi.title.set_text(f'{ids[n]}')
            axi.set(xlabel=None, ylabel=None, adjustable='box', aspect='equal')

    fig.suptitle(f'{figtitle}', fontsize=14, fontweight='bold')
    plt.savefig(f'{dir_exviz}/{fn}', dpi=300, bbox_inches='tight')
    plt.close()
    print('Multiplot Discs - done!')

def normalize(arr, t_min, t_max):
    """Returns normalised array given a new set of min/max values.

    Args:
        arr (array): Array to be normalised
        t_min (float): New minimum value.
        t_max (float): New maximum value.

    Returns:
        norm_arr: Normalised array.
    """
    # explicit function to normalize array
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def novelty_from_density(dir_data, df_dens, x_lim, y_lim, res=1000j, prt_integral=True, plot=False, save_metrics=True):
    """Returns dataframe with novelty scores calculated from the density parameter of the KDE.

    Args:
        dir_data (path): Path to the directory with the data file.
        df_dens (dataframe): Dataframe with the KDE plot data.
        x_lim (list): Limits for the x axis.
        y_lim (list): Limits for the y axis.
        res (imaginary, optional): Value of the grid size. Defaults to 1000j.
        prt_integral (bool, optional): Argument to determine if integrals are calculated and printed. Defaults to True.
        plot (bool, optional): Argument to determine if KDE is plotted. Defaults to False.
        save_metrics (bool, optional): Argument to determine if dataframe with metrics is plotted. Defaults to True.

    Returns:
        df_dens: Dataframe with column with novelty scores.
    """
    dir_exviz = Path(f'{dir_data.parent}'+r'/experimental/viz')

    # create grid
    xgrid, ygrid = np.mgrid[x_lim[0]:x_lim[1]:res,y_lim[0]:y_lim[1]:res]
    pos = np.vstack([xgrid.ravel(), ygrid.ravel()])

    # create values
    x_pt = df_dens["x_emb"]
    y_pt = df_dens["y_emb"]
    vals =np.vstack([x_pt, y_pt])

    # get kde function describing values
    kernel_func = stats.gaussian_kde(vals)

    # get array with density values at specific points
    dens_arr = kernel_func.evaluate(vals)

    # novelty array (1/density)
    novelty_arr = 1/dens_arr

    # rescaling novelty arr [0,1]
    n_novelty = normalize(novelty_arr,0,1)

    df_dens['novelty_norm'] = n_novelty             #! novelty of each solution, based on density value of the point

    # create fuction to apply to any (x,y)
    f = lambda x,y: kernel_func((x, y))

    if prt_integral == True:
        print('Calculating integral...')
        integ = integrate.nquad(f, [x_lim, y_lim])[0]
        print(f'Double integral of KDE function over area delimited by x{x_lim}, y{y_lim}: {integ:.3f}')

    if plot == True:
        Z = np.reshape(kernel_func(pos).T, xgrid.shape)
        fig, ax = plt.subplots(figsize=(15,15), layout='constrained')
        ax.imshow(np.rot90(Z), cmap=plt.cm.inferno,
                    extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        ax.plot(x_pt, y_pt, 'k.', markersize=5)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        fig.suptitle(f'Density Plot KDE to Novelty', fontsize=14, fontweight='bold')
        plt.savefig(f'{dir_exviz}/Density_KDE_Novelty', dpi=300, bbox_inches='tight')
        plt.close()

        df_topten = df_dens.loc[df_dens['novelty_norm'] >= 0.9].copy()
        df_botten = df_dens.loc[df_dens['novelty_norm'] <= 0.1].copy()

        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")

        fig, axes = plt.subplots(layout='constrained', figsize=(10, 10))
        plot = sns.kdeplot(data=df_dens, x="x_emb", y="y_emb", fill=True, levels=200, cmap='mako')
        sns.scatterplot(data=df_dens, x="x_emb", y="y_emb", color='.8', marker='+', s=50)
        sns.scatterplot(data=df_topten, x="x_emb", y="y_emb", color='red', marker='P', s=75)
        sns.scatterplot(data=df_botten, x="x_emb", y="y_emb", color='k', marker='X', s=75)
        plot.set_xlim(x_lim)
        plot.set_ylim(y_lim)

        plt.legend([],[], frameon=False)
        plt.title('Density plot with pts with novelty > 0.9 (red, +), < 0.1 (black, X) for the full DS', fontsize=14, fontweight='bold')
        axes.set(xlabel=None, ylabel=None)
        plt.savefig(f'{dir_exviz}/EX_Novelty_all_mako.png', dpi=300, bbox_inches='tight')


        print('KDE plot - done!')

    if save_metrics == True:
        with pd.ExcelWriter(f'{dir_exviz}/DS_NoveltyDensity_Metric.xlsx') as writer:
                df_dens.to_excel(writer, sheet_name="NoveltyfromDensity", index=False)

    return df_dens

