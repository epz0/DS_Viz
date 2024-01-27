#%%
import numpy as np
from scipy.spatial import Delaunay
from pathlib import Path
from design_space.read_data import read_analysis
from design_space.dist_matrix import *
from design_space.dim_reduction import *
from design_space.dspace_dist_metrics import *
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


from dash import Dash, dcc, html, Input, Output, no_update
#import dash_daq as daq
import plotly.graph_objects as go
import plotly.io as pio

#* --- initial definitions ------
my_dir = Path(r'C:/Py/DS_Viz/data')                                            # path to the data file
filenm = 'dataset.xlsx'                                                     # name of the data file
sheetnm = 'data'                                                            # sheet where the data (analysis) is
#* ------------------------------

#* --- create folder structure ---
Path(f'{my_dir.parent}/experimental/viz').mkdir(parents=True,exist_ok=True)             # folder export/stats
#* ------------------------------

dir_exviz = Path(f'{my_dir.parent}'+r'/experimental/viz')

#* reads Excel file with solutions' analysis
df_base, df_colors, labels = read_analysis(my_dir, filenm, sheetname=sheetnm)

#* calculates distance matrix
n_distmatrix = calc_distmatrix(df_base, my_dir, filenm)

#* generate the embedding of the distance matrix
embedding, graph = create_embedding(my_dir, n_distmatrix)

def cv_hull_vertices(x, y):
    """Retrns convex hull metrics and vertices.

    Args:
        x (list): List of x coordinates.
        y (list): List of y coordinates.

    Returns:
        x_vtx: List of x coordinates for the convex hull vertices.
        y_vtx: List of y coordinates for the convex hull vertices.
        cvxh_area: Area of the convex hull.
    """
    points = np.array(list(zip(x,y)))
    hull = ConvexHull(points)

    x_vtx = points[hull.vertices, 0]
    y_vtx = points[hull.vertices, 1]

    cvxh_area = hull.volume

    return x_vtx, y_vtx, cvxh_area

#* prepare data for the cvx hull calc
df_base['x_emb'] = embedding[:, 0]
df_base['y_emb'] = embedding[:, 1]
ids = df_base['ParticipantID'].unique()

df_col = df_colors[['P','HEX-Win']].copy()
df_base = df_base.merge(df_col, left_on='ParticipantID', right_on='P').copy()

#* create base figure
pio.renderers.default = "browser"

#* first trace: Full Design space
fullds_xvt, fullds_yvt, cvarea = cv_hull_vertices(x=df_base['x_emb'], y=df_base['y_emb'])


fig = go.Figure()

# convexhull full DS
fig.add_trace(go.Scatter(   x=fullds_xvt,
                            y=fullds_yvt,
                            mode='lines+markers',
                            fill='toself',
                            marker=dict(
                                size=8,
                                color='white'
                            ),
                            name='Full DS'),
                            )

fig.update_layout(
    autosize=False,
    width=950,
    height=800,
    margin=dict(
        l=25,
        r=25,
        b=25,
        t=25,
        pad=2
    ),
    paper_bgcolor="LightSteelBlue",)

for i in range(len(ids)-1):
    dfsubset = df_base[df_base['ParticipantID'] == ids[i]].copy()
    print(dfsubset)
    #dfsubset.sort_values(by='OriginalID_Sol')

    pt_xvt, pt_yvt, cvarea = cv_hull_vertices(x=dfsubset['x_emb'], y=dfsubset['y_emb'])

    # adding convex hull trace for each participant
    fig.add_trace(go.Scatter(   x=pt_xvt,
                                y=pt_yvt,
                                mode='lines+markers',
                                marker=dict(symbol='circle-open-dot',
                                            color=df_col.loc[i, 'HEX-Win'],
                                            size=8,
                                            ),
                                opacity=.45,
                                fill='toself',
                                name=f'{ids[i]}_CVH')
                                )

    # adding arrows trace for each participant
    fig.add_trace(go.Scatter(   x=dfsubset['x_emb'],
                                y=dfsubset['y_emb'],
                                mode='lines+markers',
                                marker= dict(   symbol="arrow-up",
                                                size=8,
                                                angleref="previous",
                                                color=df_col.loc[i, 'HEX-Win']),
                                name=f'{ids[i]}_ARR')
                                )

    # adding points trace for each participant
    fig.add_trace(go.Scatter(   x=dfsubset['x_emb'],
                                y=dfsubset['y_emb'],
                                mode='markers',
                                marker= dict(   symbol=0,
                                                size=5,
                                                color=df_col.loc[i, 'HEX-Win']),
                                name=f'{ids[i]}_PTS')
                                )

# all points
fig.add_trace(go.Scatter(   x=embedding[:, 0],
                            y=embedding[:, 1],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=df_base['HEX-Win']
                            ),
                            name='All Points'),
                            )

btn = ['PTS', 'ARR', 'CVH']
btlb = ['Points', 'Arrows', 'Convex Hull']
buttons = []
traces = fig.data

for bt in btn:
    buttons.append(dict(method='restyle',
                        label=bt,
                        visible=True,
                        args=[{'visible':True}, [i for i,x in enumerate(traces) if x.name[-3:]==bt]],
                        args2=[{'visible':'legendonly'},[i for i,x in enumerate(traces) if x.name == bt]]))

allButton = [
    dict(
        method='restyle',
        label='All',
        visible=True,
        args=[{'visible':True}],
        args2=[{'visible':'legendonly'}]
    )
]

# create the layout
layout = go.Layout(
    width=850,
    height=800,
    margin=dict(
        l=25,
        r=25,
        b=25,
        t=25,
        pad=2
    ),
    updatemenus=[
        dict(
            type='buttons',
            direction='right',
            x=0.45,
            y=1.05,
            showactive=True,
            buttons=allButton + buttons
        )
    ],
    title=dict(text='Toggle Traces',x=0.65),
    showlegend=True
)

fig1 = go.Figure(data=traces, layout=layout)

# add dropdown menus to the figure
#fig1.show()


#%%
# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
fig1.update_traces(hoverinfo="none", hovertemplate=None)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph-basic-2", figure=fig1, clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
])


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)


def display_hover(hoverData):

    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]

    if "pointNumber" in pt:
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        #! HERE

        df_ptc_row = df_base.iloc[num]
        img_src = df_ptc_row['video']
        name = df_ptc_row['FullID']
        form = df_ptc_row['result']
        desc = f"$ {df_ptc_row['budgetUsed']:.1f} + Max Stress: {df_ptc_row['maxStress']:.1f}"
        if len(desc) > 300:
            desc = desc[:100] + '...'

        children = [
            html.Div([
                html.Img(src=img_src, style={"width": "100%"}),
                html.H2(f"{name}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
                html.P(f"{form}"),
                html.P(f"{desc}"),
            ], style={'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children
    else:
        return False, no_update, no_update

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

# %%
