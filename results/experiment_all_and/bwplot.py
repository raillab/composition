import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np

files = ['rewards_all_and.txt','../experiment_approx_and/rewards_or.txt', '../experiment_approx_and/rewards_and.txt', '../experiment_approx_and/rewards_blue_crate.txt']
names = ['Baseline', 'Purple', 'Blue', 'Or']
colours = ['#9467bd', '#1f77b4', '#FF4136']
data = [np.loadtxt(file) for file in files]

boxes = [
    go.Box(
        y=d,
        name=names[i],
        showlegend=False,
        marker=dict(
            color='rgb(255,255,255, 0)'
        ),
        line=dict(
            width=10,
            color=colours[i]
        )
    ) for i, d in enumerate(data)
]

layout = go.Layout(
    yaxis=dict(
        title='Average Return',
        range=[-0.2, 1.05],
        showgrid=False,
        zeroline=False,
        showline=True,
        automargin=True,
        linewidth=6,
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        automargin=True
    ),
    font=dict(family='Times New Roman', size=56)

)

fig = go.Figure(data=boxes, layout=layout)
offline.plot(fig, filename="Box Plot Styling Outliers.html", image='svg', image_width=800, image_height=800)
