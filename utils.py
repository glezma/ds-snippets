
#  Importing libraries
import pandas as pd
import numpy as np
import numpy.random as npr
import matplotlib
import pandas_profiling
import plotly.graph_objects as go

def plot_fanchart(df, title= 'Titulo de grafico',dot_name='Real',line_name='Teórica', 
                    xLabel='x', yLabel='y',
                    colorListRGB=[]):

    '''

    df : Dataframe is assumed to have the following columns
        Column 1 : Corresponds to the x axis values
        Column 2 : Series to plot using dots
        Column 3 :  Series to plot using a line
        Column 4 to N: N should be an odd number. Column names from column 4 to N are taken
        as label names for each interval of a chart

        Example: 
        	x  y_real  y_pred  CI:2.5-97  CI:2.5-97_2  CI:5-95  CI:5-95_2  CI:25-75  CI:25-75_2
     
    title :  The graph title
    dot_name :  Label for the series graphed with dot style
    line_name :  Label for the series graphed with line style
    xLabel: Label for x axis
    yLabel: Label for y axis
    colorListRGB :  List of (list of size 3 [0,1,2])
        Example:  
                    colorListRGB= [ [200,222,255],
                                    [153,212,255],
                                    [0,99,174],
                                    ]

    '''
    columns = df.columns
    x = df[columns[0]].values
    yreal = df[columns[1]].values
    ypred = df[columns[2]].values
    yIntervLimits = list()
    intervNames = list()

    for ii in range(3,len(columns),2):
        yIntervLimits.append(df[columns[ii]].values)
        yIntervLimits.append((df[columns[ii+1]]-df[columns[ii]]).values)
        intervNames.append(columns[ii])

    fig = go.Figure(
    
    )

    fig.update_layout(
        title=title,
        xaxis_title=xLabel,
        yaxis_title=yLabel,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
            
        ),
        # autosize=False,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=True,
        plot_bgcolor='white',
    )
    if len(colorListRGB)==0:
        colorListRGB= [ [200,222,255],
                    [153,212,255],
                    [0,99,174],
                    ]


    # Otros intervalos
    nn = 0
    for ii in range(0,len(yIntervLimits),2):
    # Nivel base
        fig.add_trace(go.Scatter(
                x=x,
                y=  yIntervLimits[ii],
                hoverinfo='x+y',
                mode='lines',
                name='',
                fillcolor='rgba(255,255,255,0.0)',
                opacity=0.0,
                showlegend=False,
                line=dict(width=0.0, color='rgba(255, 255, 255, 0.0)'),
                stackgroup='level'+str(ii))
            )
    # Nivel sombra
        color = 'rgb({d[0]}, {d[1]}, {d[2]})'.format(d=colorListRGB[nn])
        # ii+=1
        fig.add_trace(go.Scatter(
            x=x,
            y= yIntervLimits[ii+1] ,
            hoverinfo='x+y',
            mode='lines',
            name=intervNames[nn],
            opacity=1,
            line=dict(width=0.5, color=color),
            stackgroup='level'+str(ii))
        )
        nn+=1

    # puntos
    fig.add_trace(go.Scatter(
        x=x,y=  yreal,
            # hoverinfo='x+y',
        mode='markers',
        name=dot_name,
        opacity=1,
        line=dict(width=.1, color='rgb(200, 10, 10)'),
        # stackgroup='one'
        )
    )

    # line of y point estimation
    fig.add_trace(go.Scatter(
        x=x,y=  ypred,
            # hoverinfo='x+y',
        mode='lines',
        name=line_name,
        opacity=0.7,
        line=dict(width=3, color='rgb(10, 10, 10)'),
        # stackgroup='one'
        )
    )
    return fig

