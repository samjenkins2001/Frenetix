from dash import Dash, html, dcc, Input, Output,  dash_table
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px
import os 
import yaml
import pandas as pd

import utils.visualization_helpers as vh

from PIL import Image

mod_path = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))

with open(os.path.join(mod_path, 'analyzer', 'config.yaml')) as stream:
    config = yaml.safe_load(stream)    

cut = config["cut"]  
dash_board_one = config["Dashboard"]
dash_board_two = config["Infeasible"]
trajectory_csv_path = config["Trajectories"]
logs_csv_path = config["logs"]
logs_path = vh.get_latest_log_files(os.path.join(mod_path, 'logs'))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

trajectories_csv = vh.setup_trajs(os.path.join(logs_path, trajectory_csv_path))
logs_csv = pd.read_csv(os.path.join(logs_path, logs_csv_path), delimiter=";")[dash_board_one]

img = vh.get_images(logs_path)

# fig = px.imshow(img, origin="upper", animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
# fig.update_layout(width=500, height=500)

# order of figures
app.layout = html.Div([
    html.Div([

        dbc.Row(dbc.Col(html.H2('TRAJECTORY DEBUGGING', className='text-center text-primary, mb-3'))),  # header row

        dbc.Row(
            html.Div([

                dbc.Container([
                    dbc.Label('Dashboard:'),
                    dash_table.DataTable(
                        id='tweet_table',
                        style_cell = {
                                'height': 'auto',
                                'minWidth': '100px', 'width': '140px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                                }),
                ]),
            ],
            style={'margin-left' : '-280px','width': '100%', 'display': 'inline-block'}),
        ),

        dbc.Row(dbc.Col(html.H2('', className='text-center'))),  # header row
        dbc.Row(dbc.Col(html.H2('', className='text-center'))),  # header row

        html.Div([
            dcc.Dropdown(
                trajectories_csv['feasible'].unique(),                     # filter by
                'All',
                id='crossfilter-xaxis-column',
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

    ],
    ),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': '0'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),


    #html.Div([
    #    dcc.Graph(figure=fig),
    #], style={'display': 'inline-block', 'width': '40%'}),


    html.Div(dcc.Slider(
        trajectories_csv['time_step'].min(),
        trajectories_csv['time_step'].max(),
        step=None,
        id='crossfilter-time_step--slider',
        value=trajectories_csv['time_step'].min(),
        marks={str(time_step): str(time_step) for time_step in trajectories_csv['time_step'].unique()}
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),


    html.Div([
        dcc.Graph(id='x-time-series'),
    ], style={'display': 'inline-block', 'width': '30%'}),

    html.Div([
        dcc.Graph(id='z-time-series'),
    ], style={'display': 'inline-block', 'width': '39%'}),


    dbc.Row(
        html.Div([

            dbc.Container([
                dash_table.DataTable(id='tweet_table_two'),
            ]),
        ],
        style={'margin-left' : '-280px','width': '100%', 'display': 'inline-block'}),
    ),

    dbc.Row(dbc.Col(html.H2('', className='text'))),  
    dbc.Row(dbc.Col(html.H2('', className='text_1'))),  

    dbc.Row(
        [
            dag.AgGrid(
                #enableEnterpriseModules=True,
                #licenseKey = os.environ['AGGRID_ENTERPRISE'],
                columnDefs=[{"headerName": i, "field": i} for i in logs_csv.columns],
                rowData=logs_csv.to_dict('records'),
                columnSize="sizeToFit",
                defaultColDef=dict(
                    resizable=True,
                ),
            ),
        ]
    )

])


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-time_step--slider', 'value'))
def update_graph(xaxis_column_name, yaxis_column_name,
                 time_step_value):
    dff = trajectories_csv[(trajectories_csv['time_step']==time_step_value) &
                           (trajectories_csv['variable']=="costs_cumulative_weighted") &
                           (trajectories_csv['feasible']==yaxis_column_name)][["x_positions_m", "y_positions_m",
                                                                               "trajectory_number", "value", "feasible"]]  # to be changed to total_costs
    
    fig = px.scatter(dff,
        x=dff['x_positions_m'],
        y=dff['y_positions_m'],
        hover_name = dff['trajectory_number'],
        color="value", color_continuous_scale='rdylgn_r'
        )

    fig.update_traces(customdata=dff[dff['feasible'] == yaxis_column_name]['trajectory_number'])

    fig.update_xaxes(title="x_position", type='linear')

    fig.update_yaxes(title="y_position", type='linear')

    fig.update_layout(yaxis_range=[140, 160], xaxis_range=[90, 110], margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    pyLogo = Image.fromarray(img[time_step_value])

    x, y = float(logs_csv['x_positions_m'][0].split(",")[0]), float(logs_csv['y_positions_m'][0].split(",")[0]) #98.89039368169261, 153.4609404907308   # 35 35.1 2.1

    fig.add_layout_image(
            dict(
                source=pyLogo,  
                xref="x",
                yref="y",
                x=x-35,
                y=y+35,
                sizex=70,
                sizey=70,
                sizing="stretch",
                opacity=0.5,
                layer="below")
    )

    return fig


def create_time_series_x(dff, title):   # first plot

    try:
        total = dff[dff['variable'] == "costs_cumulative_weighted"]["value"].iloc[0]
    except:
        total = 1
    dff["Percentage"] = dff["value"] / total

    fig = px.bar(dff, x='trajectory_number', y="value", color="variable", barmode='group', hover_data=["Percentage"])

    fig.update_traces()

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


def create_time_series(dff, title):   

    # todo: remove total costs

    fig = px.bar(dff, x= 'trajectory_number', y="value", color="variable") 

    fig.update_traces()

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    Output('x-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),            
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-time_step--slider', 'value'))
def update_y_timeseries(hoverData, xaxis_column_name, time_step):
    traj_name = hoverData['points'][0]['customdata']
    dff = trajectories_csv[trajectories_csv['trajectory_number'] == traj_name]
    dff = dff[dff['time_step'] == time_step]  
    dff = dff[dff['feasible'] == xaxis_column_name]
    title = '<b>{} </b><br>{}'.format(traj_name, xaxis_column_name)   # add percentage
    return create_time_series_x(dff, title)


@app.callback(
    Output('z-time-series', 'figure'),
    Input('crossfilter-indicator-scatter', 'hoverData'),    
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-time_step--slider', 'value'))
def update_z_timeseries(hoverData, yaxis_column_name, time_step):
    dff = trajectories_csv[trajectories_csv['feasible'] == yaxis_column_name]
    dff = dff[dff['time_step'] == time_step]
    dff = dff[dff['variable'] != "costs_cumulative_weighted"]
    return create_time_series(dff, yaxis_column_name)


@app.callback(
    Output(component_id='tweet_table', component_property='data'),
    Input('crossfilter-time_step--slider', 'value'))
def display_tweets(time_step):
    logs = vh.setup_logs(logs_path + "/" + 'logs.csv')
    logs = logs[(logs['trajectory_number'] == time_step)]  # filter for time_step
    logs = logs[dash_board_one]
    return logs.to_dict(orient='records')

@app.callback(
    Output(component_id='tweet_table_two', component_property='data'),
    Input('crossfilter-time_step--slider', 'value'))
def display_tweets_two(time_step):
    logs = vh.setup_logs(logs_path + "/" + 'logs.csv')
    logs = logs[(logs['trajectory_number'] == time_step)]  # filter for time_step
    logs = logs[dash_board_two]
    return logs.to_dict(orient='records')


if __name__ == '__main__':
    app.run_server(debug=True)
