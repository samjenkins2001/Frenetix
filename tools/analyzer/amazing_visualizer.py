# This script provides a dashboard layout to debug log data of a trajectory algorithm in an interactive manner

from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_daq as daq

import plotly.express as px
import plotly
from PIL import Image
import os
import numpy as np
import yaml

import utils.visualization_helpers as vh

import time

mod_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

with open(os.path.join(mod_path, 'tools', 'analyzer', 'config.yaml')) as stream:
    config = yaml.safe_load(stream)
dash_board_one = config["Dashboard"]
dash_board_two = config["Infeasible"]
trajectory_csv_path = config["Trajectories"]
logs_csv_path = config["logs"]

with open(os.path.join(mod_path, 'configurations', 'defaults', 'debug.yaml')) as stream:    # access debug info
    debug = yaml.safe_load(stream)
plot_window = debug["plot_window_dyn"]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']                       # set up dashboard style
app = Dash(__name__, external_stylesheets=external_stylesheets)

logs_path = vh.get_latest_log_files(os.path.join(mod_path, 'logs'))                         # set up log file path & load csv's
trajectories_csv = vh.setup_trajs(os.path.join(logs_path, trajectory_csv_path))
log = vh.setup_logs(os.path.join(logs_path, logs_csv_path))

img = vh.get_images(logs_path)                                                              # load scenario visualization

time_step = 0
trajectory_number = 0                                                           # load scenario visualization

# order of figures
app.layout = html.Div([
    html.Div([

        dbc.Row(dbc.Col(html.H2('TRAJECTORY DEBUGGING', className='text-center text-primary, mb-3'))),  # header row

        dbc.Row(
            html.Div([

                dbc.Container([  # info table 1: shows useful information such as trajectory number and the ego vehicle
                    dbc.Label('Dashboard:'),
                    dash_table.DataTable(
                        id='tweet_table',
                        style_cell={
                            'height': 'auto',
                            'minWidth': '100px', 'width': '140px', 'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        }),
                ]),
            ],
                style={'margin-left': '-140px', 'margin-bottom': '40px', 'width': '100%', 'display': 'inline-block'}),
        ),

        html.Div([
            dcc.Dropdown(  # filter for all/feasible/infeasible trajectories
                trajectories_csv['feasible'].unique(),
                'All',
                id='crossfilter-xaxis-column',
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),

    ],
    ),
    html.Div([
        dcc.Graph(  # scenario visualization with interactive trajectory interface
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': '0'}]}
        )
    ], style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Input(  # Input field to choose timestep for debugging, updates in: update_graph
            id="crossfilter-time_step--slider", type="number", value=trajectories_csv['time_step'].min(),
            min=trajectories_csv['time_step'].min(), max=trajectories_csv['time_step'].max(), step=1,
        ),

        dcc.Input(  # Input field to choose timestep for debugging, updates in: update_graph
            id="trajectory-number-input", type="number", value=trajectories_csv['trajectory_number'].min(),
            min=trajectories_csv['trajectory_number'].min(), max=trajectories_csv['trajectory_number'].max(), step=1,
        ),
    ], style={'margin-left': '160px', 'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([  # cost allocation for chosen trajectory, updates in: update_y_timeseries
        dcc.Graph(id='x-time-series'),
    ], style={'display': 'inline-block', 'width': '30%'}),

    html.Div([  # cost allocation for chosen trajectory, updates in: update_z_timeseries
        dcc.Graph(id='z-time-series'),
    ], style={'display': 'inline-block', 'width': '39%'}),

    dbc.Row(  # info table 2: shows number of infeasible listed by the reason
        html.Div([

            dbc.Container([
                dash_table.DataTable(id='tweet_table_two'),
            ]),
        ],
            style={'margin-left': '-140px', 'margin-bottom': '40px', 'width': '100%', 'display': 'inline-block'}),
    ),

    dbc.Row(
        [
            dag.AgGrid(
                columnDefs=[{"headerName": i, "field": i} for i in log[dash_board_one].columns],
                # info table 3: shows log data that allows to check additional info while debugging
                rowData=log[dash_board_one].to_dict('records'),
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
    Output('crossfilter-time_step--slider', 'value'),
    Output('trajectory-number-input', 'value'),
    Input('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-time_step--slider', 'value'),
    Input('trajectory-number-input', 'value'))
def update_graph(figure, xaxis_column_name, yaxis_column_name,
                 time_step_input, trajectory_number_input):
    global time_step
    global trajectory_number

    time_step_changed = time_step != time_step_input
    trajectory_number_changed = trajectory_number != trajectory_number_input

    if time_step_changed == False & trajectory_number_changed == False:
        time_step_changed = True

    if time_step_changed:  # make new figure

        if time_step_input == None:  # if someone inputs non-sense just put to last value
            time_step_input = time_step

        if trajectory_number_input == None:  # if someone inputs non-sense just put to last value
            trajectory_number_input = trajectory_number

        dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                               (trajectories_csv['variable'] == "costs_cumulative_weighted") &
                               (trajectories_csv['feasible'] == yaxis_column_name)][["x_positions_m", "y_positions_m",
                                                                                     "trajectory_number", "value",
                                                                                     "feasible"]]

        fig = px.scatter(dff,
                         x=dff['x_positions_m'],
                         y=dff['y_positions_m'],
                         hover_name=dff['trajectory_number'],
                         color="value", color_continuous_scale='rdylgn_r'
                         )

        logg = log[log['trajectory_number'] == time_step_input]
        x, y = float(logg["x_position_vehicle_m"][time_step_input]), float(
            logg["y_position_vehicle_m"][time_step_input])

        background = Image.fromarray(img[time_step_input])

        fig.update_traces(customdata=dff[dff['feasible'] == yaxis_column_name]['trajectory_number'])

        fig.update_xaxes(title="x_position", type='linear')

        fig.update_yaxes(title="y_position", type='linear')

        fig.update_layout(yaxis_range=[y - 15, y + 15], xaxis_range=[x - 15, x + 15], hovermode='closest',
                          autosize=False)

        fig.update_xaxes(
            scaleanchor="y",
            scaleratio=1,
        )

        fig.add_layout_image(
            dict(
                source=background,
                xref="x",
                yref="y",
                x=x - plot_window,
                y=y + plot_window,
                sizex=2 * plot_window,
                sizey=2 * plot_window,
                layer="below")
        )

        # highlight current trajectory

        dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                               (trajectories_csv['trajectory_number'] == trajectory_number_input) &
                               (trajectories_csv['variable'] == "costs_cumulative_weighted")][
            ["x_positions_m", "y_positions_m",
             "trajectory_number", "value", "feasible"]]

        if yaxis_column_name == "All":
            dff = dff[dff['feasible'] == "All"].reset_index()
        else:
            dff = dff[dff['feasible'] != "All"].reset_index()
            if len(trajectories_csv[(trajectories_csv['time_step'] == time_step_input) & (
                    trajectories_csv['feasible'] == yaxis_column_name)]) == 0:
                return fig

        while dff['feasible'][
            0] != yaxis_column_name:  # find next higher trajectory that fullfills feasibility requirements
            trajectory_number_input += 1
            dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                                   (trajectories_csv['trajectory_number'] == trajectory_number_input) &
                                   (trajectories_csv['feasible'] != "All") &
                                   (trajectories_csv['variable'] == "costs_cumulative_weighted")][
                ["x_positions_m", "y_positions_m",
                 "trajectory_number", "value", "feasible"]]

            if len(dff) == 0:  # if max trajectory number reached start from beginning
                trajectory_number_input = 0
                dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                                       (trajectories_csv['trajectory_number'] == trajectory_number_input) &
                                       (trajectories_csv['feasible'] != "All") &
                                       (trajectories_csv['variable'] == "costs_cumulative_weighted")][
                    ["x_positions_m", "y_positions_m",
                     "trajectory_number", "value", "feasible"]]

            dff = dff.reset_index()

        x, y = dff["x_positions_m"][0], dff["y_positions_m"][0]

        fig.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text="current",
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=10,
                color="#000000"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#000000",
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=2,
            bgcolor="#F0F8FF",
            opacity=0.8
        )

        time_step = time_step_input
        trajectory_number = trajectory_number_input

        figure = fig

    if trajectory_number_changed:  # remove old annotation and add new annotation

        figure['layout']['annotations'][-1]['visible'] = False  # remove earlier annotation

        dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                               (trajectories_csv['trajectory_number'] == trajectory_number_input) &
                               (trajectories_csv['variable'] == "costs_cumulative_weighted")][
            ["x_positions_m", "y_positions_m",
             "trajectory_number", "value", "feasible"]]

        if yaxis_column_name == "All":
            dff = dff[dff['feasible'] == "All"].reset_index()
        else:
            dff = dff[dff['feasible'] != "All"].reset_index()
            if len(trajectories_csv[(trajectories_csv['time_step'] == time_step_input) & (
                    trajectories_csv['feasible'] == yaxis_column_name)]) == 0:
                return fig

        while dff['feasible'][
            0] != yaxis_column_name:  # find next higher trajectory that fullfills feasibility requirements
            trajectory_number_input += 1
            dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                                   (trajectories_csv['trajectory_number'] == trajectory_number_input) &
                                   (trajectories_csv['feasible'] != "All") &
                                   (trajectories_csv['variable'] == "costs_cumulative_weighted")][
                ["x_positions_m", "y_positions_m",
                 "trajectory_number", "value", "feasible"]]

            if len(dff) == 0:  # if max trajectory number reached start from beginning
                trajectory_number_input = 0
                dff = trajectories_csv[(trajectories_csv['time_step'] == time_step_input) &
                                       (trajectories_csv['trajectory_number'] == trajectory_number_input) &
                                       (trajectories_csv['feasible'] != "All") &
                                       (trajectories_csv['variable'] == "costs_cumulative_weighted")][
                    ["x_positions_m", "y_positions_m",
                     "trajectory_number", "value", "feasible"]]

            dff = dff.reset_index()

        x, y = dff["x_positions_m"][0], dff["y_positions_m"][0]  # get coordinates of current trajectory

        figure.add_annotation(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text="current",
            showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=10,
                color="#000000"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#000000",
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=2,
            bgcolor="#F0F8FF",
            opacity=0.8
        )

        trajectory_number = trajectory_number_input

    return figure, time_step, trajectory_number  # update inputs in case they were changed due to invalidity


def create_time_series_one(dff, title):
    fig = px.bar(dff, x='trajectory_number', y="value", color="variable")

    fig.update_traces()

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type='linear')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


def create_time_series_two(dff, title):
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


@app.callback(
    Output('z-time-series', 'figure'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-time_step--slider', 'value'))
def update_z_timeseries(yaxis_column_name, time_step):
    dff = trajectories_csv[trajectories_csv['feasible'] == yaxis_column_name]
    dff = dff[dff['time_step'] == time_step]
    dff = dff[dff['variable'] != "costs_cumulative_weighted"]
    return create_time_series_one(dff, yaxis_column_name)


@app.callback(  # trajectory-number-input
    Output('x-time-series', 'figure'),
    Input('trajectory-number-input', 'value'),  # Input('crossfilter-indicator-scatter', 'hoverData'),
    Input('crossfilter-xaxis-column', 'value'),
    Input('crossfilter-time_step--slider', 'value'))
def update_y_timeseries(traj_name, xaxis_column_name, time_step):  # (hoverData, xaxis_column_name, time_step):
    # traj_name = hoverData['points'][0]['customdata']
    dff = trajectories_csv[trajectories_csv['trajectory_number'] == traj_name]
    dff = dff[dff['time_step'] == time_step]
    dff = dff[dff['feasible'] == xaxis_column_name]
    title = '<b>{} </b><br>{}'.format(traj_name, xaxis_column_name)  # add percentage
    return create_time_series_two(dff, title)


@app.callback(
    Output(component_id='tweet_table', component_property='data'),
    Input('crossfilter-time_step--slider', 'value'))
def display_tweets(time_step):
    logs = log[(log['trajectory_number'] == time_step)][dash_board_one]  # filter for time_step
    return logs.to_dict(orient='records')


@app.callback(
    Output(component_id='tweet_table_two', component_property='data'),
    Input('crossfilter-time_step--slider', 'value'))
def display_tweets_two(time_step):
    logs = log[(log['trajectory_number'] == time_step)][dash_board_two]  # filter for time_step
    return logs.to_dict(orient='records')


if __name__ == '__main__':
    app.run_server(debug=True)
