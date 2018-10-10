from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app
from apps import vector_surveillance, correlation

vertical = True

app.layout = html.Div([
    html.Div(
        dcc.Tabs(
            tabs=[
                {'label': 'Vector Surveillance Index', 'value': 1},
                {'label': 'Street View Index', 'value': 2},
                {'label': 'Correlations', 'value': 3}
            ],
            value=3,
            id='tabs',
            vertical=vertical,
            style={
                'height': '100vh',
                'borderRight': 'thin lightgrey solid',
                'textAlign': 'left'
            }
        ),
        style={'width': '10%', 'float': 'left'}
    ),
    html.Div(
        html.Div(id='tab-content'),
        style={'width': '90%', 'float': 'right'}
    )
], style={
    'fontFamily': 'Sans-Serif',
    'margin-left': 'auto',
    'margin-right': 'auto',
})


@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value')])
def display_content(value):
    if value == 1:
        return vector_surveillance.layout
    elif value == 2:
        return '2'
    elif value == 3:
        return correlation.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=5000)
