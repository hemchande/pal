import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Wildfire Mitigation System Dashboard"

# API base URL
API_BASE_URL = "http://localhost:8000"

def create_risk_map():
    """Create a map showing wildfire risk levels"""
    # Sample data - in production, this would come from the API
    sample_data = pd.DataFrame({
        'lat': [37.7749, 34.0522, 32.7157, 38.5816],
        'lon': [-122.4194, -118.2437, -117.1611, -121.4944],
        'risk_score': [0.8, 0.6, 0.3, 0.7],
        'location': ['San Francisco', 'Los Angeles', 'San Diego', 'Sacramento']
    })
    
    fig = px.scatter_mapbox(
        sample_data,
        lat='lat',
        lon='lon',
        size='risk_score',
        color='risk_score',
        hover_name='location',
        hover_data=['risk_score'],
        color_continuous_scale='Reds',
        size_max=20,
        zoom=5,
        center={'lat': 36.7783, 'lon': -119.4179}  # Center of California
    )
    
    fig.update_layout(
        mapbox_style='open-street-map',
        title='Wildfire Risk Assessment - California',
        height=500
    )
    
    return fig

def create_weather_gauge(temp, humidity, wind_speed):
    """Create weather gauge charts"""
    # Temperature gauge
    temp_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=temp,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Temperature (°C)"},
        delta={'reference': 20},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 35], 'color': "yellow"},
                {'range': [35, 50], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40
            }
        }
    ))
    
    # Humidity gauge
    humidity_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=humidity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Humidity (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    return temp_fig, humidity_fig

def create_activity_chart():
    """Create human activity chart"""
    # Sample activity data
    activity_data = pd.DataFrame({
        'activity_type': ['Hiking', 'Camping', 'Construction', 'Recreation'],
        'intensity': [0.7, 0.3, 0.2, 0.5],
        'color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    })
    
    fig = px.bar(
        activity_data,
        x='activity_type',
        y='intensity',
        color='activity_type',
        title='Human Activity Intensity',
        color_discrete_map={
            'Hiking': '#FF6B6B',
            'Camping': '#4ECDC4',
            'Construction': '#45B7D1',
            'Recreation': '#96CEB4'
        }
    )
    
    fig.update_layout(
        yaxis_title='Intensity (0-1)',
        height=300
    )
    
    return fig

def create_alerts_table():
    """Create alerts table"""
    # Sample alerts data
    alerts_data = [
        {
            'type': 'Evacuation',
            'location': 'San Francisco Bay Area',
            'severity': 'High',
            'message': 'Voluntary evacuation recommended due to high fire risk',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        },
        {
            'type': 'Warning',
            'location': 'Los Angeles County',
            'severity': 'Medium',
            'message': 'Avoid outdoor activities in high-risk areas',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    ]
    
    return alerts_data

def create_recommendations_cards():
    """Create recommendation cards"""
    recommendations = [
        {
            'audience': 'Firefighters',
            'priority': 'High',
            'description': 'Deploy additional units to high-risk areas',
            'impact': 'Reduce response time by 40%'
        },
        {
            'audience': 'Officials',
            'priority': 'Medium',
            'description': 'Issue public safety alerts',
            'impact': 'Reach 10,000+ residents'
        },
        {
            'audience': 'Public',
            'priority': 'Medium',
            'description': 'Avoid outdoor activities in high-risk areas',
            'impact': 'Reduce ignition sources'
        }
    ]
    
    return recommendations

# Dashboard layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🔥 Wildfire Mitigation System", className="text-center mb-4"),
            html.H4("Real-time Risk Monitoring & Response", className="text-center text-muted mb-4")
        ])
    ]),
    
    # System Status
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("System Status", className="card-title"),
                    html.P("🟢 Operational", className="card-text"),
                    html.Small(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Active Fires", className="card-title"),
                    html.H2("2", className="text-danger"),
                    html.Small("Currently monitored")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("High Risk Areas", className="card-title"),
                    html.H2("5", className="text-warning"),
                    html.Small("Risk score > 0.7")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Active Alerts", className="card-title"),
                    html.H2("3", className="text-info"),
                    html.Small("Public notifications")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Risk Map
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(
                        id='risk-map',
                        figure=create_risk_map()
                    )
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Weather and Activity
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Weather Conditions", className="card-title"),
                    dcc.Graph(
                        id='temp-gauge',
                        figure=create_weather_gauge(28, 35, 15)[0],
                        style={'height': '200px'}
                    ),
                    dcc.Graph(
                        id='humidity-gauge',
                        figure=create_weather_gauge(28, 35, 15)[1],
                        style={'height': '200px'}
                    )
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Human Activity", className="card-title"),
                    dcc.Graph(
                        id='activity-chart',
                        figure=create_activity_chart()
                    )
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Alerts and Recommendations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Active Alerts", className="card-title"),
                    html.Div(id='alerts-table')
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Recommendations", className="card-title"),
                    html.Div(id='recommendations-cards')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Refresh interval
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # 30 seconds
        n_intervals=0
    )
], fluid=True)

@app.callback(
    Output('alerts-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_alerts_table(n):
    """Update alerts table"""
    alerts = create_alerts_table()
    
    table_rows = []
    for alert in alerts:
        severity_color = {
            'High': 'danger',
            'Medium': 'warning',
            'Low': 'info'
        }.get(alert['severity'], 'secondary')
        
        row = dbc.Row([
            dbc.Col(html.Span(alert['type'], className=f"badge bg-{severity_color}"), width=2),
            dbc.Col(alert['location'], width=3),
            dbc.Col(alert['message'], width=5),
            dbc.Col(alert['timestamp'], width=2)
        ], className="mb-2")
        table_rows.append(row)
    
    return table_rows

@app.callback(
    Output('recommendations-cards', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_recommendations_cards(n):
    """Update recommendations cards"""
    recommendations = create_recommendations_cards()
    
    cards = []
    for rec in recommendations:
        priority_color = {
            'High': 'danger',
            'Medium': 'warning',
            'Low': 'info'
        }.get(rec['priority'], 'secondary')
        
        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"{rec['audience']} - {rec['priority']}", 
                       className=f"text-{priority_color}"),
                html.P(rec['description'], className="card-text"),
                html.Small(f"Impact: {rec['impact']}", className="text-muted")
            ])
        ], className="mb-2")
        cards.append(card)
    
    return cards

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8001) 