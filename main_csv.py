import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import os

# CSV paths (absolute) - updated dataset file names
TRAIN_CSV = r"C:\Users\black\Desktop\projects\Aviation Component Anomaly Reporting Dashboard\raw_sensor_data.csv"
TRUTH_CSV = r"C:\Users\black\Desktop\projects\Aviation Component Anomaly Reporting Dashboard\component_rul.csv"

# --- Load CSVs ---
# raw_sensor_data contains: Unit_ID, Operational_Cycle, settings..., s1..s21
# component_rul contains: Unit_ID, Remaining_Life
df_sensor = pd.read_csv(TRAIN_CSV)
df_rul = pd.read_csv(TRUTH_CSV)

# Ensure correct column names (some CSVs may use lowercase)
if 'id' in df_sensor.columns and 'Unit_ID' not in df_sensor.columns:
    df_sensor = df_sensor.rename(columns={'id': 'Unit_ID'})
if 'cycle' in df_sensor.columns and 'Operational_Cycle' not in df_sensor.columns:
    df_sensor = df_sensor.rename(columns={'cycle': 'Operational_Cycle'})

if 'Unit_ID' not in df_rul.columns and 'id' in df_rul.columns:
    df_rul = df_rul.rename(columns={'id': 'Unit_ID'})

# normalize remaining-life column name (accept variants: cycle, RUL, remaining_life, Remaining_Life, etc.)
if 'cycle' in df_rul.columns and 'Remaining_Life' not in df_rul.columns:
    df_rul = df_rul.rename(columns={'cycle': 'Remaining_Life'})
elif 'Remaining_Life' not in df_rul.columns:
    for c in df_rul.columns:
        if any(keyword in c.lower() for keyword in ['remain', 'rul', 'life']):
            if c != 'Remaining_Life':
                df_rul = df_rul.rename(columns={c: 'Remaining_Life'})
            break

# Merge sensor data with remaining-life table on Unit_ID
if 'Unit_ID' in df_sensor.columns and 'Unit_ID' in df_rul.columns:
    df_merged = pd.merge(df_sensor, df_rul, on='Unit_ID', how='left')
else:
    df_merged = df_sensor.copy()

# Use merged DataFrame for analytics
df_report = df_merged

# Ensure s11 exists
if 's11' in df_report.columns:
    # Rolling average per unit
    df_report['rolling_avg_s11'] = df_report.groupby('Unit_ID')['s11'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    # Z-score within each unit over rolling_avg_s11
    def zscore(s):
        denom = s.std(ddof=0)
        # If std is zero or NaN, return a series of NA to avoid divide-by-zero
        if denom == 0 or pd.isna(denom):
            return pd.Series([pd.NA] * len(s), index=s.index)
        return (s - s.mean()) / denom
    df_report['Z_Score'] = df_report.groupby('Unit_ID')['rolling_avg_s11'].transform(lambda x: zscore(x))
    # Mark anomalies
    df_report['Anomaly_Flag'] = df_report['Z_Score'].apply(lambda v: 'HIGH ANOMALY' if pd.notna(v) and v > 3 else '')
else:
    df_report['rolling_avg_s11'] = pd.NA
    df_report['Z_Score'] = pd.NA
    df_report['Anomaly_Flag'] = ''

# Create a helpful default figure for RUL compare when data is missing
def _empty_rul_figure(msg: str = "Estimated RUL not available (need s11 rolling_avg and Remaining_Life)"):
    return {
        'data': [],
        'layout': {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'annotations': [{
                'text': msg,
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False,
                'font': {'size': 14}
            }]
        }
    }

# Expose Unit options
UNIT_OPTIONS = [{'label': f'Engine Unit {i}', 'value': i} for i in df_report['Unit_ID'].unique()]

# --- Dash App ---
app = dash.Dash(__name__, title="Airbus CSV Anomaly Report")
server = app.server

app.layout = html.Div(
    style={'backgroundColor': '#f8fafc', 'fontFamily': 'Arial, sans-serif', 'padding': '10px 20px'},
    children=[
        html.Div(style={'backgroundColor': '#00205B', 'color': 'white', 'padding': '12px 20px'}, children=[
            html.H2('Aircraft Engine Predictive Maintenance Dashboard (CMAPSS)', style={'margin': 0}),
            html.P('A compact interactive dashboard built from the CMAPSS-style PM_train and PM_truth files to explore Remaining Useful Life (RUL) and sensor degradation.', style={'margin': '4px 0 0 0'})
        ]),

        # Dataset description
        html.Div(style={'backgroundColor': '#ffffff', 'padding': '12px', 'borderRadius': '6px', 'marginTop': '12px'}, children=[
            html.H4('Dataset Overview', style={'marginTop': 0}),
            html.P("raw_sensor_data.csv: multivariate time series for multiple aircraft engines. Each row is an operational cycle with 21 sensor readings (s1–s21), settings, Unit_ID, and Operational_Cycle."),
            html.P("component_rul.csv: Remaining Useful Life (RUL) labels with Unit_ID and remaining cycles before failure."),
            html.P("This dashboard visualizes sensor degradation and derives anomaly signals (rolling average + Z-score on sensor s11) to highlight potential critical events."),
        ]),

        html.Div(style={'marginTop': '15px'}, children=[
            dcc.Dropdown(id='unit-dropdown', options=UNIT_OPTIONS, value=UNIT_OPTIONS[0]['value'] if UNIT_OPTIONS else None, clearable=False),

            html.Div(id='no-data-message', style={'color': 'red', 'marginTop': '8px'}),

            dcc.Graph(id='trend-chart'),
            dcc.Graph(id='rul-compare'),
            html.P("Trend chart: shows sensor degradation over operational cycles (rolling average of s11) when available; otherwise shows Remaining Life. Red markers indicate detected HIGH ANOMALY events (Z > 3).", style={'fontSize': '12px', 'color': '#333'}),
            html.P("RUL comparison: true Remaining_Life vs estimated RUL (simple sliding-window linear estimator). If not available, a message will appear explaining why.", style={'fontSize': '12px', 'color': '#333'}),

            html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                dcc.Graph(id='zscore-hist', style={'flex': '1'}),
                dcc.Graph(id='s11-box', style={'flex': '1'})
            ],),
            html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '6px'}, children=[
                html.Div(children=html.P("Z-Score Histogram: distribution of per-unit Z-scores computed on the rolling average of s11. Used to identify outliers and extreme degradation." , style={'fontSize':'12px'}), style={'flex':'1'}),
                html.Div(children=html.P("s11 Box Plot: shows variability of sensor s11 across units. Useful to compare baseline levels and spread." , style={'fontSize':'12px'}), style={'flex':'1'})
            ]),

            dcc.Graph(id='anomaly-pie'),
            html.P("Anomaly Pie: proportion of detected anomaly states (e.g., HIGH ANOMALY vs normal) across the dataset.", style={'fontSize': '12px', 'color': '#333'}),

            dash_table.DataTable(id='anomaly-table', columns=[{'name': c, 'id': c} for c in ['Operational_Cycle', 's11', 'Z_Score', 'Anomaly_Flag']], page_size=10),
            html.P("Anomaly Table: lists detected HIGH ANOMALY events for the selected unit with cycle, sensor value and Z-score for quick inspection.", style={'fontSize': '12px', 'color': '#333', 'marginTop': '6px'})
        ])
    ]
)

# --- Callbacks ---
@app.callback(
    [Output('trend-chart', 'figure'), Output('rul-compare', 'figure'), Output('anomaly-table', 'data'), Output('no-data-message', 'children'),
     Output('zscore-hist', 'figure'), Output('s11-box', 'figure'), Output('anomaly-pie', 'figure')],
    [Input('unit-dropdown', 'value')]
)
def update_from_csv(selected_unit):
    if df_report.empty or selected_unit is None:
        empty_fig = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No data", "xref": "paper", "yref": "paper", "showarrow": False}]}} 
        # return 7 outputs: trend, rul-compare, table, message, zhist, s11box, apie
        return empty_fig, empty_fig, [], "No data", empty_fig, empty_fig, empty_fig

    df_filtered = df_report[df_report['Unit_ID'] == selected_unit].copy()
    if df_filtered.empty:
        empty_fig = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"text": "No data for this unit", "xref": "paper", "yref": "paper", "showarrow": False}]}}
        return empty_fig, empty_fig, [], "No data for this unit", empty_fig, empty_fig, empty_fig

    # Trend chart: rolling_avg_s11 or Remaining_Life if s11 missing
    if 'rolling_avg_s11' in df_filtered.columns and df_filtered['rolling_avg_s11'].notna().any():
        fig = px.line(df_filtered, x='Operational_Cycle', y='rolling_avg_s11', title=f'Rolling Avg s11 (Unit {selected_unit})')
        # overlay anomalies if present
        if 'Anomaly_Flag' in df_filtered.columns:
            anomalies = df_filtered[df_filtered['Anomaly_Flag'] == 'HIGH ANOMALY']
            if not anomalies.empty:
                fig.add_scatter(x=anomalies['Operational_Cycle'], y=anomalies['rolling_avg_s11'], mode='markers', marker={'color':'red','size':8}, name='HIGH ANOMALY')
    else:
        fig = px.line(df_filtered, x='Operational_Cycle', y='Remaining_Life', title=f'Remaining Life (Unit {selected_unit})')

    # Anomaly table
    table_df = df_filtered[df_filtered.get('Anomaly_Flag', '') == 'HIGH ANOMALY'] if 'Anomaly_Flag' in df_filtered.columns else pd.DataFrame()
    table_data = table_df[['Operational_Cycle','s11','Z_Score','Anomaly_Flag']].to_dict('records') if not table_df.empty else []

    # --- RUL comparison: estimated RUL (sliding linear fit on rolling_avg_s11) vs true Remaining_Life ---
    rul_compare_fig = _empty_rul_figure()
    if 'rolling_avg_s11' in df_filtered.columns and df_filtered['rolling_avg_s11'].notna().sum() >= 3 and 'Remaining_Life' in df_filtered.columns:
        # sliding window linear fit
        import numpy as np
        window = 10
        est_rul = []
        cycles = df_filtered['Operational_Cycle'].to_numpy()
        vals = pd.to_numeric(df_filtered['rolling_avg_s11'], errors='coerce').to_numpy()
        for i in range(len(vals)):
            start = max(0, i - window + 1)
            x = cycles[start:i+1]
            y = vals[start:i+1]
            # require at least 3 valid points
            if len(x) >= 3 and not np.isnan(y).all():
                mask = ~np.isnan(y)
                if mask.sum() >= 3:
                    xp = x[mask]
                    yp = y[mask]
                    # linear fit
                    try:
                        m, b = np.polyfit(xp, yp, 1)
                    except Exception:
                        est_rul.append(pd.NA)
                        continue
                    last_val = yp[-1]
                    # failure threshold: mean + 3*std of window
                    thresh = np.nanmean(yp) + 3 * np.nanstd(yp)
                    if m == 0 or np.isnan(m):
                        est_rul.append(pd.NA)
                    else:
                        cycles_to_thresh = (thresh - last_val) / m
                        est_rul.append(cycles_to_thresh if cycles_to_thresh > 0 else pd.NA)
                else:
                    est_rul.append(pd.NA)
            else:
                est_rul.append(pd.NA)

        df_filtered = df_filtered.assign(Estimated_RUL=pd.Series(est_rul, index=df_filtered.index))
        # Ensure both columns are numeric (same type) for Plotly
        df_filtered['Remaining_Life'] = pd.to_numeric(df_filtered['Remaining_Life'], errors='coerce')
        df_filtered['Estimated_RUL'] = pd.to_numeric(df_filtered['Estimated_RUL'], errors='coerce')
        # plot true Remaining_Life and Estimated_RUL
        rul_compare_fig = px.line(df_filtered, x='Operational_Cycle', y=['Remaining_Life', 'Estimated_RUL'],
                                 title=f'Remaining Life vs Estimated RUL (Unit {selected_unit})',
                                 labels={'value': 'Cycles', 'variable': 'Source'})
        rul_compare_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    else:
        rul_compare_fig = _empty_rul_figure("Estimated RUL not computed: need at least 3 valid s11 points and a Remaining_Life column")

    # Z-score histogram over all units
    if 'Z_Score' in df_report.columns:
        zseries = pd.to_numeric(df_report['Z_Score'], errors='coerce').dropna()
        if not zseries.empty:
            zhist = px.histogram(x=zseries, nbins=30, title='Z_Score Distribution')
        else:
            zhist = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}
    else:
        zhist = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}

    # s11 box by unit
    if 's11' in df_report.columns:
        s11df = df_report[['Unit_ID','s11']].dropna()
        if not s11df.empty:
            s11box = px.box(s11df, x='Unit_ID', y='s11', title='s11 by Unit')
        else:
            s11box = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}
    else:
        s11box = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}

    # anomaly pie
    if 'Anomaly_Flag' in df_report.columns:
        pie_df = df_report['Anomaly_Flag'].value_counts().reset_index()
        pie_df.columns = ['Anomaly_Flag','count']
        if not pie_df.empty:
            apie = px.pie(pie_df, names='Anomaly_Flag', values='count', title='Anomaly Flag Distribution')
        else:
            apie = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}
    else:
        apie = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}}}

    return fig, rul_compare_fig, table_data, "", zhist, s11box, apie

if __name__ == '__main__':
    app.run(debug=True)
