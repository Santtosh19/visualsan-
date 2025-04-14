import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, ctx, no_update
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Load data
PARQUET_HEADER = "C:/Users/E1009134/Documents/Santtosh tests/data_head_ori.parquet"
PARQUET_SUMMARY = "C:/Users/E1009134/Documents/Santtosh tests/data_summary_ori.parquet"
PARQUET_DATA = "C:/Users/E1009134/Documents/Santtosh tests/data_data_ori.parquet"
machine_map = pd.read_excel("C:/Users/E1009134/Documents/Santtosh tests/GT_EquipName_to_StandardNaming.xlsx")
df_data = pd.read_parquet(PARQUET_DATA)
df_header = pd.read_parquet(PARQUET_HEADER)
df_summary = pd.read_parquet(PARQUET_SUMMARY)


# Merge to bring in STANDARD_NAME
df_header = df_header.merge(machine_map, on="MACHINE_NAME", how="inner")

# Replace MACHINE_NAME with STANDARD_NAME (or create a new column if you prefer)
df_header["MACHINE_NAME"] = df_header["STANDARD_NAME"]


# CCD unpivot logic
ccd_cols = ["CCD1", "CCD2", "CCD3", "CCD4", "CCD5", "CCD6", "CCD7", "CCD8"]
df_data = df_data[df_data[ccd_cols].sum(axis=1) > 0]  # keep only rows with actual detections

# Unpivot CCD columns
df_unpivot = df_data.melt(
    id_vars=["HEADER_ID", "PARAMETER"],
    value_vars=ccd_cols,
    var_name="CAMERA",
    value_name="COUNT"
)

# Step 2: Merge with GT_HEADER to bring in metadata
df_unpivot = df_unpivot.merge(
    df_header[["HEADER_ID", "LOT_NUMBER", "EMPLOYEE_ID", "RECIPE_NAME", "DEVICE","STANDARD_NAME"]],
    on="HEADER_ID",
    how="inner"
)


df_header["TIMESTAMP"] = pd.to_datetime(df_header["TIMESTAMP"])
df_header["YEAR"] = df_header["TIMESTAMP"].dt.year
df_header["MONTH"] = df_header["TIMESTAMP"].dt.month_name()
df_header["YEAR_MONTH"] = df_header["TIMESTAMP"].dt.to_period("M").astype(str)
df_header["YEAR_QUARTER"] = df_header["TIMESTAMP"].dt.to_period("Q").astype(str)

for col in ['DEVICE', 'PRODUCT', 'LOT_NUMBER', 'EMPLOYEE_ID', 'RECIPE_NAME','STANDARD_NAME']:
    df_header[col] = df_header[col].astype(str).str.strip().str.title()

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css",
        "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-alpine.css"
    ]
)

# 🔁 Top-level layout with Tabs
app.layout = dbc.Container([
    dcc.Tabs(id='main-tabs', value='tab-1', children=[
        dcc.Tab(label="📈 Overview & Yield", value='tab-1'),
        dcc.Tab(label="📷 Camera & Reject Analysis", value='tab-2'),
    ]),
    html.Div(id='tab-content')
], fluid=True)

# ✅ Section 1 layout wrapped in a function
def layout_section1():
    return dbc.Container([
        dcc.Store(id="reset-trigger"),
        dcc.Store(id="selected-defect"),
        html.H3("📊 Interactive Yield & Defect Dashboard", className="mt-4 mb-4"),

        dbc.Row([
            dbc.Col(html.Label("🎯 Target Yield Threshold (%)"), md=3),
            dbc.Col(dcc.Slider(id='target-slider', min=90, max=100, step=0.5, value=97,
                            marks={i: str(i) for i in range(90, 101)}, tooltip={"placement": "bottom"}), md=9),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Dropdown(id='filter-device', multi=True, placeholder="Device",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['DEVICE'].dropna().unique())]), md=2),
            dbc.Col(dcc.Dropdown(id='filter-product', multi=True, placeholder="Product",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['PRODUCT'].dropna().unique())]), md=2),
            dbc.Col(dcc.Dropdown(id='filter-year', multi=True, placeholder="Year",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['YEAR'].dropna().unique())]), md=2),
            dbc.Col(dcc.Dropdown(id='filter-month', multi=True, placeholder="Month",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['MONTH'].dropna().unique())]), md=2),
            dbc.Col(dcc.Dropdown(id='filter-lot', multi=True, placeholder="Lot Number",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['LOT_NUMBER'].dropna().unique())]), md=2),
            dbc.Col(dcc.Dropdown(id='filter-employee', multi=True, placeholder="Employee ID",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['EMPLOYEE_ID'].dropna().unique())]), md=2),
        ], className="g-2 mb-2"),

        dbc.Row([
            dbc.Col(dcc.Dropdown(id='filter-recipe', multi=True, placeholder="Recipe Name",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['RECIPE_NAME'].dropna().unique())]), md=2),
            dbc.Col(dcc.Dropdown(id='filter-machine_name', multi=True, placeholder="Machine Name",
                                options=[{'label': i, 'value': i} for i in sorted(df_header['STANDARD_NAME'].dropna().unique())]), md=2),
            dbc.Col(html.Button("Apply Filters", id="apply-filters", className="btn btn-primary w-100"), md=2),
            dbc.Col(html.Button("Reset Filters", id="reset-filters", className="btn btn-secondary w-100"), md=2),
        ], className="g-2 mb-4"),

        dbc.Row([
            dbc.Col(dcc.Dropdown(id='time-view', value='MONTH', options=[
                {'label': 'Monthly', 'value': 'MONTH'},
                {'label': 'Quarterly', 'value': 'QUARTER'},
                {'label': 'Yearly', 'value': 'YEAR'}
            ], clearable=False), md=3),
            dbc.Col(dcc.Dropdown(id='defect-metric', value='QTY', options=[
                {'label': 'Total Quantity', 'value': 'QTY'},
                {'label': 'Lot Count', 'value': 'COUNT'}
            ], clearable=False), md=3),
        ], className="mb-4"),

        dbc.Row(id='kpi-cards', className="mb-4"),
        html.Div(id='insight-card'),
        html.Div(id='yield-alert'),

        dbc.Row([
            dbc.Col(dcc.Graph(id="yield-trend", clear_on_unhover=True), md=6),
            dbc.Col(dcc.Graph(id="qty-bar"), md=6),
        ], className="mb-4"),

        html.H5("🔝 Top 10 Defect Parameters"),
        dcc.Graph(id='top-defects', className="mb-4"),

        html.H5("📋 Lots Below Target Yield"),
        dag.AgGrid(
            id='low-yield-table',
            columnDefs=[], rowData=[],
            columnSize="sizeToFit",
            className="ag-theme-alpine mb-5",
            dashGridOptions={"pagination": True, "paginationPageSize": 15, "rowHeight": 35},
            style={"height": "500px", "width": "100%", "border": "1px solid #ccc"}
        ),

        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("🔍 Lots for Selected Defect")),
                dbc.ModalBody(dag.AgGrid(id='defect-lot-table', columnSize="sizeToFit")),
                dbc.ModalFooter(dbc.Button("Close", id="close-defect-modal", className="ms-auto", n_clicks=0)),
            ],
            id="defect-modal",
            size="xl",
            is_open=False,
        )
    ], fluid=True)


@app.callback(
    Output("apply-filters", "n_clicks"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True
)
def auto_trigger_apply(n):
    return n or 1

@app.callback(
    Output('filter-device', 'value'), Output('filter-product', 'value'),
    Output('filter-year', 'value'), Output('filter-month', 'value'),
    Output('filter-lot', 'value'), Output('filter-employee', 'value'),
    Output('filter-recipe', 'value'), Output('time-view', 'value'),Output('filter-machine_name', 'value'),
    Output('defect-metric', 'value'),
    Input('reset-filters', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n):
    return [None] * 8 + ['MONTH', 'QTY']

@app.callback(
    Output('yield-trend', 'clickData'),
    Output('top-defects', 'clickData'),
    Input('reset-filters', 'n_clicks'),
    prevent_initial_call=True
)
def clear_clickdata(n):
    return None, None

@app.callback(
    Output('kpi-cards', 'children'),
    Output('yield-alert', 'children'),
    Output('insight-card', 'children'), 
    Output('yield-trend', 'figure'),
    Output('qty-bar', 'figure'),
    Output('top-defects', 'figure'),
    Output('low-yield-table', 'columnDefs'),
    Output('low-yield-table', 'rowData'),
    Input('apply-filters', 'n_clicks'),
    State('target-slider', 'value'),
    State('filter-device', 'value'), State('filter-product', 'value'),
    State('filter-year', 'value'), State('filter-month', 'value'),
    State('filter-lot', 'value'), State('filter-employee', 'value'),
    State('filter-recipe', 'value'),State('filter-machine_name', 'value'),
    State('yield-trend', 'clickData'), State('top-defects', 'clickData'),
    State('time-view', 'value'), State('defect-metric', 'value')
)
def update_dashboard(n, target, device, product, year, month, lot, employee, recipe,standard_name, trend_click, defect_click, time_view, defect_metric):
    dff = df_header.copy()

    if ctx.triggered_id == 'reset-filters':
        trend_click = None
        defect_click = None

    if device: dff = dff[dff['DEVICE'].isin(device)]
    if product: dff = dff[dff['PRODUCT'].isin(product)]
    if year: dff = dff[dff['YEAR'].isin(year)]
    if month: dff = dff[dff['MONTH'].isin(month)]
    if lot: dff = dff[dff['LOT_NUMBER'].isin(lot)]
    if employee: dff = dff[dff['EMPLOYEE_ID'].isin(employee)]
    if recipe: dff = dff[dff['RECIPE_NAME'].isin(recipe)]
    if standard_name:
        if isinstance(standard_name, str):
            standard_name = [standard_name]
        dff = dff[dff["STANDARD_NAME"].isin(standard_name)]


    if trend_click and 'points' in trend_click:
        clicked_month = trend_click['points'][0].get('customdata', [None])[0]
        if clicked_month:
            dff = dff[dff['YEAR_MONTH'] == clicked_month]

    if defect_click and 'points' in defect_click:
        clicked_param = defect_click['points'][0].get('customdata', [None])[0]
        if clicked_param:
            matched_ids = df_summary[df_summary['PARAMETER'] == clicked_param]['HEADER_ID'].unique()
            dff = dff[dff['HEADER_ID'].isin(matched_ids)]

    if dff.empty:
        empty_fig = px.scatter(title="No Data")
        return [], None, None, empty_fig, empty_fig, empty_fig, [], []

    total_in = dff['TOTAL_IN_QTY'].sum()
    total_out = dff['TOTAL_OUT_QTY'].sum()
    yield_pct = (total_out / total_in) * 100 if total_in > 0 else 0
    below_target = dff[dff['TOTAL_OUT_PCT'] < target].shape[0]

    kpis = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("🧮 Total Input Quantity"), dbc.CardBody(f"{total_in/1e6:.2f}M")], color="primary", inverse=True), md=3),
        dbc.Col(dbc.Card([dbc.CardHeader("📦 Total Output Quantity"), dbc.CardBody(f"{total_out/1e6:.2f}M")], color="success", inverse=True), md=3),
        dbc.Col(dbc.Card([dbc.CardHeader("📈 Overall Yield Percentage"), dbc.CardBody(f"{yield_pct:.2f}%")], color="info", inverse=True), md=3),
        dbc.Col(dbc.Card([dbc.CardHeader(f"⚠️ Lots < {target}% Yield"), dbc.CardBody(str(below_target))], color="danger", inverse=True), md=3),
    ])

    group_col = {'MONTH': 'YEAR_MONTH', 'QUARTER': 'YEAR_QUARTER', 'YEAR': 'YEAR'}[time_view]
    trend_df = dff.groupby(group_col).agg(YIELD_PCT=('TOTAL_OUT_PCT', 'mean')).reset_index()
    trend_fig = px.line(trend_df, x=group_col, y='YIELD_PCT', markers=True, custom_data=[group_col])
    trend_fig.update_traces(hovertemplate="Period: %{x}<br>Yield: %{y:.2f}%<extra></extra>")

    qty_df = dff.groupby(group_col).agg(IN_QTY=('TOTAL_IN_QTY', 'sum'), OUT_QTY=('TOTAL_OUT_QTY', 'sum')).reset_index()
    bar_fig = px.bar(qty_df, x=group_col, y=['IN_QTY', 'OUT_QTY'], barmode='group')

    filtered_summary = df_summary[df_summary['HEADER_ID'].isin(dff['HEADER_ID'])]
    if defect_metric == 'QTY':
        top_defects = filtered_summary.groupby("PARAMETER")["QTY"].sum().nlargest(10).reset_index()
    else:
        top_defects = filtered_summary.groupby("PARAMETER")["HEADER_ID"].nunique().nlargest(10).reset_index(name='QTY')
    defect_fig = px.bar(top_defects, x="PARAMETER", y="QTY", text_auto=True, custom_data=["PARAMETER"])

    table_df = dff[dff['TOTAL_OUT_PCT'] < target]
    table_cols = ['HEADER_ID', 'LOT_NUMBER', 'PRODUCT', 'DEVICE', 'TOTAL_OUT_PCT', 'EMPLOYEE_ID', 'RECIPE_NAME', 'TIMESTAMP','STANDARD_NAME']
    col_defs = [{
    "headerName": col,
    "field": col,
    "sortable": True,
    "filter": 'agTextColumnFilter',
    "resizable": True,
    "floatingFilter": True,
    "wrapText": False,          # Disabled to prevent overlap
    "autoHeight": False         # Disabled to keep row height consistent


    
} for col in table_cols]
    row_data = table_df[table_cols].to_dict("records")
    
    low_df = dff[dff['TOTAL_OUT_PCT'] < target]
    low_summary = df_summary[df_summary['HEADER_ID'].isin(low_df['HEADER_ID'])]
    if not low_summary.empty:
        top_defects = low_summary.groupby("PARAMETER")["QTY"].sum().nlargest(2)
        top_names = top_defects.index.tolist()
        top_pct = top_defects.sum() / low_summary["QTY"].sum() * 100
        insight_text = f"{' & '.join(top_names)} account for {top_pct:.1f}% of defects in low-yield lots."
    else:
        insight_text = "No critical defects found in current filter."

    insight_card = html.Div(
        dbc.Card([
            dbc.CardHeader("🧠 Smart Insight"),
            dbc.CardBody(insight_text)
        ], color="secondary", inverse=True),
        className="mb-4"
    )
    yield_alert = None

    if time_view == 'MONTH' and not trend_df.empty and trend_df.shape[0] >= 2:
        last_two = trend_df.tail(2)
        prev_yield = last_two.iloc[0]['YIELD_PCT']
        curr_yield = last_two.iloc[1]['YIELD_PCT']
        drop = prev_yield - curr_yield

        if drop >= 5.0:
            yield_alert = dbc.Alert(
                f"⚠️ Average Yield dropped by {drop:.2f}% compared to the previous month.",
                color="danger", className="mb-4"
        )
    return insight_card, yield_alert, kpis, trend_fig, bar_fig, defect_fig, col_defs, row_data

@app.callback(
    Output("selected-defect", "data"),
    Input("top-defects", "clickData"),
    prevent_initial_call=True
)
def capture_clicked_defect(clickData):
    if clickData and "points" in clickData:
        return clickData["points"][0]["customdata"][0]
    return None

@app.callback(
    Output("defect-modal", "is_open"),
    Output("defect-lot-table", "rowData"),
    Output("defect-lot-table", "columnDefs"),
    Input("selected-defect", "data"),
    Input("close-defect-modal", "n_clicks"),
    State("defect-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(defect, close_click, is_open):
    ctx_id = ctx.triggered_id

    if ctx_id == "selected-defect" and defect:
        matching_ids = df_summary[df_summary["PARAMETER"] == defect]["HEADER_ID"].unique()
        table_df = df_header[df_header["HEADER_ID"].isin(matching_ids)]

        display_cols = ['HEADER_ID', 'LOT_NUMBER', 'PRODUCT', 'DEVICE', 'EMPLOYEE_ID', 'RECIPE_NAME', 'TOTAL_OUT_PCT', 'TIMESTAMP','STANDARD_NAME']
        col_defs = [{"field": col, "headerName": col, "filter": True} for col in display_cols]
        row_data = table_df[display_cols].to_dict("records")

        return True, row_data, col_defs

    if ctx_id == "close-defect-modal":
        return False, [], []

    return is_open, no_update, no_update

def layout_section2():
    return dbc.Container([
        html.H3("📷 Camera & Reject Analysis", className="mt-4 mb-4"),

        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id="heatmap-group-col",
                options=[
                    {'label': 'Camera', 'value': 'CAMERA'},
                    {'label': 'Machine Name', 'value': 'STANDARD_NAME'},
                    {'label': 'Employee ID', 'value': 'EMPLOYEE_ID'},
                    {'label': 'Recipe Name', 'value': 'RECIPE_NAME'},
                    {'label': 'Device', 'value': 'DEVICE'},
                ],
                value='CAMERA',
                clearable=False
            ), md=4),
        ], className="mb-3"),

    # 🔧 Scrolling container for heatmap
    html.Div(
    dcc.Graph(
        id="parameter-heatmap",
        config={"displayModeBar": True},
        style={
            "height": "1200px",   # 👈 increase height for bigger boxes
            "width": "2200px",    # 👈 increase width for more space per column
        }
    ),
    style={
        "overflowX": "auto",      # horizontal scroll inside container
        "overflowY": "auto",      # vertical scroll inside container
        "maxHeight": "700px",     # max height before vertical scroll appears
        "maxWidth": "100%",
        "border": "1px solid #ccc",
        "padding": "10px"
        }

    ),
        html.Div(id='detection-smart-text', className='mt-3'),

        html.H5("📊 Final Rejects by Parameter"),
        dcc.Graph(id="final-reject-bar"),

        html.H5("📋 Reject Root Cause Explorer"),
        dag.AgGrid(
            id='reject-root-table',
            columnDefs=[], rowData=[],
            columnSize="sizeToFit",
            className="ag-theme-alpine mb-4",
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 15,
                "rowHeight": 35
            },
            style={"height": "500px", "width": "100%", "border": "1px solid #ccc"}
        )
    ])

@app.callback(
    Output("parameter-heatmap", "figure"),
    Input("heatmap-group-col", "value")
)
def update_heatmap(group_by_col):
    if df_unpivot.empty:
        return go.Figure()

    # Limit size to prevent crash
    max_columns = 30
    max_parameters = 30

    top_values = df_unpivot[group_by_col].unique()
    top_params = df_unpivot["PARAMETER"].unique()

    df_heat = df_unpivot[
        df_unpivot[group_by_col].isin(top_values) &
        df_unpivot["PARAMETER"].isin(top_params)
    ]
    grouped = df_heat.groupby(["PARAMETER", group_by_col])["COUNT"].sum().reset_index()
    heat_df = grouped.pivot(index="PARAMETER", columns=group_by_col, values="COUNT").fillna(0)
    heat_df = heat_df[heat_df.sum().sort_values(ascending=False).index]

    # 🧠 Format numbers (e.g. 12.3k, 1.2M)
    def format_val(v):
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        elif v >= 1_000:
            return f"{v/1_000:.1f}k"
        else:
            return str(int(v))

    text_matrix = heat_df.applymap(format_val).values

    fig = go.Figure(data=go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns,
        y=heat_df.index,
        text=text_matrix,
        texttemplate="%{text}",   # 👈 this shows the value in each cell
        hovertemplate='Parameter: %{y}<br>%{x}: %{z}<extra></extra>',
        colorscale="YlOrRd",
        showscale=True
    ))

    fig.update_layout(
        title=f"Heatmap: Defect Parameter vs {group_by_col}",
        width=1000,
        height=800,
        margin=dict(t=50, l=150, r=30, b=50),
        font=dict(size=14)
    )

    return fig

@app.callback(
    Output("final-reject-bar", "figure"),
    Input('apply-filters', 'n_clicks'),
    State('filter-device', 'value'), State('filter-product', 'value'),
    State('filter-year', 'value'), State('filter-month', 'value'),
    State('filter-lot', 'value'), State('filter-employee', 'value'),
    State('filter-recipe', 'value'), State('filter-machine_name', 'value')
)
def update_final_reject_chart(n, device, product, year, month, lot, employee, recipe, standard_name):
    dff = df_header.copy()

    if device: dff = dff[dff['DEVICE'].isin(device)]
    if product: dff = dff[dff['PRODUCT'].isin(product)]
    if year: dff = dff[dff['YEAR'].isin(year)]
    if month: dff = dff[dff['MONTH'].isin(month)]
    if lot: dff = dff[dff['LOT_NUMBER'].isin(lot)]
    if employee: dff = dff[dff['EMPLOYEE_ID'].isin(employee)]
    if recipe: dff = dff[dff['RECIPE_NAME'].isin(recipe)]
    if standard_name:
        if isinstance(standard_name, str):
            standard_name = [standard_name]
        dff = dff[dff["STANDARD_NAME"].isin(standard_name)]

    matched_ids = dff["HEADER_ID"].unique()
    summary_filtered = df_summary[df_summary["HEADER_ID"].isin(matched_ids)]

    if summary_filtered.empty:
        return px.bar(title="No Data")

    top_params = summary_filtered.groupby("PARAMETER")["QTY"].sum().nlargest(10).reset_index()

    fig = px.bar(top_params, x="PARAMETER", y="QTY", text_auto=True)
    fig.update_layout(title="Top Final Reject Parameters", xaxis_title="Parameter", yaxis_title="Reject QTY")

    return fig



@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-1':
        return layout_section1()
    elif tab == 'tab-2':
        return layout_section2()


if __name__ == '__main__':
    app.run(debug=True)
