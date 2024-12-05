import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import pickle
import math


# Access the Flask server object
server = app.server

# Load data
with open('PHI_NORMALIZED.pkl', 'rb') as f:
    PHI_NORMALIZED = pickle.load(f)

# Read TT_FINAL with specified dtypes and encoding
TT_FINAL = pd.read_csv('TT_FINAL.csv', encoding='latin1')
TT_FINAL['year'] = TT_FINAL['year'].astype(str).str.zfill(2)

# Extract unique country-year combinations
country_year_options = sorted(TT_FINAL['full_country_year'].unique())
country_year_options = [{'label': ctyr, 'value': ctyr} for ctyr in country_year_options]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Poverty Space Visualization", style={'textAlign': 'center','fontFamily': 'Arial'}),
    # Input controls
    html.Div([
        # Wrap label and dropdown in a Div
        html.Div([
            html.Label("Select Country and Year:", style={
                'fontFamily': 'Arial',
                'fontSize': '14px',
                'color': 'black',
                'textAlign': 'center',
                'display': 'block',
                'marginBottom': '10px'
            }),
            dcc.Dropdown(
                id='country-year-dropdown',
                options=country_year_options,
                value='Kyrgyz Republic 2006',
                placeholder='Select a country and year',
                style={
                    'width': '300px',
                    'fontFamily': 'Arial',
                    'fontSize': '14px',
                    'color': 'black',
                    'display': 'block',
                    'marginLeft': 'auto',
                    'marginRight': 'auto'
                },
                searchable=True,
                clearable=True
            ),
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center'
        }),
        # Center the button as well
        html.Button('Update Graph', id='update-button', n_clicks=0, style={
            'fontFamily': 'Arial',
            'fontSize': '14px',
            'color': 'black',
            'marginTop': '20px',
            'display': 'block',
            'marginLeft': 'auto',
            'marginRight': 'auto'
        }),
    ], style={
        'textAlign': 'center',
        'marginTop': '20px',
        'marginBottom': '20px',
    }),
    # Optional horizontal line
    html.Hr(style={'width': '80%', 'margin': '0 auto'}),
    # Description and graph
    html.Div([
        dcc.Markdown(
            id='description',
            style={
                'textAlign': 'justify',
                'fontSize': '16px',
                'lineHeight': '1.6',
                'marginTop': '20px',
                'fontFamily': 'Arial',
                'color': 'black'
            }
        ),
        dcc.Graph(id='network-graph'),
        # Add download buttons
        html.Div([
            html.Button('Download Poverty Space Data', id='download-phi-button', n_clicks=0, style={
                'fontFamily': 'Arial',
                'fontSize': '14px',
                'color': 'black',
                'marginRight': '10px'
            }),
            html.Button('Download Poverty Centrality Data', id='download-filtered-button', n_clicks=0, style={
                'fontFamily': 'Arial',
                'fontSize': '14px',
                'color': 'black'
            }),
            # Include Download components
            dcc.Download(id='download-phi'),
            dcc.Download(id='download-filtered')
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
    ], style={'width': '800px', 'margin': '0 auto'}),
    dcc.Store(id='selected-node'),
    # Add dcc.Store components to store data frames
    dcc.Store(id='phi-data'),
    dcc.Store(id='filtered-data')
])

# Define the callback to update the graph
@app.callback(
    [Output('network-graph', 'figure'),
     Output('selected-node', 'data'),
     Output('description', 'children'),
     Output('phi-data', 'data'),
     Output('filtered-data', 'data')],
    [Input('update-button', 'n_clicks'),
     Input('network-graph', 'clickData')],
    [State('country-year-dropdown', 'value'),
     State('selected-node', 'data')]
)

def update_graph(n_clicks, clickData, ctry_yr,selected_node):
    try:
        rows = TT_FINAL[TT_FINAL['full_country_year'] == ctry_yr].copy()
        ctry_init = rows['country'].unique()[0]
        ctry_full = rows['country_full'].unique()[0]
        yyr_init = rows['year'].unique()[0]
        yyr_full = rows['year_full'].unique()[0]
        key = f'phi_{ctry_init}_{yyr_init}'
        # Ensure the key exists in your data
        if key not in PHI_NORMALIZED:
            print(f"Key {key} not found in PHI_NORMALIZED")
            return go.Figure(), selected_node  # Return empty figure if key not found

        PHI_w_norm = PHI_NORMALIZED[key].copy()
        filtered_df = TT_FINAL[(TT_FINAL['year'] == yyr_init) & (TT_FINAL['country'] == ctry_init)]
        if filtered_df.empty:
            print(f"No data found for country {ctry_init} and year {yyr_init}")
            return go.Figure(), selected_node  # Return empty figure if no data

        # Extract necessary data from filtered_df
        A_int = filtered_df['A_int'].copy()
        A_int_flat = A_int.copy()
        var_sample = filtered_df[['full_names']].copy()
        centrality_w_norm_flat = filtered_df['centrality_w_norm'].copy()

        # Normalize and scale A_int values for coloring
        pos_colors = zscore(A_int.to_numpy())
        scaler = MinMaxScaler(feature_range=(1, 100))
        pos_colors_scaled = scaler.fit_transform(pos_colors.reshape(-1, 1)).flatten().round().astype(int)

        # Use a matplotlib colormap similar to MATLAB's parula
        cmap = plt.cm.get_cmap('viridis', 100)
        node_colors = np.array([cmap(pos_colors_scaled[k]-1) for k in range(len(pos_colors_scaled))])

        # Convert to Plotly colorscale
        plotly_cmap = [[i/99, f'rgb({c[0]*255},{c[1]*255},{c[2]*255})'] for i, c in enumerate(cmap.colors)]

        # Node labels
        node_labels = var_sample['full_names'].tolist()
        node_labels_dict = {i: label for i, label in enumerate(node_labels)}

        # Graph threshold and construction
        if PHI_w_norm.size == 0:
            print(f"PHI_w_norm for key {key} is empty.")
            return go.Figure(), selected_node

        # Threshold PHI_w_norm to create XX
        thresh = np.min(np.max(PHI_w_norm, axis=1))
        XX = np.copy(PHI_w_norm)
        XX = XX + XX.T
        thresh = np.min(np.max(XX, axis=1))
        XX = np.copy(XX)
        XX[XX < thresh] = 0
        XX = (XX > 0).astype(float)
       

        # Create the graph with edge weights
        G = nx.from_numpy_array(XX)

        # Hover texts
        hover_texts = []
        for i in range(len(G.nodes())):
            label = node_labels_dict[i]
            CHR_value = A_int_flat.iloc[i] if isinstance(A_int_flat, pd.Series) else A_int_flat[i]
            PC_value = centrality_w_norm_flat.iloc[i] if isinstance(centrality_w_norm_flat, pd.Series) else centrality_w_norm_flat[i]
            hover_text = f"{label}<br>CHR = {CHR_value:.2f}<br>PC = {PC_value:.2f}"
            hover_texts.append(hover_text)

        # Fix random state for layout
        pos = nx.spring_layout(G, seed=42)

        # Extract edge weights and map to line widths
        edge_weights = []
        for edge in G.edges(data=True):
            weight = edge[2].get('weight', 1)
            edge_weights.append(weight)

        edge_weights = np.array(edge_weights)
        edge_weights = np.nan_to_num(edge_weights, nan=0)

        # Normalize edge weights to line widths
        scaler = MinMaxScaler(feature_range=(1, 5))  # Adjust range as needed
        edge_widths = scaler.fit_transform(edge_weights.reshape(-1, 1)).flatten()

        # Define the generate_arc function for smooth curves
        def generate_arc(x0, y0, x1, y1, num_points=50, curvature=0.2):
            # Calculate midpoint
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2

            # Calculate distance between points
            dx = x1 - x0
            dy = y1 - y0
            d = np.sqrt(dx**2 + dy**2)

            # Calculate angle perpendicular to the line
            angle = np.arctan2(dy, dx) + np.pi / 2

            # Calculate control point offset
            offset = curvature * d

            # Calculate control point coordinates
            cx = mx + offset * np.cos(angle)
            cy = my + offset * np.sin(angle)

            # Generate t values
            t = np.linspace(0, 1, num_points)

            # Quadratic Bezier curve formula
            x = (1 - t)**2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
            y = (1 - t)**2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1

            return x, y

        # Handle click events
        ctx = dash.callback_context
        if not ctx.triggered:
            event_id = 'No clicks yet'
        else:
            event_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Initialize selected_node if it's None
        if selected_node is None:
            selected_node = {'node': None}

        # Check if the graph was clicked
        if event_id == 'network-graph' and clickData is not None:
            # Get the point number of the clicked node
            point_data = clickData['points'][0]
            point_index = point_data['pointIndex']
            selected_node['node'] = point_index
        else:
            selected_node['node'] = None

        # Determine connected nodes
        connected_nodes = []
        if selected_node['node'] is not None:
            connected_nodes = list(G.neighbors(selected_node['node']))
            connected_nodes.append(selected_node['node'])  # Include the selected node itself

        # Node positions
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Scale node sizes based on centrality
        scaler = MinMaxScaler(feature_range=(10, 50))  # Adjust the range as needed
        centrality_scaled = scaler.fit_transform(centrality_w_norm_flat.values.reshape(-1, 1)).flatten()

        # Update node colors and sizes based on selection
        node_colors_final = []
        node_sizes_final = []
        for i, node in enumerate(G.nodes()):
            if selected_node['node'] is not None:
                if i in connected_nodes:
                    # Highlight connected nodes
                    node_colors_final.append(pos_colors_scaled[i])
                    node_sizes_final.append(centrality_scaled[i])
                else:
                    # Dim non-connected nodes
                    node_colors_final.append('lightgray')
                    node_sizes_final.append(centrality_scaled[i] * 0.5)
            else:
                # No node selected, use default colors and sizes
                node_colors_final.append(pos_colors_scaled[i])
                node_sizes_final.append(centrality_scaled[i])

        # Update edge colors and widths based on selection
        edge_colors_final = []
        edge_widths_final = []
        for i, edge in enumerate(G.edges()):
            if selected_node['node'] is not None:
                if edge[0] == selected_node['node'] or edge[1] == selected_node['node']:
                    # Highlight connected edges
                    edge_colors_final.append('#888')
                    edge_widths_final.append(edge_widths[i])
                else:
                    # Dim non-connected edges
                    edge_colors_final.append('lightgray')
                    edge_widths_final.append(edge_widths[i] * 0.5)
            else:
                # No node selected, use default colors and widths
                edge_colors_final.append('#888')
                edge_widths_final.append(edge_widths[i])

        # Create edge traces with updated colors and widths
        edge_traces = []
        for i, edge in enumerate(G.edges(data=True)):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_width = edge_widths_final[i]
            edge_color = edge_colors_final[i]

            # Generate curved edge
            x_bezier, y_bezier = generate_arc(x0, y0, x1, y1, num_points=50, curvature=0.2)

            edge_trace = go.Scatter(
                x=x_bezier,
                y=y_bezier,
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition="top center",
            text=[node_labels_dict[i] for i in range(len(G.nodes()))],
            hovertext=hover_texts,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale=plotly_cmap,
                size=node_sizes_final,  # Use updated sizes
                color=node_colors_final,  # Use updated colors
                colorbar=dict(
                    thickness=15,
                    xanchor='center',
                    titleside='top',
                    tickvals=[1, 100],
                    ticktext=['Low', 'High'],
                    orientation='h',
                    x=0.5,
                    y=-0.1
                ),
                line_width=2),
            showlegend=False)

        # Create dummy traces for node size legend
        size_legend = [
            dict(size=np.min(centrality_scaled), label='Low Centrality'),
            dict(size=np.median(centrality_scaled), label='Medium Centrality'),
            dict(size=np.max(centrality_scaled), label='High Centrality')
        ]

        legend_traces = []
        for idx, item in enumerate(size_legend):
            legend_traces.append(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=item['size'],
                    color='lightgray',
                    line=dict(width=1, color='black')
                ),
                legendgroup='Node Size',
                showlegend=True,
                name=item['label'],
                hoverinfo='none',  # No hover info for dummy traces
                legendrank=idx  # Ensure ordering
            ))

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)

        # Update layout
        fig.update_layout(
            height=500,
            width=800,
            title=f'{ctry_full} {yyr_full}',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(b=80, l=40, r=40, t=40),
            annotations=[
                dict(
                    text="Censored Headcount Ratio",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.1,
                    xanchor='center',
                    yanchor='bottom',
                    font=dict(
                        family='Arial',
                        size=12,
                        color='black'
                    )
                )
            ],
            font=dict(
                family='Arial',
                size=12,
                color='black'
            ),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(
                    family='Arial',
                    size=12,
                    color='black'
                ),
                # Ensure legend items are ordered as traces are added
                traceorder='normal',
                # Group titles are shown
                groupclick="toggleitem"
            )
        )

        centrality_w_norm_flat = centrality_w_norm_flat.reset_index(drop=True)
        var_sample = var_sample.reset_index(drop=True)

        # Extract the indicators with the highest and lowest PC
        max_index = centrality_w_norm_flat.idxmax()
        min_index = centrality_w_norm_flat.idxmin()

        indicator_max_pc = var_sample['full_names'].iloc[max_index]
        indicator_min_pc = var_sample['full_names'].iloc[min_index]

        # Polished description text
        description_text = f"""
        **Poverty Space Visualization for {ctry_full} in {yyr_full}**

        The figure below shows the Poverty Space for **{ctry_full}** in **{yyr_full}**. Nodes represent poverty indicators, with edges indicating important connections (thresholded to ensure each indicator has at least one connection). Node size reflects Poverty Centrality (PC), and node color corresponds to the Censored Headcount Ratio (CHR) of the indicator.

        In {ctry_full} in {yyr_full}, **{indicator_max_pc}** was the indicator with the highest PC, while **{indicator_min_pc}** had the lowest PC.
        """


        phi_df = pd.DataFrame(PHI_w_norm)
        phi_df.index = node_labels  # Set row labels
        phi_df.columns = node_labels  # Set column labels
        phi_data_json = phi_df.to_json(date_format='iso', orient='split')
        filtered_df = filtered_df.rename(columns={'A_int': 'CHR', 'centrality_w_norm': 'PC', 'full_names': 'indicator'})[['country', 'year', 'indicator', 'CHR', 'PC']]
        filtered_data_json = filtered_df.to_json(date_format='iso', orient='split')

        return fig, selected_node, description_text, phi_data_json, filtered_data_json

    except Exception as e:
        error_message = f"Error processing {ctry_init.upper()} {yyr_init}: {e}"
        print(error_message)
        default_description = f"An error occurred while processing the data for {ctry_init.upper()} in {yyr_init}."
        return go.Figure(), selected_node, default_description, None, None
    

# Callback for downloading PHI_w_norm CSV
@app.callback(
    Output('download-phi', 'data'),
    Input('download-phi-button', 'n_clicks'),
    State('phi-data', 'data'),
    prevent_initial_call=True
)
def download_phi(n_clicks, phi_data_json):
    if phi_data_json is None:
        raise PreventUpdate
    else:
        # Convert JSON back to DataFrame
        phi_df = pd.read_json(phi_data_json, orient='split')
        # Generate CSV data
        return dcc.send_data_frame(phi_df.to_csv, filename='poverty_space.csv')

# Callback for downloading filtered_df CSV
@app.callback(
    Output('download-filtered', 'data'),
    Input('download-filtered-button', 'n_clicks'),
    State('filtered-data', 'data'),
    prevent_initial_call=True
)
def download_filtered(n_clicks, filtered_data_json):
    if filtered_data_json is None:
        raise PreventUpdate
    else:
        # Convert JSON back to DataFrame
        filtered_df = pd.read_json(filtered_data_json, orient='split')
        # Generate CSV data
        return dcc.send_data_frame(filtered_df.to_csv, filename='poverty_centrality.csv', index=False)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
