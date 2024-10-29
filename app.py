import networkx as nx
import pandas as pd
from community import community_louvain
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash_table import DataTable
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load student data
student_performance_df = pd.read_csv("StudentPerformanceFactors.csv")

# Add calculated columns
student_performance_df['Performance_Rating'] = pd.cut(
    student_performance_df['Exam_Score'],
    bins=[0, 50, 75, 100],
    labels=['Low', 'Medium', 'High']
)
student_performance_df['Study_Efficiency'] = student_performance_df['Exam_Score'] / student_performance_df['Hours_Studied']

# 1. Visualisation de la distribution des scores d'examen
fig_distribution = px.histogram(
    student_performance_df, x="Exam_Score", nbins=20,
    title="Distribution des Scores d'Examen"
)

# 2. Clustering K-means avec interaction
# Standardisation des données pour K-means
features = student_performance_df[['Hours_Studied', 'Exam_Score']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

def perform_kmeans(n_clusters):
    """Applique K-means avec le nombre de clusters spécifié."""
    kmeans = KMeans(n_clusters=n_clusters)
    student_performance_df['Cluster'] = kmeans.fit_predict(features_scaled)

    # Création du graphique de clustering
    fig_clusters = px.scatter(
        student_performance_df, x="Hours_Studied", y="Exam_Score", color="Cluster",
        title=f"Clustering des Étudiants avec {n_clusters} Clusters (K-means)"
    )
    return fig_clusters

# Initial clustering plot with default 3 clusters
fig_clusters = perform_kmeans(3)

# Initialize a graph and community detection
G = nx.Graph()
for index in student_performance_df.index:
    G.add_node(f"Student_{index}")

grouped_students = student_performance_df.groupby(['Motivation_Level', 'School_Type', 'Gender'])
for _, group in grouped_students:
    indices = group.index.tolist()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            G.add_edge(f"Student_{indices[i]}", f"Student_{indices[j]}", weight=1)

partition = community_louvain.best_partition(G)
student_performance_df['community'] = [partition.get(f"Student_{idx}") for idx in range(len(student_performance_df))]

# Dash app setup
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create initial visualizations
scatter_plot = px.scatter(
    student_performance_df,
    x="Hours_Studied",
    y="Exam_Score",
    color="community",
    title="Exam Score vs. Hours Studied by Community"
)

# Correlation heatmap for numeric variables
correlation_matrix = student_performance_df[['Hours_Studied', 'Exam_Score', 'Attendance', 'Previous_Scores']].corr()

# Create the annotated heatmap with improved readability
heatmap_fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns.tolist(),
    y=correlation_matrix.columns.tolist(),
    colorscale="Viridis",
    showscale=True,
    annotation_text=correlation_matrix.round(2).values  # Round to two decimal places
)

# Update the layout to improve annotation readability
heatmap_fig.update_layout(
    title="Correlation Heatmap of Numeric Factors",
    font=dict(size=12)  # Set general font size
)

# Change annotation colors and add contrast based on background color
for i in range(len(heatmap_fig.layout.annotations)):
    # Set font color to black if the background is light, otherwise white
    if heatmap_fig.layout.annotations[i].text in ["0.0", "-0.0", "1.0"]:
        heatmap_fig.layout.annotations[i].font.size = 10
    else:
        heatmap_fig.layout.annotations[i].font.size = 12
        heatmap_fig.layout.annotations[i].font.color = "white" if float(heatmap_fig.layout.annotations[i].text) < 0.5 else "black"

# Layout for Dash app
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Student Performance Dashboard", className="text-center"))]),

    # Section pour la distribution des scores d'examen
    dbc.Row([
        dbc.Col(dcc.Graph(id="distribution-plot", figure=fig_distribution), width=6),
        dbc.Col(dcc.Graph(id="correlation-heatmap", figure=heatmap_fig), width=6),
    ]),

    # Section pour le clustering K-means
    dbc.Row([
        dbc.Col(dcc.Graph(id="scatter-hours-vs-scores", figure=scatter_plot), width=6),
        dbc.Col([
            dcc.Graph(id="cluster-plot", figure=fig_clusters),
            dcc.Slider(id="cluster-slider", min=2, max=10, step=1, value=3,
                       marks={i: str(i) for i in range(2, 11)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col(html.H3("Student Data Table"), width=12),
        dbc.Col(dcc.Dropdown(
            id="community-filter",
            options=[{'label': f'Community {i}', 'value': i} for i in student_performance_df['community'].unique()],
            placeholder="Select a Community",
            multi=True
        ), width=4),
        dbc.Col(dcc.Dropdown(
            id="motivation-filter",
            options=[{'label': level, 'value': level} for level in student_performance_df['Motivation_Level'].unique()],
            placeholder="Select Motivation Level",
            multi=True
        ), width=4),
        dbc.Col(dcc.Dropdown(
            id="school-filter",
            options=[{'label': school, 'value': school} for school in student_performance_df['School_Type'].unique()],
            placeholder="Select School Type",
            multi=True
        ), width=4),
    ]),
    dbc.Row([dbc.Col(DataTable(
        id='student-table',
        columns=[{"name": i, "id": i} for i in student_performance_df.columns],
        data=student_performance_df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    ), width=12)])
])

# Callback for updating clustering based on the slider value
@app.callback(
    Output("cluster-plot", "figure"),
    [Input("cluster-slider", "value")]
)
def update_kmeans_clusters(n_clusters):
    return perform_kmeans(n_clusters)

# Callbacks for filtering table based on dropdown selections
@app.callback(
    Output("student-table", "data"),
    [Input("community-filter", "value"),
     Input("motivation-filter", "value"),
     Input("school-filter", "value")]
)
def update_table(selected_communities, selected_motivation, selected_school):
    filtered_df = student_performance_df.copy()
    if selected_communities:
        filtered_df = filtered_df[filtered_df["community"].isin(selected_communities)]
    if selected_motivation:
        filtered_df = filtered_df[filtered_df["Motivation_Level"].isin(selected_motivation)]
    if selected_school:
        filtered_df = filtered_df[filtered_df["School_Type"].isin(selected_school)]
    return filtered_df.to_dict("records")

if __name__ == "__main__":
    app.run_server(debug=True)
