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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load student data
student_performance_df = pd.read_csv("StudentPerformanceFactors.csv")

# Standardisation des données pour K-means
features = student_performance_df[['Hours_Studied', 'Exam_Score']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialisation des transformations PCA et t-SNE pour visualisation 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
student_performance_df['PCA1'] = pca_result[:, 0]
student_performance_df['PCA2'] = pca_result[:, 1]

tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=0)
tsne_result = tsne.fit_transform(features_scaled)
student_performance_df['TSNE1'] = tsne_result[:, 0]
student_performance_df['TSNE2'] = tsne_result[:, 1]

# Graphique par défaut pour la distribution
fig_distribution = px.histogram(
    student_performance_df, x="Exam_Score", nbins=20,
    title="Distribution des Scores d'Examen"
)

# Correlation heatmap for numeric variables
correlation_matrix = student_performance_df[['Hours_Studied', 'Exam_Score', 'Attendance', 'Previous_Scores']].corr()
heatmap_fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns.tolist(),
    y=correlation_matrix.columns.tolist(),
    colorscale="Viridis",
    showscale=True,
    annotation_text=correlation_matrix.round(2).values
)
heatmap_fig.update_layout(title="Correlation Heatmap of Numeric Factors", font=dict(size=12))

# Dash app setup
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout for Dash app
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Student Performance Dashboard", className="text-center"))]),

    # Section pour la distribution des scores d'examen et la heatmap de corrélation
    dbc.Row([
        dbc.Col(dcc.Graph(id="distribution-plot", figure=fig_distribution), width=6),
        dbc.Col(dcc.Graph(id="correlation-heatmap", figure=heatmap_fig), width=6),
    ]),

    # Visualisation PCA / t-SNE avec clustering K-means
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id="dimensionality-method",
            options=[{'label': 'PCA', 'value': 'PCA'}, {'label': 't-SNE', 'value': 't-SNE'}],
            value='PCA',
            labelStyle={'display': 'inline-block'}
        )),
        dbc.Col(dcc.Slider(
            id="cluster-slider", min=2, max=10, step=1, value=3,
            marks={i: str(i) for i in range(2, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        )),
        dbc.Col(dcc.Graph(id="dimensionality-plot"), width=12),
    ]),

    # Tableau de données et filtres
    dbc.Row([
        dbc.Col(html.H3("Student Data Table"), width=12),
    ]),
    dbc.Row([dbc.Col(DataTable(
        id='student-table',
        columns=[{"name": i, "id": i} for i in student_performance_df.columns],
        data=student_performance_df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
    ), width=12)]),
])

# Callback pour mettre à jour la distribution
@app.callback(
    Output("distribution-plot", "figure"),
    [Input("community-filter", "value"),
     Input("motivation-filter", "value"),
     Input("school-filter", "value")]
)
def update_histogram(selected_communities, selected_motivation, selected_school):
    filtered_df = student_performance_df.copy()
    if selected_communities:
        filtered_df = filtered_df[filtered_df["community"].isin(selected_communities)]
    if selected_motivation:
        filtered_df = filtered_df[filtered_df["Motivation_Level"].isin(selected_motivation)]
    if selected_school:
        filtered_df = filtered_df[filtered_df["School_Type"].isin(selected_school)]

    fig_distribution = px.histogram(
        filtered_df, x="Exam_Score", nbins=20,
        title="Distribution des Scores d'Examen",
        labels={"Exam_Score": "Score d'Examen"},
    )
    return fig_distribution

# Callback pour la sélection PCA/t-SNE avec le nombre de clusters
@app.callback(
    Output("dimensionality-plot", "figure"),
    [Input("dimensionality-method", "value"),
     Input("cluster-slider", "value")]
)
def update_dimensionality_plot(method, n_clusters):
    # Appliquer le clustering K-means avec le nombre de clusters sélectionné
    kmeans = KMeans(n_clusters=n_clusters)
    student_performance_df['Cluster'] = kmeans.fit_predict(features_scaled)

    # Sélectionner la projection en fonction de la méthode (PCA ou t-SNE)
    if method == 'PCA':
        fig = px.scatter(
            student_performance_df, x="PCA1", y="PCA2", color="Cluster",
            title=f"Clustering avec PCA ({n_clusters} clusters)"
        )
    else:
        fig = px.scatter(
            student_performance_df, x="TSNE1", y="TSNE2", color="Cluster",
            title=f"Clustering avec t-SNE ({n_clusters} clusters)"
        )
    return fig

# Callback pour mettre à jour le tableau de données en fonction des filtres sélectionnés
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

# Démarrer le serveur Dash
if __name__ == "__main__":
    app.run_server(debug=True)

