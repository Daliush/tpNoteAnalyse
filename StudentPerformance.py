import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash_table import DataTable

# Charger les données
student_performance_df = pd.read_csv("StudentPerformanceFactors.csv")

# Prétraitement des données
columns_to_keep = student_performance_df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
features_scaled = scaler.fit_transform(student_performance_df[columns_to_keep])

# Encodage des colonnes catégorielles
features_scaled = pd.get_dummies(
    student_performance_df,
    columns=['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
             'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
             'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level',
             'Distance_from_Home', 'Gender']
)

# PCA pour la réduction de dimension
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
student_performance_df['PCA1'] = pca_result[:, 0]
student_performance_df['PCA2'] = pca_result[:, 1]

# t-SNE pour la réduction de dimension
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
tsne_result = tsne.fit_transform(features_scaled)
student_performance_df['TSNE1'] = tsne_result[:, 0]
student_performance_df['TSNE2'] = tsne_result[:, 1]

# Distribution des scores d'examen
fig_distribution = px.histogram(
    student_performance_df, x="Exam_Score", nbins=20,
    title="Distribution des Scores d'Examen"
)

# Heatmap de corrélation
correlation_matrix = student_performance_df[['Hours_Studied', 'Exam_Score']].corr()
heatmap_fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns.tolist(),
    y=correlation_matrix.columns.tolist(),
    colorscale="Viridis",
    showscale=True,
    annotation_text=correlation_matrix.round(2).values
)
heatmap_fig.update_layout(title="Correlation Heatmap of Numeric Factors")

# Initialiser l'application Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Layout de l'application
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Student Performance Dashboard", className="text-center"))]),

    # Distribution et heatmap
    dbc.Row([
        dbc.Col(dcc.Graph(id="distribution-plot", figure=fig_distribution), width=6),
        dbc.Col(dcc.Graph(id="correlation-heatmap", figure=heatmap_fig), width=6),
    ]),

    # Filtres
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="motivation-filter",
            options=[{'label': m, 'value': m} for m in student_performance_df["Motivation_Level"].unique()],
            multi=True,
            placeholder="Filter by Motivation"
        ), width=6),
        dbc.Col(dcc.Dropdown(
            id="school-filter",
            options=[{'label': s, 'value': s} for s in student_performance_df["School_Type"].unique()],
            multi=True,
            placeholder="Filter by School Type"
        ), width=6),
    ]),

    # PCA / t-SNE et clustering
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

    # Tableau de données
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

# Callbacks
@app.callback(
    Output("distribution-plot", "figure"),
    [Input("motivation-filter", "value"),
     Input("school-filter", "value")]
)
def update_histogram(selected_motivation, selected_school):
    filtered_df = student_performance_df.copy()
    if selected_motivation:
        filtered_df = filtered_df[filtered_df["Motivation_Level"].isin(selected_motivation)]
    if selected_school:
        filtered_df = filtered_df[filtered_df["School_Type"].isin(selected_school)]

    return px.histogram(
        filtered_df, x="Exam_Score", nbins=20,
        title="Distribution des Scores d'Examen"
    )

@app.callback(
    Output("dimensionality-plot", "figure"),
    [Input("dimensionality-method", "value"),
     Input("cluster-slider", "value")]
)
def update_dimensionality_plot(method, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    student_performance_df['Cluster'] = kmeans.fit_predict(features_scaled)

    if method == 'PCA':
        return px.scatter(
            student_performance_df, x="PCA1", y="PCA2", color="Cluster",
            title=f"Clustering avec PCA ({n_clusters} clusters)"
        )
    else:
        return px.scatter(
            student_performance_df, x="TSNE1", y="TSNE2", color="Cluster",
            title=f"Clustering avec t-SNE ({n_clusters} clusters)"
        )

@app.callback(
    Output("student-table", "data"),
    [Input("motivation-filter", "value"),
     Input("school-filter", "value")]
)
def update_table(selected_motivation, selected_school):
    filtered_df = student_performance_df.copy()
    if selected_motivation:
        filtered_df = filtered_df[filtered_df["Motivation_Level"].isin(selected_motivation)]
    if selected_school:
        filtered_df = filtered_df[filtered_df["School_Type"].isin(selected_school)]

    return filtered_df.to_dict("records")

# Démarrer l'application
if __name__ == "__main__":
    app.run_server(debug=True)
