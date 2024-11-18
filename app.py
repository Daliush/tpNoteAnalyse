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
heatmap_fig.update_layout(
    title="Correlation Heatmap of Numeric Factors",
    font=dict(size=12)
)

# Initialiser l'application Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Page d'accueil embellie
homepage_content = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Bienvenue sur le Tableau de Bord des Performances Étudiantes",
                            className="text-center my-4",
                            style={"color": "black"}),  # Titre principal en noir
    )]),
        dbc.Row([
            dbc.Col(html.Img(src="https://cdn.pixabay.com/photo/2019/08/06/22/48/artificial-intelligence-4389372_1280.jpg",
                             alt="Image représentant les performances étudiantes",
                             className="img-fluid mb-4 rounded shadow"), width=12)
        ]),
        dbc.Row([
            dbc.Col(html.H3("Explorez et Analysez les Données Étudiantes",
                            className="text-center mb-3 text-secondary"), width=12),
            dbc.Col(html.P("""
                Ce tableau de bord interactif permet d'explorer les performances des élèves lors d'examens
                en fonction de divers facteurs comme l'implication des parents, le type d'école, et les heures d'étude.
                Identifiez les corrélations clés et comprenez les facteurs influençant les scores des étudiants.
            """, className="text-center"), width=12)
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H3("Fonctionnalités", className="text-center mt-4 mb-4 text-success"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Visualisez les Scores", className="card-title text-primary"),
                    html.P("""
                        Analysez la distribution des scores des examens avec des graphiques interactifs.
                    """, className="card-text"),
                ])
            ], className="mb-4 shadow"), width=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Relations entre Facteurs", className="card-title text-warning"),
                    html.P("""
                        Explorez les corrélations entre différents facteurs grâce à une heatmap.
                    """, className="card-text"),
                ])
            ], className="mb-4 shadow"), width=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("Clustering et Analyse Avancée", className="card-title text-success"),
                    html.P("""
                        Découvrez les regroupements d'étudiants grâce à des analyses de clustering (PCA et t-SNE).
                    """, className="card-text"),
                ])
            ], className="mb-4 shadow"), width=4),
        ]),
    ])
])

# Tableau de bord
dashboard_content = dbc.Container([
    dbc.Row([dbc.Col(html.H1("Student Performance Dashboard", className="text-center"))]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="distribution-plot", figure=fig_distribution), width=6),
        dbc.Col(dcc.Graph(id="correlation-heatmap", figure=heatmap_fig), width=6),
    ]),
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

# Navigation
app.layout = dbc.Container([
    dcc.Location(id="url"),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Accueil", href="/")),
            dbc.NavItem(dbc.NavLink("Tableau de Bord", href="/dashboard")),
        ],
        brand="Performances Étudiantes",
        brand_style={"color": "black"},  # Bandeau de menu : Titre en noir
        color="white",  # Fond blanc pour le bandeau
        dark=False,  # Définir "dark" sur False pour fond clair
        className="mb-4 border-bottom"
    ),
    html.Div(id="page-content")
], fluid=True)

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/dashboard":
        return dashboard_content
    return homepage_content

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

if __name__ == "__main__":
    app.run_server(debug=True)
