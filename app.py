import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash_table import DataTable
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

file_path = 'StudentPerformanceFactors.csv'
student_performance_df = pd.read_csv(file_path)
# Séparer les colonnes numériques et catégorielles
numeric_features = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
    'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score'
]
categorical_features = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

# Construire le transformateur pour normaliser les numériques et encoder les catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Appliquer le fit_transform
features_scaled = preprocessor.fit_transform(student_performance_df)


# PCA pour la réduction de dimension
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
student_performance_df['PCA1'] = pca_result[:, 0]
student_performance_df['PCA2'] = pca_result[:, 1]

# t-SNE pour la réduction de dimension
tsne = TSNE(n_components=2, perplexity=2, n_iter=300, random_state=0)
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
                            style={"color": "black"}))
        ]),
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
        html.Hr(),
    ])
])

# Dashboard content
dashboard_content = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1(
            "Student Performance Dashboard",
            className="text-center",
            style={"color": "#2C3E50", "fontWeight": "700", "padding": "20px"}
        ))
    ]),
        dbc.Row([
        dbc.Col(html.Div(
            """
            **Analyse des Dimensions Réduites :**
            - La projection **PCA** (Principal Component Analysis) est utilisée pour réduire les dimensions des données tout en conservant un maximum de variance. Les axes PCA1 et PCA2 représentent les directions principales où les données varient le plus.
            - La projection **t-SNE** (t-Distributed Stochastic Neighbor Embedding) est une technique non linéaire qui conserve les relations locales entre les points, utile pour identifier les regroupements naturels dans les données.
            """,
            style={
                "fontFamily": "'Open Sans', sans-serif",
                "color": "#34495E",
                "backgroundColor": "#F9F9F9",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0px 2px 5px rgba(0, 0, 0, 0.1)",
                "fontStyle": "italic",
                "lineHeight": "1.8"
            }
        ))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="distribution-plot", figure=fig_distribution), width=6),
        dbc.Col(dcc.Graph(id="correlation-heatmap", figure=heatmap_fig), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id="clustering-method",
            options=[
                {"label": "K-Means", "value": "KMeans"},
                {"label": "Agglomerative", "value": "Agglomerative"},
                {"label": "Spectral Clustering", "value": "Spectral"},
                {"label": "Mean-Shift", "value": "MeanShift"},
                {"label": "Gaussian Mixture", "value": "GaussianMixture"}
            ],
            value="KMeans",
            placeholder="Choisir une méthode de clustering"
        ), width=4),
        dbc.Col(dcc.Slider(
            id="cluster-slider", min=2, max=10, step=1, value=3,
            marks={i: str(i) for i in range(2, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ), width=8)
    ]),
    dbc.Row([
        dbc.Col(dcc.RadioItems(
            id="dimensionality-method",
            options=[{'label': 'PCA', 'value': 'PCA'}, {'label': 't-SNE', 'value': 't-SNE'}],
            value='PCA',
            labelStyle={'display': 'inline-block'}
        ), width=12)
    ]),
        dbc.Row([
        dbc.Col(html.Div(
            """
            Plus un élève est a gauche, plus il a eu une bonne note à l'exam et a assisté à beaucoup de cours.
            Un elève en haut du graphique est un élève qui a eu une bonne note au dernier exam et qui dors beaucoup.
            """,
            style={
                "fontFamily": "'Open Sans', sans-serif",
                "color": "#34495E",
                "backgroundColor": "#F9F9F9",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0px 2px 5px rgba(0, 0, 0, 0.1)",
                "fontStyle": "italic",
                "lineHeight": "1.8"
            }
        ))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="dimensionality-plot"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="silhouette-score-text", style={
            "fontSize": "18px",
            "fontWeight": "bold",
            "marginBottom": "20px",
            "color": "black",
            "textAlign": "center"
        }), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H3("Student Data Table", className="text-center mt-4"), width=12),
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
            dbc.NavItem(
                dbc.Button(
                    "Tableau de Bord",
                    href="/dashboard",
                    color="primary",
                    className="btn-sm shadow",
                    style={"fontWeight": "bold", "borderRadius": "5px"}
            )

            ),
        ],
        brand="Performances Étudiantes",
        brand_style={"color": "black"},
        color="white",
        dark=False,
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
    [Output("dimensionality-plot", "figure"),
     Output("silhouette-score-text", "children")],
    [Input("dimensionality-method", "value"),
     Input("clustering-method", "value"),
     Input("cluster-slider", "value")]
)
def update_dimensionality_and_silhouette(method, clustering_method, n_clusters):

    if(method == "PCA") :
        result = pca_result
    else : 
        result = tsne_result
    if clustering_method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = model.fit_predict(result)
    elif clustering_method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = model.fit_predict(result)
    elif clustering_method == "Spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='nearest_neighbors')
        clusters = model.fit_predict(result)
    elif clustering_method == "MeanShift":
        model = MeanShift()
        clusters = model.fit_predict(result)
    elif clustering_method == "GaussianMixture":
        model = GaussianMixture(n_components=n_clusters, random_state=0)
        clusters = model.fit_predict(result)
    else:
        raise ValueError("Méthode de clustering non supportée.")

    student_performance_df['Cluster'] = clusters

    if n_clusters > 1:
        score_silhouette = silhouette_score(result, student_performance_df['Cluster'])
        if score_silhouette > 0.7:
            quality = "Excellente qualité"
            color = "green"
        elif score_silhouette > 0.5:
            quality = "Bonne qualité"
            color = "blue"
        elif score_silhouette > 0.25:
            quality = "Qualité moyenne"
            color = "orange"
        else:
            quality = "Mauvaise qualité"
            color = "red"
        silhouette_text = f"Score silhouette: {score_silhouette:.2f} ({quality})"
    else:
        silhouette_text = "Score silhouette non calculable pour un seul cluster."

    if method == 'PCA':
        fig = px.scatter(student_performance_df, x="PCA1", y="PCA2", color="Cluster",
                          hover_data={"PCA1": False,
                                       "PCA2": False,
                                         "Exam_Score": True,
                                         "Hours_Studied": True,
                                         "Distance_from_Home" : True,
                                         "Motivation_Level" : True,
                                         "Previous_Scores" : True,
                                         "Sleep_Hours" : True,
                                         },
                         title=f"Clustering avec {clustering_method} et PCA")
    else:
        fig = px.scatter(student_performance_df, x="TSNE1", y="TSNE2", color="Cluster",
                          hover_data={"TSNE1": False,
                                       "TSNE2": False,
                                         "Exam_Score": True,
                                         "Hours_Studied": True,
                                         "Distance_from_Home" : True,
                                         "Motivation_Level" : True,
                                         "Previous_Scores" : True,
                                         "Sleep_Hours" : True,
                                         },
                         title=f"Clustering avec {clustering_method} et t-SNE")

    return fig, silhouette_text

if __name__ == "__main__":
    app.run_server(debug=True)
