# -*- coding: utf-8 -*-

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from datetime import date
from os import path as os_path

import pandas as pd
import plotly.express as px

from flask_caching import Cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation


PREFIX = '/'

app = dash.Dash(
    __name__,
    url_base_pathname=PREFIX,
    assets_url_path=PREFIX + 'assets/',
    assets_folder= os_path.join(os_path.dirname(os_path.abspath(__file__)), 'assets/')
)

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60


########## Body of app ##########
app.layout = html.Div(
    className="body_app",
    children=[
        html.Div(id='export_df', style={'display': 'none'}),
        html.Div(id='coord_word', style={'display': 'none'}),
        html.Div(id='taux_tweet_sentiment', style={'display': 'none'}),
        ##### Menu du body
        html.Div(
            className="left_menu",
            children=[
                html.H1("Menu", className="titre_menu"),
                html.Ul(
                    className="menu_lien",
                    children=[
                        html.Li([
                            html.A(className="menu_lien_text", href="#analyse_global", children="Analyse globale")
                        ],className="menu_lien_boite"),
                        html.Li([
                            html.A(className="menu_lien_text", href="#sentiment_analysis", children="Sentiment Analysis")
                        ],className="menu_lien_boite"),
                        html.Li([
                            html.A(className="menu_lien_text", href="#topic_modeling", children="Topic Modeling")
                        ],className="menu_lien_boite")
                    ]
                ),
                html.H1("Filtres", className="titre_menu"),
                html.H2("Candidats", className="selecteur_menu"),
                dcc.Dropdown(id='select_city',
                                 options=[
                                     {'label': "ND-Aignan", 'value': "aignan"},
                                     {'label': "N-Arthaud", 'value': "arthaud"},
                                     {'label': "F-Asselineau", 'value': "asselineau"},
                                     {'label': "X-Bertrand", 'value': "bertrand"},
                                     {'label': "A-Hidalgo", 'value': "hidalgo"},
                                     {'label': "Y-Jadot", 'value': "jadot"},
                                     {'label': "M-Lepen", 'value': "lepen"},
                                     {'label': "E-Macron", 'value': "macron"},
                                     {'label': "JL-Melenchon", 'value': "melenchon"},
                                     {'label': "V-Pecresse", 'value': "pecresse"},
                                     {'label': "P-Poutou", 'value': "poutou"},
                                     {'label': "F-Roussel", 'value': "roussel"}],value='aignan'),
                html.H2("Plage de dates", className="selecteur_menu"),
                dcc.DatePickerRange(
                        id='date_selection',
                        min_date_allowed=date(2021, 1, 1),
                        max_date_allowed=date(2022, 1, 1),
                        start_date=date(2021, 1, 1),
                        end_date=date(2021, 2, 1),
                        display_format='YYYY/MM/DD'
                    )
            ]),
        ##### Partie droite du body
        html.Div(
            className="right_content",
            children=[
                html.Header([html.H1("Analyse Twitter", className="titre_application")], className="block_titre"),
                html.P("Cette application a pour but de proposer un outil pour visualiser et analyser les tweets évoquant les candidats à l'élection présidentielle de 2022 en France. Les Tweets sont récoltés grâce à un script python executé tout les jours. Vous pouvez retrouver l'ensemble du projet sur mon Github."),
                html.Br(),
                html.Section(
                    className="section_content",
                    children=[
                        html.H1('Analyse Globale', id="analyse_global", className="titre_menu"),
                        html.P("Dans cette première partie, nous vous présentons un état des lieux des tweets récoltés. Vous pouvez sélectionner un candidat ou un potentiel candidat ainsi qu'une plage de dates spécifique pour affiner votre recherche."),
                        ##### 3 infobox
                        html.Div(
                            className="group_infobox_analyse",
                            children=[
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://assets.stickpng.com/images/580b57fcd9996e24bc43c53e.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("NOMBRE DE TWEETS", className="text_infobox"),
                                            html.P(id="nb_tweet", children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ]),
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://cdn3.iconfinder.com/data/icons/glypho-free/64/performance-clock-speed-512.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("NB TWEETS PAR JOUR", className="text_infobox"),
                                            html.P(id="nb_tweet_moyen", children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ]),
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://www.icone-png.com/png/45/44907.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("DATE RECORD", className="text_infobox"),
                                            html.P(id="max_date", children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ])
                        ]),
                        # graphique evolution jour
                        html.Div(
                                    className="analyse_graph",
                                    children=[
                                        dcc.Graph(id='graph-with-slider')
                                    ])
                    ]),
                html.Section(
                    className="section_content",
                    children=[
                        html.H1('Sentiment Analysis', id="sentiment_analysis", className="titre_menu"),
                        html.P("Dans cette deuxième partie, nous étudions le sentiment des tweets. Le sentiment d'un tweet est déterminé en fonction d'un algorithme de machine learning pré-entraîné. Il permet de determiner la popularité d'un candidat sur Twitter."),
                        ##### 3 infobox
                        html.Div(
                            className="group_infobox_analyse",
                            children=[
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://image.flaticon.com/icons/png/512/20/20664.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("NOTE DE POPULARITE", className="text_infobox"),
                                            html.P(id = 'note_satisfaction', children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ]),
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://static.thenounproject.com/png/6830-200.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("TAUX DE TWEETS POSITIFS", className="text_infobox"),
                                            html.P(id='tx_tweet_positif', children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ]),
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://image.flaticon.com/icons/png/128/334/334047.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("TAUX DE TWEETS NEGATIFS", className="text_infobox"),
                                            html.P(id='tx_tweet_negatif', children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ])
                        ]),
                        html.Div(
                                    className="analyse_graph",
                                    children=[
                                        dcc.Graph(id='graph_sentiment')
                                    ])
                    ]),
                html.Section(
                    className="section_content",
                    children=[
                        html.H1('Topic Modeling', id="topic_modeling", className="titre_menu"),
                        html.P("Dans cette troisieme partie, nous vous proposons un outil pour identifier les differentes tematiques abordées dans les tweets recoltés."),
                        html.P("Vous pourrez ensuite analyser les différentes thématiques et voir leur degré de popularité sur Twitter. Vous pouvez sélectionner un nombre de thématiques à identifier, ainsi qu'un algorithme pour calculer les rapprochements. Vous avez le choix entre la Factorisation par matrice non négative (NMF) ou l'allocation de Dirichlet latente (LDA). L'algorithme de LDA est beaucoup plus long que la NMF. Pour cette rasion je vous conseille d'utiliser la NMF."),
                        html.Div(
                            className="group_topic_modeling",
                            children=[
                                html.Div(
                                    className="selection_topic",
                                    children=[
                                        html.P("Number of topic"),
                                        dcc.Slider(id="nb_topic",min=2, max=20, marks={i: str(i) for i in range(1, 20)}, value=5 ),
                                        html.P("Select a method for identify Topics"),
                                        dcc.Dropdown(id='select_methode_modeling',options=[{'label': 'NMF', 'value': 'nmf'}, {'label': 'LDA', 'value': 'lda'}], value="nmf"),
                                        dcc.Graph(id='topic_pie')
                                ]),
                                # Contenue graphique
                                html.Div(
                                    className="graph_nb_topic",
                                    children=[
                                        dcc.Graph(id='topic_repart')
                                    ])
                                #dcc.Graph(id='topic_repart2')
                            ]),
                        html.P("Vous pouvez selectionner un Topic et l'analyser."),
                        dcc.Dropdown(id='selecteur_topic', value="Topic 1"),
                        html.Div(
                            className="group_infobox_analyse",
                            children=[
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://assets.stickpng.com/images/580b57fcd9996e24bc43c53e.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("NOTE DE SATISFACTION", className="text_infobox"),
                                            html.P(id='note_topic',children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ]),

                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://cdn3.iconfinder.com/data/icons/glypho-free/64/performance-clock-speed-512.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("TAUX DE TWEETS POSITIFS", className="text_infobox"),
                                            html.P(id="tx_tweet_topic_positif",children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ]),
                                #Infobox sur le nombre de tweets
                                html.Div(
                                    className="analyse_infobox",
                                    children=[
                                        html.Div([html.Img(className="image_infobox", src="https://www.icone-png.com/png/45/44907.png")],className="infobox_image_div"),
                                        html.Div([
                                            html.P("TAUX DE TWEETS NEGATIFS", className="text_infobox"),
                                            html.P(id="tx_tweet_topic_negatif",children=["..."], className="text_infobox")
                                        ], className="infobox_text_div")
                                    ])
                        ]),
                        html.Div(
                            className="analyse_graph",
                            children=[
                                dcc.Graph(id='top_word_topic')
                            ]
                        )
                    ]),

            ]
        )
    ])

########## Elements interactifs ##########
@cache.memoize(timeout=TIMEOUT)
def import_data(city):
    df = pd.read_csv("data/tweet_" + str(city) + ".csv")
    df.drop_duplicates(inplace=True)
    df["created_at"] = pd.to_datetime(df.created_at)
    return(df)


##### Mise a jour des elements #####
@app.callback(
    [Output('graph-with-slider', 'figure'),
    Output('nb_tweet', 'children'),
    Output('nb_tweet_moyen', 'children'),
    Output('max_date', 'children'),
    Output('graph_sentiment', 'figure'),
    Output('note_satisfaction', 'children'),
    Output('tx_tweet_positif','children'),
    Output('tx_tweet_negatif','children'),
    Output('export_df', 'children')],
    [Input('select_city', 'value'),
     Input('date_selection', 'start_date'),
     Input('date_selection', 'end_date')]
)
def apply_date_filter(select_city,start_date,end_date):
    #Importation du jeu de données
    df = import_data(select_city)
    df = df[df.tweet_process.isna()==False]
    #Filtre sur les dates
    df = df[(df.created_at >= start_date) & (df.created_at <= end_date)]

    #Nombre de Tweets
    nb_tweets = str(df.shape[0])

    #Nombre de Tweets moyen par jours
    nb_tweets_moyen = round(df.shape[0] / (df.created_at.max() - df.created_at.min()).days)

    #Note de satisfaction
    note_satisfaction = str(round((df.note_sentiment.mean() + 100)/2)) + "/100"
    note_satisfaction = str(round((df[df.sentiment == "positive"].shape[0]*100 + df[df.sentiment == "neutral"].shape[0]*50)/df.shape[0])) + "/100"


    #Taux de tweets positif
    tx_tweet_positif = str(round(df[df.sentiment == "positive"].shape[0]/df.shape[0]*100)) + "%"
    tx_tweet_negatif = str(round(df[df.sentiment == "neutral"].shape[0]/df.shape[0]*100)) + "%"

    test = pd.to_datetime(df.created_at)
    test = test.groupby(test.dt.floor('1H')).count()
    df_graph = pd.DataFrame({"date": test.index, "nb": test.values}).sort_values(by="date")
    fig = px.line(df_graph, x="date", y="nb", title='Répartition des tweets dans le temps')
    fig.update_traces(line_color='#f56a6a')
    fig.update_layout(
        xaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        yaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        autosize=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_x=0.5,
        xaxis_showgrid=False, yaxis_showgrid=False
    )

    import datetime as dt
    df['date'] = pd.to_datetime(df.created_at).dt.floor('24H')
    grph_sentiment = df[["date", "note_sentiment"]].groupby(["date"]).count().reset_index()
    max_date = str(grph_sentiment[grph_sentiment.note_sentiment == grph_sentiment.note_sentiment.max()].date.dt.date.values[0])

    grph_sentiment = df[["date", "sentiment", "note_sentiment"]].groupby(["date", "sentiment"]).count()
    grph_sentiment = grph_sentiment.reset_index().rename(columns={"note_sentiment": "nb"})
    fig2 = px.bar(grph_sentiment, x="date", y="nb", color="sentiment",
                  title="Répartition des tweets par jour et par sentiment",
                  color_discrete_sequence=["#f14e56", "#c0c0c0", "#5ca369"],)
    fig2.update_layout(
        xaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        yaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        autosize=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_x=0.5,
        xaxis_showgrid=False, yaxis_showgrid=False
    )
    export_df = df[["tweet_process","sentiment","note_sentiment","date"]].to_json(date_format='iso', orient='split')

    return ([fig, nb_tweets, nb_tweets_moyen, max_date, fig2, note_satisfaction, tx_tweet_positif, tx_tweet_negatif,export_df])

@app.callback(
    [Output('topic_pie', 'figure'),
     Output('topic_repart', 'figure'),
     Output('coord_word','children'),
     Output('taux_tweet_sentiment','children')],
    [Input('export_df', 'children'),
     Input('nb_topic', 'value'),
     Input('select_methode_modeling', 'value')
     ]
)

def modeling_topic(export_df,nb_topic,select_methode_modeling):
    df = pd.read_json(export_df, orient='split')
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.01 , ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(df.tweet_process)

    ##### NMF
    if select_methode_modeling == "nmf":
        nmf = NMF(n_components=nb_topic, random_state=123).fit(X_train)
    elif select_methode_modeling == "lda":
        nmf = LatentDirichletAllocation(n_components=nb_topic, random_state=123).fit(X_train)

    # nmf = NMF(n_components=nb_topic, random_state=123).fit(X_train)
    train_nmf = nmf.transform(X_train)

    ##### Top word Topic
    coord_word = pd.DataFrame(nmf.components_).T
    coord_word.index = vectorizer.get_feature_names()
    coord_word.columns = ["Topic " + str(i+1) for i in range(nmf.components_.shape[0])]

    ##### tweet topic
    df.reset_index(inplace=True)
    df["topic"] = pd.DataFrame(train_nmf).idxmax(axis=1) + 1

    ##### Répartition des topics
    graph_value = df.topic.value_counts().values
    graph_label = ["topic " + str(int(i)) for i in df.topic.value_counts().index]

    topic_pie = px.pie(values=graph_value, names=graph_label, title='Topic repartition',
                       height=274,
                       color_discrete_sequence=px.colors.sequential.RdBu)
    # color_discrete_map = {'Thur': 'lightcyan','Fri': 'cyan'})
    topic_pie.update_layout(
        xaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        yaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        autosize=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_x=0.5,
        xaxis_showgrid=False, yaxis_showgrid=False
    )

    ##### Répartition des topics par jours
    df['topic'] = "Topic " + df.topic.astype(str)
    grph_sentiment = df[["date", "topic", "note_sentiment"]].groupby(["date", "topic"]).count()
    grph_sentiment = grph_sentiment.reset_index().rename(columns={"note_sentiment": "nb"})
    topic_repart = px.bar(grph_sentiment, x="date", y="nb", color="topic",
                          title="Répartition des tweets par jour et par topic",
                          color_discrete_sequence=px.colors.sequential.RdBu)
    # color_discrete_sequence=["#f14e56", "#c0c0c0", "#5ca369"],)
    topic_repart.update_layout(
        xaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        yaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        autosize=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_x=0.5,
        xaxis_showgrid=False, yaxis_showgrid=False
    )
    taux_tweet_sentiment = df[['topic', 'sentiment', 'note_sentiment']].groupby(['topic', 'sentiment']).count().reset_index()
    taux_tweet_sentiment = taux_tweet_sentiment.to_json(date_format='iso', orient='split')

    coord_word = coord_word.to_json(date_format='iso', orient='split')
    return([topic_pie,topic_repart, coord_word, taux_tweet_sentiment])


@app.callback(
    Output('selecteur_topic', 'options'),
    [Input('coord_word', 'children')]
)
def list_topic(coord_word):
    coord_word = pd.read_json(coord_word, orient='split')
    return([{'label': k, 'value': k} for k in coord_word.columns])

@app.callback(
    [Output('top_word_topic', 'figure'),
     Output('note_topic', 'children'),
     Output('tx_tweet_topic_positif', 'children'),
     Output('tx_tweet_topic_negatif', 'children')],
    [Input('coord_word', 'children'),
    Input('taux_tweet_sentiment','children'),
    Input('selecteur_topic', 'value')]
)
def analyse_topic(coord_word,taux_tweet_sentiment,selecteur_topic):
    coord_word = pd.read_json(coord_word, orient='split')
    taux_tweet_sentiment = pd.read_json(taux_tweet_sentiment, orient='split')

    nb_total = taux_tweet_sentiment[taux_tweet_sentiment.topic == selecteur_topic].note_sentiment.sum()
    nb_positif = taux_tweet_sentiment[
        (taux_tweet_sentiment.topic == selecteur_topic) & (taux_tweet_sentiment.sentiment == "positive")].note_sentiment.sum()
    nb_negatif = taux_tweet_sentiment[
        (taux_tweet_sentiment.topic == selecteur_topic) & (taux_tweet_sentiment.sentiment == "negative")].note_sentiment.sum()
    tx_tweet_topic_positif = str(round((nb_positif / nb_total) * 100)) + "%"
    tx_tweet_topic_negatif = str(round((nb_negatif / nb_total) * 100)) + "%"
    note_topic = str(round((nb_positif * 100 + (nb_total - nb_positif - nb_negatif) * 50) / nb_total)) + "/100"



    top_word = coord_word.sort_values(selecteur_topic, ascending=False)[selecteur_topic][:15]
    top_word_topic = px.bar(x=top_word.values, y=top_word.index, title="Mots les plus discriminants")
    top_word_topic.update_layout(
        xaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        yaxis=dict(showline=False, showticklabels=True,
                   tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        autosize=False, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title_x=0.5,
        xaxis_showgrid=False, yaxis_showgrid=False
    )
    return([top_word_topic,note_topic,tx_tweet_topic_positif,tx_tweet_topic_negatif])


if __name__ == '__main__':
    app.run_server(debug=True)