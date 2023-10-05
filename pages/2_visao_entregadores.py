# Libraries
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
from haversine import haversine
from datetime import datetime

# Bibliotecas necessarias
import pandas as pd
import streamlit as st
import folium
from PIL import Image

from streamlit_folium import folium_static

st.set_page_config( page_title='Visão Entregadores', page_icon='🏍', layout='wide')

# ----------------------------------------
# Funções 
# ----------------------------------------
def clean_code( df1 ):
    """ Esta funcao tem a responsabilidade de limpar o dataframe

        Tipos de limpeza
        1. Remocao dos dados NaN
        2. Mudanca do tipo da coluna de dados
        3. Remocao dos espacos das variaveis de textos
        4. Formatacao da coluna de dados
        5. Limpeza da coluna de tempo ( remocao do texto da variavel numerica )

        Input: Dataframe
        Output: Dataframe
    """
    # 1. Convertendo a coluna Age de texto para numero
    linhas_selecionadas = ( df1['Delivery_person_Age'] != 'NaN ' )
    df1 = df1.loc[linhas_selecionadas, :].copy()

    linhas_selecionadas = ( df1['Road_traffic_density'] != 'NaN ' )
    df1 = df1.loc[linhas_selecionadas, :].copy()

    linhas_selecionadas = ( df1['City'] != 'NaN ' )
    df1 = df1.loc[linhas_selecionadas, :].copy()

    linhas_selecionadas = ( df1['Festival'] != 'NaN ' )
    df1 = df1.loc[linhas_selecionadas, :].copy()

    df1['Delivery_person_Age'] = df1['Delivery_person_Age'].astype( int )

    # 2. Convertendo a coluna Ratings de texto para numero decimal ( float )
    df1['Delivery_person_Ratings'] = df1['Delivery_person_Ratings'].astype( float )

    # 3. Convertendo a coluna order_date de texto para data
    df1['Order_Date'] = pd.to_datetime( df1['Order_Date'], format= '%d-%m-%Y' )

    # 4. Convertendo multiple_deliveries de texto para numero inteiro ( int )
    linhas_selecionadas = ( df1['multiple_deliveries'] != 'NaN ' )
    df1 = df1.loc[linhas_selecionadas, :].copy()
    df1['multiple_deliveries'] = df1['multiple_deliveries'].astype( int )

    # 6. Removendo os espaços de strings
    df1.loc[:, 'ID'] = df1.loc[:, 'ID'].str.strip()
    df1.loc[:, 'Delivery_person_ID'] = df1.loc[:, 'Delivery_person_ID'].str.strip()
    df1.loc[:, 'Road_traffic_density'] = df1.loc[:, 'Road_traffic_density'].str.strip()
    df1.loc[:, 'Type_of_order'] = df1.loc[:, 'Type_of_order'].str.strip()
    df1.loc[:, 'Type_of_vehicle'] = df1.loc[:, 'Type_of_vehicle'].str.strip()
    df1.loc[:, 'City'] = df1.loc[:, 'City'].str.strip()
    df1.loc[:, 'Festival'] = df1.loc[:, 'Festival'].str.strip()

    # 7. Limpando a coluna de time taken
    df1['Time_taken(min)'] = df1['Time_taken(min)'].apply( lambda x: x.split( '(min) ')[1] )
    df1['Time_taken(min)'] = df1['Time_taken(min)'].astype( int )

    return df1

def top_delivers( df1, top_asc ):
    df2 = ( df1.loc[:, ['Delivery_person_ID', 'City', 'Time_taken(min)']]
                .groupby( ['City', 'Delivery_person_ID'] )
                .mean()
                .sort_values( ['City', 'Time_taken(min)'], ascending=top_asc )
                .reset_index() )

    df_aux01 = df2.loc[df2['City'] == 'Metropolitian', :].head(10)
    df_aux02 = df2.loc[df2['City'] == 'Urban', :].head(10)
    df_aux03 = df2.loc[df2['City'] == 'Semi-Urban', :].head(10)
            
    df3 = pd.concat( [df_aux01, df_aux02, df_aux03] ).reset_index( drop=True )

    return df3 

# Import dataset
df = pd.read_csv( 'dataset/train.csv' )

# cleaning dataset
df1 = clean_code( df )

# ================================================================================
# Barra Lateral
# ================================================================================
st.header('Marketplace - Visão Entregadores')

#image_path = 'logo.png'
image = Image.open( 'logo.png' )
st.sidebar.image( image, width=180 )

st.sidebar.markdown( '# Curry Company')
st.sidebar.markdown( '## Fastest Delivery in Town')
st.sidebar.markdown( """___""")

st.sidebar.markdown( '## Selecione uma data limite')

date_slider = st.sidebar.slider( 
    'Até qual valor ?',
    value=datetime( 2022, 3, 12 ),
    min_value=datetime(2022, 2, 11 ),
    max_value=datetime( 2022, 4, 6 ),
    format='DD-MM-YYY' )

st.sidebar.markdown( """___""")

traffic_options = st.sidebar.multiselect(
    'Quais as condições do trânsito',
    ['Low', 'Medium', 'High', 'Jam'],
    default=['Low', 'Medium', 'High', 'Jam'] )

st.sidebar.markdown( """___""")
st.sidebar.markdown( '###### Powered by Edinan Marinho with Comunidade DS')

# Filtro de Data
linhas_selecionadas = df1['Order_Date'] < date_slider
df1 = df1.loc[linhas_selecionadas,:]

# Filtro de Transito
linhas_selecionadas = df1['Road_traffic_density'].isin( traffic_options )
df1 = df1.loc[linhas_selecionadas, :]

# ================================================================================
# Layout no Streamlit
# ================================================================================
tab1, = st.tabs( ['Visão Gerencial'] )

with tab1:
    with st.container():
        st.markdown( '### Overall Metrics' )

        col1, col2, col3, col4 = st.columns( 4, gap='large' )
        with col1:



            # A maior idade dos entregadores
            maior_idade = df1.loc[:, 'Delivery_person_Age'].max()
            col1.metric( 'Maior de Idade', maior_idade )

        with col2:
            # A menor idade dos entregadores
            menor_idade = df1.loc[:, 'Delivery_person_Age'].min()
            col2.metric( 'Menor Idade', menor_idade )

        with col3:
            # A melhor condicao de veiculo
            melhor_condicao = df1.loc[:,'Vehicle_condition'].max()
            col3.metric( 'Melhor Condição', melhor_condicao )

        with col4:
            # A pior condicao de veiculo
            pior_condicao = df1.loc[:,'Vehicle_condition'].min()
            col4.metric( 'Pior Condição', pior_condicao )

    with st.container():
        st.markdown( """---""" )
        st.markdown( '#### Avaliações' )

        col1, col2 = st.columns( 2 )
        with col1:
            st.markdown( '##### Avaliação média por Entregador' )
            df_avg_ratings_per_deliver = ( df1.loc[:, ['Delivery_person_ID', 'Delivery_person_Ratings']]
                                              .groupby( 'Delivery_person_ID' )
                                              .mean()
                                              .reset_index() )
            st.dataframe( df_avg_ratings_per_deliver )

        with col2:
            st.markdown( '##### Avaliação média por Trânsito' )
            df_agg_rating_by_traffic = ( df1.loc[:, ['Delivery_person_Ratings', 'Road_traffic_density']]
                                            .groupby( 'Road_traffic_density' )
                                            .agg({ 'Delivery_person_Ratings': ['mean', 'std'] } ) )

            # mudanca de nome das colunas
            df_agg_rating_by_traffic.columns = ['delivery_mean', 'delivery_std']

            # reset_index
            df_agg_rating_by_traffic = df_agg_rating_by_traffic.reset_index()
            st.dataframe( df_agg_rating_by_traffic )

            st.markdown( '##### Avaliação média por Clima' )
            df_agg_rating_by_weatherconditions = ( df1.loc[:, ['Delivery_person_Ratings', 'Weatherconditions']]
                                                      .groupby( 'Weatherconditions' )
                                                      .agg({ 'Delivery_person_Ratings': ['mean', 'std'] } ) )

            # mudanca de nome das colunas
            df_agg_rating_by_weatherconditions.columns = ['Weatherconditions_mean', 'Weatherconditions_std']

            # reset_index
            df_agg_rating_by_weatherconditions = df_agg_rating_by_weatherconditions.reset_index()
            st.dataframe( df_agg_rating_by_weatherconditions )

    with st.container():
        st.markdown( """---""" )
        st.markdown( '#### Velocidade de Entrega' )

        col1, col2 = st.columns( 2 )

        with col1:
            st.markdown( '##### Top Entregadores mais rápidos' )
            df3 = top_delivers( df1, top_asc=True )
            st.dataframe( df3 )

        with col2:
            st.markdown( '##### Top Entregadores mais lentos' )
            df3 = top_delivers( df1, top_asc=False )
            st.dataframe( df3 )


           
            
