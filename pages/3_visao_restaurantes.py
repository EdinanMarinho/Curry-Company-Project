# Libraries
from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Bibliotecas necessarias
import folium
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from streamlit_folium import folium_static

st.set_page_config( page_title='Visão Restaurantes', page_icon='🍽', layout='wide')

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

def distance( df1, fig ):
        if fig == False:
            cols = ['Delivery_location_latitude', 'Delivery_location_longitude', 
                    'Restaurant_latitude', 'Restaurant_longitude']
            df1['distance'] = df1.loc[:,cols].apply( lambda x: haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']), 
                                                (x['Delivery_location_latitude'], x['Delivery_location_longitude']) ), axis=1 )

            avg_distance = np.round( df1['distance'].mean(), 2 )
            return avg_distance

        else:
            cols = ['Delivery_location_latitude', 'Delivery_location_longitude', 
                    'Restaurant_latitude', 'Restaurant_longitude']
            df1['distance'] = df1.loc[:,cols].apply( lambda x: haversine( (x['Restaurant_latitude'], x['Restaurant_longitude']), 
                                                (x['Delivery_location_latitude'], x['Delivery_location_longitude']) ), axis=1 )

            avg_distance = df1.loc[:, ['City', 'distance']].groupby( 'City' ).mean().reset_index()
            fig = go.Figure( data=[ go.Pie( labels = avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0])])

            return fig


def avg_std_time_delivery( df1, festival, op ):
    """
        Esta função calcula o tempo médio e o desvio padrão do tempo de entrega.
            Parâmetros:
            Input:
                - df: Dataframe com os dados necessários para o cálculo.
                - op: Tipo de operação que precisa ser calculado.
                        'avg_time': Cálcula o tempo médio.
                        'std_time': Cálcula o desvio padrão do tempo.
            Output:
                - df: Dataframe com 2 colunas e 1 linha
    """

    df_aux = ( df1.loc[:, ['Time_taken(min)', 'Festival']]
                  .groupby( ['Festival'] )
                  .agg( {'Time_taken(min)':['mean', 'std']} ) )

    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()
    df_aux = np.round( df_aux.loc[df_aux['Festival'] == festival, op],2 )

    return df_aux

def avg_std_time_graph( df1 ):
    df_aux = df1.loc[:, ['City', 'Time_taken(min)']].groupby( 'City' ).agg( {'Time_taken(min)':['mean', 'std']} )
    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()

    fig =  go.Figure()
    fig.add_trace( go.Bar( name='Comtrol', x=df_aux['City'], y=df_aux['avg_time'], error_y=dict( type= 'data', array=df_aux['std_time'] ) ) )
    fig.update_layout( barmode='group' )

    return fig

def avg_std_time_on_traffic( df1 ):
    df_aux = ( df1.loc[:, ['City', 'Time_taken(min)', 'Road_traffic_density']]
                  .groupby( ['City', 'Road_traffic_density'] )
                  .agg( {'Time_taken(min)':['mean', 'std']} ) )

    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()

    fig = px.sunburst( df_aux, path=['City','Road_traffic_density'], values='avg_time',
                        color='std_time', color_continuous_scale='tempo',
                        color_continuous_midpoint=np.average(df_aux['std_time']))

    return fig

# Import dataset
df = pd.read_csv( 'dataset/train.csv' )

# Cleaning code
df1 = clean_code( df )

# ================================================================================
# Barra Lateral
# ================================================================================
st.header('Marketplace - Visão Restaurantes')

#image_path = 'logo.png'
image = Image.open( 'logo.png' )
st.sidebar.image( image, width=180 )

st.sidebar.markdown( '# Curry Company')
st.sidebar.markdown( '## Fastest Delivery in Town')
st.sidebar.markdown( """___""")

st.sidebar.markdown( '## Selecione uma data limite')

date_slider = st.sidebar.slider( 
    'Até qual valor ?',
    value=datetime( 2022, 4, 13 ),
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
        st.title( 'Overall Metrics' )
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            delivery_unique = ( len( df1.loc[:, 'Delivery_person_ID'].unique() ) )
            col1.metric( 'Entregadores', delivery_unique)

        with col2:
            avg_distance = distance( df1, fig=False )
            col2.metric(  'A distância média das entregas', avg_distance)

        with col3:
            df_aux = avg_std_time_delivery( df1, 'Yes', 'avg_time' )
            col3.metric( 'Tempo Médio de Entrega c/Festival', df_aux )

        with col4:
            df_aux = avg_std_time_delivery( df1, 'No', 'std_time' )
            col4.metric( 'STD Entrega c/Festival', df_aux )
            
        with col5:
            df_aux = avg_std_time_delivery( df1, 'Yes', 'std_time' )
            col5.metric( 'Tempo Médio de Entrega c/Festival', df_aux )
            
        with col6:
            df_aux = avg_std_time_delivery( df1, 'No', 'std_time' )
            col6.metric( 'Desvio Padrão Médio de Entrega c/Festival', df_aux )

    with st.container():
        st.markdown( """---""")
        col1, col2 =  st.columns( 2 )

        with col1:
            st.markdown( '#### Tempo médio de entrega por Cidade' )
            fig = avg_std_time_graph(df1 )
            st.plotly_chart( fig )

        with col2:
            st.markdown( '#### Distribuição da distância' )
            df_aux = ( df1.loc[:, ['City', 'Time_taken(min)', 'Type_of_order']]
                          .groupby( ['City', 'Type_of_order'] )
                          .agg( {'Time_taken(min)':['mean', 'std']} ))

            df_aux.columns = ['avg_time', 'std_time']
            df_aux = df_aux.reset_index()

            st.dataframe( df_aux )

    with st.container():
        st.markdown( """---""")
        st.markdown( '#### Distribuição do tempo' )

        col1, col2 = st.columns( 2 )
        with col1:
            fig = distance( df1, fig=True )
            st.plotly_chart( fig )
        
        with col2:
            fig = avg_std_time_on_traffic( df1 )
            st.plotly_chart( fig )
    
