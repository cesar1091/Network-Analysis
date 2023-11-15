import networkx as nx
from pyvis import network as net
import streamlit as st
import pandas as pd
import os
import numpy as np



def summary_graph(G):
    summary_graph = pd.DataFrame().from_dict(dict(G.in_degree()),orient='index')
    summary_graph.columns = ['in_degree']
    summary_graph['out_degree'] = dict(G.out_degree())
    summary_graph['in_degree_centrality'] = dict(nx.in_degree_centrality(G))
    summary_graph['out_degree_centrality'] = dict(nx.out_degree_centrality(G))
    centers = list(nx.center(G.to_undirected()))
    summary_graph['center_nodes'] = [int(x in centers) for x in summary_graph.index]
    periphery = list(nx.periphery(G.to_undirected()))
    summary_graph['periphery'] = [int(x in periphery) for x in summary_graph.index]
    summary_graph['betweeness_centrality'] = nx.betweenness_centrality(G,normalized=True)
    summary_graph['closeness_centrality'] = nx.closeness_centrality(G)
    #hubs, auth = nx.hits(G)
    #summary_graph['hubs_values'] = hubs
    #summary_graph['auth_values'] = auth
    summary_graph['avg_neighbor_degree'] = nx.average_neighbor_degree(G)
    return summary_graph

def color(row):
    if row['FLAG_EPIC007']==1:
        return "#FF0000"
    else:
        if row['TYPE'] == 'PN':
            return "#F2FA56"
        else:
            return "#4d82e3"

def shape(row):
    if row['FLAG_EPIC007']==1:
        return "star"
    else:
        if row['TYPE'] == 'PN':
            return "triangle"
        else:
            return "box"

def size(row):
    if row['FLAG_EPIC007']==1:
        return 55
    else:
        if row['TYPE'] == 'PN':
            return 30
        else:
            return 30

def title(row):
    return "Type:{}\nMonto Entrada: {}\nMonto Salida: {}\nDepartamento: {}\nAN: {}\nLSB NP: {}\nPEP: {}\nROS: {}".format(row['TYPE'],row['MONTO_ENTRADA'],row['MONTO_SALIDA'],row['DEPARTAMENTO'],row['FLAG_AN'],row['FLAG_LSB_NP'],row['FLAG_PEP'],row['FLAG_ROS'])


st.set_page_config(layout='wide')
st.title('NETWORK ANÁLISIS')

user_df = pd.read_csv('data/users.csv')
trx_df = pd.read_csv('data/trx.csv')
trx_df = trx_df[trx_df['ORIGEN']!=trx_df['DESTINO']]
trx_df_agg = trx_df.groupby(['ORIGEN','DESTINO']).agg({'MONTO':'sum',
                                                       'CTD_TRX':'sum',
                                                       'CANAL':lambda x: ','.join(x),
                                                       'TIPO':lambda x: ','.join(x)}).reset_index()
max_val = np.max(trx_df_agg['MONTO'].values)
min_val = np.min(trx_df_agg['MONTO'].values)
trx_df_agg['value'] = trx_df_agg['MONTO'].apply(lambda x: x)
trx_df_agg['title'] = trx_df_agg['MONTO'].apply(lambda x: "MONTO: ${:>,.2f}".format(x))

filter = user_df[user_df['FLAG_EPIC007']==1]['USERS'].unique()

diccionario = {'DEPARTAMENTO':list(user_df['DEPARTAMENTO'].unique()),
               'TYPE':list(user_df['TYPE'].unique()),
               'FLAG_AN':list(user_df['FLAG_AN'].unique())}


G = nx.from_pandas_edgelist(trx_df_agg,source='ORIGEN',target='DESTINO',edge_attr=True,create_using=nx.DiGraph())
node_attr = user_df.set_index('USERS').to_dict('index')
nx.set_node_attributes(G,node_attr)
summary = summary_graph(G)
trx_in = trx_df.groupby('ORIGEN')[['MONTO']].sum().rename(columns={'MONTO':'MONTO_SALIDA'})
trx_out = trx_df.groupby('DESTINO')[['MONTO']].sum().rename(columns={'MONTO':'MONTO_ENTRADA'})
amount_summary = pd.concat([trx_out,trx_in],axis=1)
summary_concat = pd.concat([summary,user_df[user_df['USERS'].isin(list(summary.index))].set_index('USERS'),amount_summary.loc[list(summary.index),:]],axis=1)
summary_concat['color'] = summary_concat.apply(lambda x: color(x),axis=1)
summary_concat['title'] = summary_concat.apply(lambda x: title(x),axis=1)
summary_concat['shape'] = summary_concat.apply(lambda x: shape(x),axis=1)
summary_concat['size'] = summary_concat.apply(lambda x: size(x),axis=1)
nx.set_node_attributes(G,summary_concat.to_dict('index'))

sospechoso = st.selectbox('SELECCIONA EL SOSPECHOSO EPIC007:',filter)
deph = st.selectbox('SELECCIONE LA PROFUNDIDAD', (1,2,3,4))
orientacion = st.radio("SELECCIONE LA ORIENTACION",["SALIDA","ENTRADA","AMBOS"])
on_filter = st.toggle('ACTIVAR FILTRO EN NODOS VECINOS')
if on_filter:
    filter_attr = st.selectbox("SELECCIONE EL FILTRO:",["DEPARTAMENTO","TYPE","FLAG_AN"])
    attr = st.selectbox("SELECCIONA EL ATRIBUTO A CUMPLIR:",diccionario[filter_attr])
    print(diccionario[filter_attr])
columns = ['in_degree','out_degree','in_degree_centrality','out_degree_centrality','auth_values','betweeness_centrality','closeness_centrality','hubs_values','center_nodes','periphery','avg_neighbor_degree','TYPE','FLAG_LSB_NP','FLAG_AN','FLAG_EPIC007','FLAG_PEP','FLAG_ROS','DEPARTAMENTO','EDAD','MONTO_SALIDA','MONTO_ENTRADA']
if orientacion == 'SALIDA':
    ego_graph = nx.ego_graph(G,sospechoso,radius=int(deph),undirected=False)
    F = ego_graph.copy()
    if on_filter:
        if attr!='TODOS':
            F.remove_nodes_from([n for n, q in F.nodes(data=filter_attr) if (q != attr) & (n!=sospechoso)])
        lower, upper = st.slider('FILTRAR MONTO:',0.0,5E4,(0.0,5E4),step=250.0)
        F.remove_edges_from([(n1,n2) for n1, n2, w in F.edges(data='MONTO') if (w>upper) | (w<lower)])
        nodes = nx.isolates(F.to_undirected())
        F.remove_nodes_from(nodes)
    nx.set_node_attributes(F,summary_concat.loc[list(F.nodes())].to_dict('index'))
    nt = net.Network(notebook=True,directed=True,width='800px', height='800px')
    nt.force_atlas_2based()
    nt.set_edge_smooth('dynamic')
    nt.from_nx(F)
    nt.write_html('src/salida.html')
    HtmlFile = open('src/salida.html','r',encoding='utf-8')
    source_code = HtmlFile.read()
    st.components.v1.html(source_code,width=800,height=800,scrolling=False)
    with st.expander("Revisar resumen:"):
        st.dataframe(summary_concat.loc[list(F.nodes()),columns])

elif orientacion == 'ENTRADA':
    ego_graph = nx.ego_graph(G.reverse(),sospechoso,radius=int(deph),undirected=False)
    ego_graph = ego_graph.reverse()
    F = ego_graph.copy()
    if on_filter:
        if attr!='TODOS':
            F.remove_nodes_from([n for n, q in F.nodes(data=filter_attr) if (q != attr) & (n!=sospechoso)])
        lower, upper = st.slider('FILTRAR MONTO:',0.0,5E4,(0.0,5E4),step=250.0)
        F.remove_edges_from([(n1,n2) for n1, n2, w in F.edges(data='MONTO') if (w>upper) | (w<lower)])
        nodes = nx.isolates(F.to_undirected())
        F.remove_nodes_from(nodes)
    nx.set_node_attributes(F,summary_concat.loc[list(F.nodes())].to_dict('index'))
    nt = net.Network(notebook=True,directed=True,width='800px', height='800px')
    nt.force_atlas_2based()
    nt.set_edge_smooth('dynamic')
    nt.from_nx(F)
    nt.write_html('src/entrada.html')
    HtmlFile = open('src/entrada.html','r',encoding='utf-8')
    source_code = HtmlFile.read()
    st.components.v1.html(source_code,width=800,height=800,scrolling=False)
    with st.expander("Revisar resumen:"):
        st.dataframe(summary_concat.loc[list(F.nodes()),columns])

elif orientacion == 'AMBOS':
    ego_graph = nx.ego_graph(G.to_undirected(),sospechoso,radius=int(deph),undirected=True)
    sub_graph = G.subgraph(list(ego_graph.nodes()))
    F = sub_graph.copy()
    if on_filter:
        if attr!='TODOS':
            F.remove_nodes_from([n for n, q in F.nodes(data=filter_attr) if (q != attr) & (n!=sospechoso)])
        lower, upper = st.slider('FILTRAR MONTO:',0.0,5E4,(0.0,5E4),step=250.0)
        F.remove_edges_from([(n1,n2) for n1, n2, w in F.edges(data='MONTO') if (w>upper) | (w<lower)])
        nodes = nx.isolates(F.to_undirected())
        F.remove_nodes_from(nodes)
    nx.set_node_attributes(F,summary_concat.loc[list(F.nodes())].to_dict('index'))
    nt = net.Network(notebook=True,directed=True,width='800px', height='800px')
    nt.force_atlas_2based()
    nt.set_edge_smooth('dynamic')
    nt.from_nx(F)
    nt.write_html('src/ambos.html')
    HtmlFile = open('src/ambos.html','r',encoding='utf-8')
    source_code = HtmlFile.read()
    st.components.v1.html(source_code,width=800,height=800,scrolling=False)
    with st.expander("Revisar resumen:"):
        st.dataframe(summary_concat.loc[list(F.nodes()),columns])
