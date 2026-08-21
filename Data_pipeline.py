# -*- coding: utf-8 -*-
"""
Main Pipeline - Converted from Jupyter Notebook
Generated on: 2025-11-30
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import networkx as nx
from datetime import timedelta
from datetime import datetime
import sortedcontainers
from sortedcontainers import SortedSet
import river
from river import drift
import itertools
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import tensorflow as tf
import os, json, pickle, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras import layers, Model, Input
import glob
from collections import defaultdict
from itertools import combinations

# =============================================================================
# DATA CLEANING
# =============================================================================

file_path = 'egar.csv'
data_year=2022 #should be defined by user, for removing residual instances from the year before and after

egar = pd.read_csv(file_path, sep=';', dtype={"ler_codigo": "string", "produtor_nif": "string", "destinatario_nif": "string",
"ler_recebido_codigo": "string"}, parse_dates=['data_criacao','data_estado','data_inicio_transporte','data_fim_transporte'])

bool_cols = ['ler_perigosidade','ler_recebido_perigosidade', 'autorizada', 'mercadorias_perigosas']
for col in bool_cols:
    egar[col] = egar[col].map({'True': True, 'False': False}).astype('bool')

# List of columns to replace <NA> with NaN
columns_to_replace = ["ler_codigo", "produtor_nif", "destinatario_nif", "ler_recebido_codigo"]
egar[columns_to_replace] = egar[columns_to_replace].applymap(lambda x: np.nan if pd.isna(x) else x)

valid_states = ["Concluída (certificado de receção)", "Aceite", "Corrigida", "Correção negada", "Emitida"]
egar=egar[egar.estado.isin(valid_states)]

egar= egar[egar['data_inicio_transporte'].dt.year == data_year]

def nif_classification(nif):
    if nif is None or (isinstance(nif, float) and np.isnan(nif)):
        return "nif_desconhecido"
    # ensure it's a str
    n = str(nif).strip()
    if n == "" or n.lower() == "nan":
        return "nif_desconhecido"
    # now safe to index
    if n.startswith(('1', '2', '3')) or n.startswith('45'):
        return "pessoa_singular"
    if n.startswith('5'):
        return "empresa"
    # second character == '6' rule: check length first
    if len(n) > 1 and n[1] == '6':
        return "admin_publica"
    if n.startswith(('7', '9')):
        return "casos_especiais"
    if n.startswith('8'):
        return "nif_invalido"
    return "nif_estrangeiro"
# Filter 'pessoa_singular' and 'empresa' nodes
egar['produtor_class'] = egar['produtor_nif'].apply(nif_classification)
egar['destinatario_class'] = egar['destinatario_nif'].apply(nif_classification)
# Filter relevant data
egar = egar[egar['destinatario_class'] == 'empresa']
dates = pd.to_datetime(egar['data_inicio_transporte']).dt.date
empresas = egar['destinatario_nif'].unique()

def clean_quantidade(row):
    """Function to determine the value of 'quantidade_clean' based on the logic."""
    if row['quantidade'] > 50000 and row['quantidade_recebida'] > 50000:
        # Both are greater than 50000
        return row['quantidade_recebida']
    elif row['quantidade'] > 50000:
        # 'quantidade' is greater than 50000
        return row['quantidade_recebida']
    elif row['quantidade_recebida'] > 50000:
        # 'quantidade_recebida' is greater than 50000, take 'quantidade'
        return row['quantidade']
    else:
        # None are greater than 50000
        return row['quantidade_recebida']

# Apply the function to create the new column
egar['quantidade_clean'] = egar.apply(clean_quantidade, axis=1)

egar = egar[egar['produtor_origem'] != 'OBRAS_RCD']

egar.to_csv('egar_2022_clean.csv')

# =============================================================================
# FEATURE CREATION
# =============================================================================

egar = pd.read_csv(
    'egar_2022_clean.csv', index_col=0,
    dtype={
        "ler_codigo": "string", "produtor_nif": "string",
        "destinatario_nif": "string", "ler_recebido_codigo": "string"
    },
    parse_dates=[
        'data_criacao','data_estado',
        'data_inicio_transporte','data_fim_transporte'
    ]
)

egar['week_start'] = egar['data_inicio_transporte'].dt.to_period('W').apply(lambda r: r.start_time)
egar['week_start_str'] = egar['week_start'].dt.strftime('%Y-%m-%d')

empresas = egar.destinatario_nif.unique().tolist()

## Weekly Nr of Individual Senders

relevant = egar[egar['produtor_class']=='pessoa_singular'].copy()
relevant['week'] = relevant['week_start_str']

weeks = SortedSet(relevant['week'])
weekly_nrindividuals = pd.DataFrame(index=weeks, columns=empresas, data=0)

tmp = (relevant
       .groupby(['week','destinatario_nif'])['produtor_nif']
       .nunique())

for (week, empresa), value in tmp.items():
    weekly_nrindividuals.loc[week, empresa] = value

weekly_nrindividuals = weekly_nrindividuals.iloc[1:]
weekly_nrindividuals.to_csv("weekly_nrindividuals.csv")

## Weekly Waste Quantity Received

qty = (egar
       .pivot_table(
            index=egar['week_start'],
            columns='destinatario_nif',
            values='quantidade_clean',
            aggfunc='sum',
            fill_value=0)
      )

qty.index = qty.index.strftime('%Y-%m-%d')
qty = qty.iloc[1:]
qty.to_csv("weekly_qty_received.csv")

## Weekly Waste Quantity Received from Individual Senders

indiv = egar[egar['produtor_class']=='pessoa_singular'].copy()
indiv['week'] = indiv['week_start_str']

qty_indiv = (indiv
             .pivot_table(
                 index='week',
                 columns='destinatario_nif',
                 values='quantidade_recebida',
                 aggfunc='sum',
                 fill_value=0)
            )

qty_indiv = qty_indiv.reindex(columns=empresas, fill_value=0)
qty_indiv = qty_indiv.iloc[1:]
qty_indiv.to_csv("weekly_qty_individuals.csv")

## Weekly Betweenness centrality

bet_df = pd.DataFrame(index=SortedSet(egar['week_start_str']), columns=empresas, data=0.0)

for week in bet_df.index:
    edges = egar[egar['week_start_str']==week][['produtor_nif','destinatario_nif']]
    G = nx.DiGraph()
    G.add_edges_from(edges.itertuples(index=False, name=None))
    bc = nx.betweenness_centrality(G, normalized=True)
    for c in empresas:
        bet_df.loc[week, c] = bc.get(c, 0.0)

bet_df = bet_df.iloc[1:]
bet_df.to_csv("weekly_betweenness.csv")

## Weekly Ego network density

ego_df = pd.DataFrame(index=SortedSet(egar['week_start_str']), columns=empresas, data=np.nan)

for week in ego_df.index:
    edges = (egar[egar['week_start_str']==week]
             [['produtor_nif','destinatario_nif']]
             .drop_duplicates())
    G = nx.DiGraph()
    G.add_edges_from(edges.itertuples(index=False, name=None))

    UG = nx.Graph(G)  # undirected

    for c in empresas:
        if c in UG:
            ego = nx.ego_graph(UG, c, radius=1)
            ego.remove_edges_from(nx.selfloop_edges(ego))
            n = ego.number_of_nodes()
            ego_df.loc[week, c] = np.nan if n < 3 else nx.density(ego)

ego_df = ego_df.iloc[1:]
ego_df.to_csv("weekly_ego_density.csv")

## Weekly number of Triangles

def weekly_triangle_counts_fast(df):
    results = {}

    for week, grp in df.groupby('week_start_str'):
        G = nx.Graph()
        G.add_edges_from(grp[['produtor_nif','destinatario_nif']].itertuples(index=False, name=None))
        
        triangles = set()

        for u in G.nodes():
            Nu = set(G.neighbors(u))
            for v in Nu:
                Nv = set(G.neighbors(v))
                w_set = Nu & Nv
                for w in w_set:
                    tri = tuple(sorted([u,v,w]))
                    triangles.add(tri)

        # Count participation only for empresas
        counts = {e:0 for e in empresas}
        for a,b,c in triangles:
            for x in (a,b,c):
                if x in counts:
                    counts[x]+=1

        results[week] = counts

    return pd.DataFrame(results).T

triangles_weekly = weekly_triangle_counts_fast(egar)
triangles_weekly = triangles_weekly.iloc[1:]
triangles_weekly.to_csv("weekly_triangle_counts_companies.csv")

## Monthly Connectivity with Individual Senders in Triangles

egar['month'] = egar['data_inicio_transporte'].dt.to_period('M').astype(str)
# Individual senders = those with produtor_class == "pessoa_singular"
individual_senders = (
    egar.loc[egar['produtor_class'] == "pessoa_singular", 'produtor_nif']
    .unique()
    .tolist()
)

def monthly_triangle_counts_fast(df, individual_senders):
    """
    Compute monthly triangle participation counts for individual senders only.
    Fast adjacency-based triangle detection.
    """
    months = sorted(df['month'].unique())
    results = {}

    for month in months:
        sub = df[df['month'] == month][['produtor_nif', 'destinatario_nif']]

        G = nx.Graph()
        G.add_edges_from(sub.itertuples(index=False, name=None))

        neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
        triangles = set()

        for u in neighbors:
            Nu = neighbors[u]
            for v in Nu:
                if v <= u:
                    continue
                wsp = Nu & neighbors[v]
                for w in wsp:
                    if w <= v:
                        continue
                    triangles.add(tuple(sorted([u, v, w])))

        # Count participation per node
        counts = {}
        for a, b, c in triangles:
            for x in (a, b, c):
                counts[x] = counts.get(x, 0) + 1

        # keep only individual senders
        row = {i: counts.get(i, 0) for i in individual_senders}
        results[month] = row

    return pd.DataFrame.from_dict(results, orient='index').fillna(0)


monthly_triangle_counts_individuals = monthly_triangle_counts_fast(
    egar, individual_senders
)

monthly_triangle_counts_individuals.index.name = None
monthly_triangle_counts_individuals.to_csv("monthly_triangle_counts_individual_senders.csv")

links = egar[['month', 'produtor_nif', 'destinatario_nif']].copy()
receivers = empresas

connectivity = pd.DataFrame(
    0.0,
    index=monthly_triangle_counts_individuals.index,
    columns=receivers
)

for month in connectivity.index:
    ml = links[links['month'] == month]

    triangle_row = monthly_triangle_counts_individuals.loc[month]

    for comp in receivers:
        # all senders to this company
        senders = ml.loc[ml['destinatario_nif'] == comp, 'produtor_nif']

        # keep only individual senders
        senders = [s for s in senders if s in triangle_row.index]

        if senders:
            connectivity.loc[month, comp] = triangle_row[senders].sum()
        else:
            connectivity.loc[month, comp] = 0.0

connectivity.index.name = None
connectivity.to_csv("monthly_connectivity_individuals_triangles.csv")

##Weekly PCA feature

nr_individuals = pd.read_csv("weekly_nrindividuals.csv", index_col=0, parse_dates=True)
qty_individuals = pd.read_csv("weekly_qty_individuals.csv", index_col=0, parse_dates=True)
bet = pd.read_csv("weekly_betweenness.csv", index_col=0, parse_dates=True)
ego = pd.read_csv("weekly_ego_density.csv", index_col=0, parse_dates=True)
tri = pd.read_csv("weekly_triangle_counts_companies.csv", index_col=0, parse_dates=True)

features = {
    'bet': bet.fillna(0),
    'ego': ego.fillna(0),
    'tri': tri.fillna(0),
    'indiv_cnt': nr_individuals.fillna(0),
    'indiv_qty': qty_individuals.fillna(0)
}

# Standardize each feature frame
for k in features:
    scaler = StandardScaler()
    df = features[k]
    features[k] = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# PCA per company
first_PC = pd.DataFrame(index=features['bet'].index)
explained = pd.Series(index=empresas)

for c in empresas:
    X = pd.concat([f[c] for f in features.values()], axis=1)
    ipca = IncrementalPCA(n_components=1)
    pc = ipca.fit_transform(X)
    first_PC[c] = pc[:,0]
    explained[c] = ipca.explained_variance_ratio_[0]

first_PC = first_PC.iloc[1:]
first_PC.to_csv("first_components_df_4_weekly.csv")

# =============================================================================
# CHANGE DETECTION
# =============================================================================

## PH-Ego_Density

# ---------------------------------------------------------
# LOAD + PREP
# ---------------------------------------------------------
df = pd.read_csv('weekly_ego_density.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Parameters
delta = 1.5
threshold = 2
min_instances = 3

# ---------------------------------------------------------
# FILTER ACTIVE COMPANIES + STANDARDIZE
# ---------------------------------------------------------
active_companies = (df != 0).mean()
active_companies = active_companies[active_companies > 0.2].index

ts = df[active_companies].replace(0, np.nan)
ts = (ts - ts.mean()) / ts.std()

# ---------------------------------------------------------
# PAGE–HINKLEY FOR EACH COMPANY
# ---------------------------------------------------------
scores = {}

for company in ts.columns:
    ph = drift.PageHinkley(
        threshold=threshold,
        min_instances=min_instances,
        delta=delta,
        mode='both'
    )

    strengths = []

    for value in ts[company]:
        if pd.isna(value):
            strengths.append(0)
            continue

        ph.update(value)

        if ph.drift_detected:
            su = ph._sum_increase - ph._min_increase
            sd = ph._sum_decrease - ph._max_decrease
            strengths.append(max(su, sd))
        else:
            strengths.append(0)

    # Top-3 cumulative change strengths
    strengths = np.array(strengths)
    top3 = strengths[strengths > 0]
    top3 = np.sort(top3)[-3:].sum() if len(top3) else 0
    scores[company] = top3

# ---------------------------------------------------------
# ENSURE ALL COMPANIES INCLUDED (missing = 0)
# ---------------------------------------------------------
all_scores = {c: scores.get(c, 0) for c in df.columns}

ph_strengths_cum_df = pd.DataFrame({
    'nif': list(all_scores.keys()),
    'score': list(all_scores.values())
})

ph_strengths_cum_df.to_csv('ph_scores_egodensity.csv', index=False)

## PH-Triangles

# ---------------------------------------------------------
# LOAD + PREP
# ---------------------------------------------------------
df = pd.read_csv('monthly_connectivity_individuals_triangles.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Parameters
delta = 0.7
threshold = 1.2
min_instances=1

# ---------------------------------------------------------
# STANDARDIZE
# ---------------------------------------------------------

ts = df.replace(0, np.nan)
ts = (ts - ts.mean()) / ts.std()

# ---------------------------------------------------------
# PAGE–HINKLEY FOR EACH COMPANY
# ---------------------------------------------------------
scores = {}

for company in ts.columns:
    ph = drift.PageHinkley(
        threshold=threshold,
        min_instances=min_instances,
        delta=delta,
        mode='both'
    )

    strengths = []

    for value in ts[company]:
        if pd.isna(value):
            strengths.append(0)
            continue

        ph.update(value)

        if ph.drift_detected:
            su = ph._sum_increase - ph._min_increase
            sd = ph._sum_decrease - ph._max_decrease
            strengths.append(max(su, sd))
        else:
            strengths.append(0)

    # Top-3 cumulative change strengths
    strengths = np.array(strengths)
    top3 = strengths[strengths > 0]
    top3 = np.sort(top3)[-3:].sum() if len(top3) else 0
    scores[company] = top3

# ---------------------------------------------------------
# ENSURE ALL COMPANIES INCLUDED (missing = 0)
# ---------------------------------------------------------
all_scores = {c: scores.get(c, 0) for c in df.columns}

ph_strengths_cum_df = pd.DataFrame({
    'nif': list(all_scores.keys()),
    'score': list(all_scores.values())
})

ph_strengths_cum_df.to_csv('ph_scores_triangles.csv', index=False)

## PH-PCA

# ---------------------------------------------------------
# LOAD + PREP
# ---------------------------------------------------------
df = pd.read_csv('first_components_df_4_weekly.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Parameters
delta = 2
threshold = 3
min_instances=3

# ---------------------------------------------------------
# FILTER ACTIVE COMPANIES + STANDARDIZE
# ---------------------------------------------------------
ts = df.replace(0, np.nan)
ts = (ts - ts.mean()) / ts.std()

nr_individuals = pd.read_csv("weekly_nrindividuals.csv", index_col=0, parse_dates=True)
qty_individuals = pd.read_csv("weekly_qty_individuals.csv", index_col=0, parse_dates=True)
bet = pd.read_csv("weekly_betweenness.csv", index_col=0, parse_dates=True)
ego = pd.read_csv("weekly_ego_density.csv", index_col=0, parse_dates=True)
tri = pd.read_csv("weekly_triangle_counts_companies.csv", index_col=0, parse_dates=True)

features = {
    'betweenness_cent': bet,
    'ego_density': ego,
    'nr_triangles': tri,
    'nr_individuals': nr_individuals,
    'qty_received_individuals': qty_individuals
}

# Minimum number of features that must be active at a time point
min_active_features = 3

activity_masks = [df != 0 for df in features.values()]
combined_activity = sum(activity_masks)
active_enough = combined_activity >= min_active_features
activity_ratio = active_enough.mean(axis=0)

threshold = 0.2  # must be active ≥ X% of the time

active_companies = activity_ratio[activity_ratio > threshold].index.tolist()
ts = ts[active_companies]

# ---------------------------------------------------------
# PAGE–HINKLEY FOR EACH COMPANY
# ---------------------------------------------------------
scores = {}

for company in ts.columns:
    ph = drift.PageHinkley(
        threshold=threshold,
        min_instances=min_instances,
        delta=delta,
        mode='both'
    )

    strengths = []

    for value in ts[company]:
        if pd.isna(value):
            strengths.append(0)
            continue

        ph.update(value)

        if ph.drift_detected:
            su = ph._sum_increase - ph._min_increase
            sd = ph._sum_decrease - ph._max_decrease
            strengths.append(max(su, sd))
        else:
            strengths.append(0)

    # Top-3 cumulative change strengths
    strengths = np.array(strengths)
    top3 = strengths[strengths > 0]
    top3 = np.sort(top3)[-3:].sum() if len(top3) else 0
    scores[company] = top3

# ---------------------------------------------------------
# ENSURE ALL COMPANIES INCLUDED (missing = 0)
# ---------------------------------------------------------
all_scores = {c: scores.get(c, 0) for c in df.columns}

ph_strengths_cum_df = pd.DataFrame({
    'nif': list(all_scores.keys()),
    'score': list(all_scores.values())
})

ph_strengths_cum_df.to_csv('ph_scores_pca.csv', index=False)

# =============================================================================
# LSTM-VAE
# =============================================================================

qty_received = pd.read_csv("weekly_qty_received.csv", index_col=0, parse_dates=True)
bet = pd.read_csv("weekly_betweenness.csv", index_col=0, parse_dates=True)

# -----------------------------
# CONFIG
# -----------------------------
WINDOW = 10
LATENT = 6
FEATURE_LIST = {
    'betweenness_cent': bet,
    'qty_received': qty_received
}
WEIGHTS_FILE = "lstm_vae.weights.h5"   # <--- name of saved file
OUTPUT_FILE = "lstm_vae_scores.csv"

companies = list(FEATURE_LIST[next(iter(FEATURE_LIST))].columns)
periods = sorted(set.intersection(*(set(df.index) for df in FEATURE_LIST.values())))
N_FEATURES = len(FEATURE_LIST)

# -----------------------------
# MODEL DEFINITION (same as training)
# -----------------------------
class LSTMVAE(tf.keras.Model):
    def __init__(self, window_size, n_features, latent_dim=4, beta=1.0, lstm_units=32):
        super().__init__()
        self.beta = beta

        self.encoder_inputs = tf.keras.Input(shape=(window_size, n_features))
        x = tf.keras.layers.LSTM(lstm_units)(self.encoder_inputs)
        self.z_mean = tf.keras.layers.Dense(latent_dim)(x)
        self.z_log_var = tf.keras.layers.Dense(latent_dim)(x)
        self.encoder = tf.keras.Model(self.encoder_inputs, [self.z_mean, self.z_log_var])

        self.decoder_inputs = tf.keras.Input(shape=(latent_dim,))
        x_dec = tf.keras.layers.RepeatVector(window_size)(self.decoder_inputs)
        x_dec = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x_dec)
        decoder_outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_features)
        )(x_dec)

        self.decoder = tf.keras.Model(self.decoder_inputs, decoder_outputs)

    def sample(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sample(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed

# -----------------------------
# RECONSTRUCTION ERROR FUNCTION
# -----------------------------
def reconstruction_errors(model, windows):
    pred = model.predict(windows, verbose=0)
    return np.mean((windows - pred) ** 2, axis=(1, 2))

# -----------------------------
# LOAD MODEL + WEIGHTS
# -----------------------------
vae = LSTMVAE(WINDOW, N_FEATURES, latent_dim=LATENT, beta=0.1)
vae.compile(optimizer='adam')
vae.predict(np.zeros((1, WINDOW, N_FEATURES)))      # build model
vae.load_weights(WEIGHTS_FILE)

# -----------------------------
# SCALE FEATURES
# -----------------------------
scalers = {}
for feat, df in FEATURE_LIST.items():
    vals = df.loc[periods, companies].values.flatten()

    if feat == "betweenness_cent":
        # Include zeros
        vals_to_scale = vals.reshape(-1, 1)
    else:
        # Exclude zeros (for qty_received)
        vals_to_scale = vals[vals != 0].reshape(-1, 1)

    scalers[feat] = StandardScaler().fit(vals_to_scale) if len(vals_to_scale) > 0 else None

# -----------------------------
# BUILD WINDOWS FOR EACH COMPANY
# -----------------------------
company_scores = {}

for company in companies:
    # stack features
    feat_cols = []
    for feat, df in FEATURE_LIST.items():
        raw = df.loc[periods, company].fillna(0).values.reshape(-1, 1)
        scaler = scalers[feat]
        scaled = scaler.transform(raw) if scaler is not None else raw
        feat_cols.append(scaled)

    mat = np.hstack(feat_cols)

    # sliding windows
    windows = np.array([
        mat[i:i + WINDOW]
        for i in range(len(periods) - WINDOW + 1)
    ])

    if len(windows) == 0:
        company_scores[company] = 0
        continue

    # get reconstruction errors
    errs = reconstruction_errors(vae, windows)

    # z-score normalize per company
    errs = (errs - errs.mean()) / (errs.std() + 1e-6)

    # pick top-10 spikes
    top10 = np.sort(errs)[-10:]
    company_scores[company] = top10.mean()

# -----------------------------
# OUTPUT TO CSV
# -----------------------------
output_df = pd.DataFrame({
    "nif": list(company_scores.keys()),
    "score": list(company_scores.values())
})

output_df.to_csv(OUTPUT_FILE, index=False)

# =============================================================================
# GRAPH CONSTRUCTION (and CM-GCN-LSTM-VAE setup)
# =============================================================================

"""
Inference-only pipeline for:
 - building / loading weekly motif graph cache
 - loading MotifGCN saved weights (best_motif_gcn.pt) to produce per-week company embeddings
 - loading trained LSTM-VAE weights and computing reconstruction error-based anomaly scores
Outputs: motif_gcn_scores.csv

"""
# ----------------------------
# USER CONFIG (edit if needed)
# ----------------------------
DATA_DIR = "."                                  # location of egar_2022_clean.csv
GRAPH_CACHE_DIR = os.path.join(DATA_DIR, "weekly_graphs")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
SCORES_CSV = os.path.join(DATA_DIR, "motif_gcn_scores.csv")

EGAR_FILE = os.path.join(DATA_DIR, "egar_2022_clean.csv")

MOTIF_GCN_WEIGHTS = os.path.join(DATA_DIR, "motif_gcn_model.pt")  
VAE_WEIGHTS_PATH = os.path.join(DATA_DIR, "cm_gcn_lstmvae.weights.h5")

# parameters that must match those used during training
EMBED_DIM = 8
WINDOW_SIZE = 8
LATENT_DIM = 8
LSTM_UNITS = 32
BETA = 0.5
VAE_VERBOSE = 0

# create folders
for d in (GRAPH_CACHE_DIR, EMB_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------
# device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load data (same parsing as training)
# ----------------------------
egar_df = pd.read_csv(
    EGAR_FILE,
    index_col=0,
    dtype={
        "ler_codigo": "string",
        "produtor_nif": "string",
        "destinatario_nif": "string",
        "ler_recebido_codigo": "string"
    },
    parse_dates=[
        "data_criacao",
        "data_estado",
        "data_inicio_transporte",
        "data_fim_transporte"
    ]
)
companies = egar_df.destinatario_nif.unique().tolist()
companies_set = set(companies)

egar_df['data_inicio_transporte'] = pd.to_datetime(egar_df['data_inicio_transporte'])
egar_df['week'] = egar_df['data_inicio_transporte'].dt.isocalendar().week
egar_df = egar_df[egar_df['data_inicio_transporte'] >= '2022-01-03']
weeks = sorted(egar_df['week'].unique())

#CONSTRUCT THE GRAPHS

# ----------------------------
# Helper functions
# ----------------------------
def build_week_edges(df_week):
    df_week = df_week.dropna(subset=['produtor_nif', 'destinatario_nif'])
    return [(str(u), str(v)) for u, v in zip(df_week['produtor_nif'], df_week['destinatario_nif'])]

def prune_nodes_2hop(edges, companies_set):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    first_hop = set()
    for c in companies_set:
        if c in adj:
            first_hop.update(adj[c])
    second_hop = set()
    for n in first_hop:
        if n in adj:
            second_hop.update(adj[n])
    nodes = set(companies_set) | first_hop | second_hop
    return nodes

def compute_label_aware_Mhat(node_list, edge_list, node_classes,
                             weight_individual_pairs: float = 1.0,
                             weight_company_pairs: float = 1.0):
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    nbrs = {n: set(G.neighbors(n)) for n in node_list}
    M = np.zeros((N, N), dtype=np.float32)
    tri_total = 0
    sorted_nodes = sorted(node_list)
    node_order = {n: i for i, n in enumerate(sorted_nodes)}
    for u in sorted_nodes:
        for v in (n for n in nbrs[u] if node_order[n] > node_order[u]):
            common = [w for w in nbrs[u].intersection(nbrs[v]) if node_order[w] > node_order[v]]
            if not common:
                continue
            for w in common:
                if not (node_classes.get(u, 2) == 1 or node_classes.get(v, 2) == 1 or node_classes.get(w, 2) == 1):
                    continue
                triplet = [(u, v), (v, w), (w, u)]
                for a, b in triplet:
                    ai = idx[a]; bi = idx[b]
                    if node_classes.get(a, 2) == 1 or node_classes.get(b, 2) == 1:
                        wt = float(weight_individual_pairs)
                    else:
                        wt = float(weight_company_pairs)
                    M[ai, bi] += wt
                    M[bi, ai] += wt
                tri_total += 1
    M_plus_I = M + np.eye(N, dtype=np.float32)
    row_sums = M_plus_I.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(row_sums, 1e-12))
    D_inv_sqrt = np.diag(inv_sqrt)
    M_hat = D_inv_sqrt @ M_plus_I @ D_inv_sqrt
    nz_rows, nz_cols = np.nonzero(M_hat)
    nz_vals = M_hat[nz_rows, nz_cols].astype(np.float32)
    indices = np.vstack([nz_rows, nz_cols])
    i_tensor = torch.LongTensor(indices)
    v_tensor = torch.FloatTensor(nz_vals)
    M_hat_sparse = torch.sparse_coo_tensor(i_tensor, v_tensor, (N, N)).coalesce()
    stats = {"num_triangles_counted": tri_total, "nnz": len(nz_vals), "mean_weight": float(np.mean(nz_vals)) if len(nz_vals) > 0 else 0.0}
    return M_hat_sparse, stats

def construct_weekly_cache(weeks, egar_df, companies_set, graph_cache_dir):
    index = {}
    for w in weeks:
        save_path = os.path.join(graph_cache_dir, f"week_{w:02d}.pkl")
        if os.path.exists(save_path):
            index[w] = save_path
            continue
        df_w = egar_df[egar_df['week'] == w]
        edges = build_week_edges(df_w)
        if len(edges) == 0:
            payload = {"week": w, "node_list": [], "edge_list": [], "node_classes": {}, "M_hat_indices": None, "M_hat_values": None, "M_hat_size": None, "motif_stats": {}}
            with open(save_path, "wb") as f:
                pickle.dump(payload, f)
            index[w] = save_path
            continue
        nodes_to_keep = prune_nodes_2hop(edges, companies_set)
        G = nx.Graph()
        G.add_nodes_from(nodes_to_keep)
        filtered_edges = [(u, v) for (u, v) in edges if u in nodes_to_keep and v in nodes_to_keep]
        G.add_edges_from(filtered_edges)
        node_list = list(G.nodes())
        node_classes = {}
        for _, row in df_w.iterrows():
            u = str(row['produtor_nif']); v = str(row['destinatario_nif'])
            ucls = row.get('produtor_class', None); vcls = row.get('destinatario_class', None)
            if pd.notna(ucls) and u not in node_classes:
                node_classes[u] = 1 if str(ucls) == "pessoa_singular" else 0 if str(ucls) == "empresa" else 2
            if pd.notna(vcls) and v not in node_classes:
                node_classes[v] = 1 if str(vcls) == "pessoa_singular" else 0 if str(vcls) == "empresa" else 2
        for n in node_list:
            if n not in node_classes:
                node_classes[n] = 0 if n in companies_set else 1
        M_hat_sparse, motif_stats = compute_label_aware_Mhat(node_list, list(G.edges()), node_classes)
        payload = {
            "week": w,
            "node_list": node_list,
            "edge_list": list(G.edges()),
            "node_classes": node_classes,
            "M_hat_indices": M_hat_sparse.coalesce().indices().cpu().numpy() if M_hat_sparse is not None else None,
            "M_hat_values": M_hat_sparse.coalesce().values().cpu().numpy() if M_hat_sparse is not None else None,
            "M_hat_size": M_hat_sparse.size() if M_hat_sparse is not None else (len(node_list), len(node_list)),
            "motif_stats": motif_stats
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)
        index[w] = save_path
    with open(os.path.join(graph_cache_dir, "index.json"), "w") as f:
        json.dump({int(k): v for k, v in index.items()}, f)
    return index

# ----------------------------
# Build or load weekly cache
# ----------------------------
cache_index = construct_weekly_cache(weeks, egar_df, companies_set, GRAPH_CACHE_DIR)

# =============================================================================
# CM-GCN-LSTM-VAE
# =============================================================================

# ----------------------------
# MotifGCN
# ----------------------------
CLASS_EMB_DIM = 8
HIDDEN = 16
GCN_DROPOUT = 0.0

class MotifGCN(nn.Module):
    def __init__(self, num_classes=3, hidden=HIDDEN, embed_dim=EMBED_DIM, dropout=GCN_DROPOUT):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, CLASS_EMB_DIM)
        self.fuse_lin = nn.Linear(CLASS_EMB_DIM, hidden)
        self.hidden_to_embed = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.motif_alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    def forward(self, labels_long, M_hat_sparse=None):
        device = labels_long.device
        labels_long = labels_long.long()
        h_class = self.class_emb(labels_long)
        h_hidden = F.relu(self.fuse_lin(h_class))
        if M_hat_sparse is not None and M_hat_sparse.numel() > 0:
            M = M_hat_sparse.coalesce().to(device)
            motif_msg = torch.sparse.mm(M, h_hidden)
            h_comb = h_hidden + self.motif_alpha * motif_msg
        else:
            h_comb = h_hidden
        h_out = F.relu(self.hidden_to_embed(h_comb))
        h_out = self.dropout(h_out)
        h_out = F.normalize(h_out, p=2, dim=1)
        return h_out

# ----------------------------
# Load motif GCN weights and compute per-week embeddings
# ----------------------------
model = MotifGCN(num_classes=3, hidden=HIDDEN, embed_dim=EMBED_DIM, dropout=GCN_DROPOUT)
if os.path.exists(MOTIF_GCN_WEIGHTS):
    try:
        state = torch.load(MOTIF_GCN_WEIGHTS, map_location="cpu")
        model.load_state_dict(state)
    except Exception as e:
        print(f"[WARN] Failed to load MotifGCN weights from {MOTIF_GCN_WEIGHTS}: {e}")
else:
    print(f"[WARN] Motif GCN weights not found at {MOTIF_GCN_WEIGHTS}. Using randomly initialized model.")
model = model.to(DEVICE)
model.eval()

company_index_map = {comp: i for i, comp in enumerate(companies)}
final_week_embeddings = {}
emb_stats_summary = {}

for w in weeks:
    w_int = int(w)
    save_path = os.path.join(EMB_DIR, f"week_{w_int:02d}_final.npz")
    cache_file = cache_index.get(w_int)
    if cache_file is None or not os.path.exists(cache_file):
        print(f"[WARN] Missing cache for week {w_int}, saving zeros.")
        week_matrix = np.zeros((len(companies), EMBED_DIM), dtype=np.float32)
        np.savez_compressed(save_path, embeddings=week_matrix)
        final_week_embeddings[w_int] = week_matrix
        continue
    with open(cache_file, "rb") as f:
        payload = pickle.load(f)
    node_list = payload.get("node_list", [])
    if len(node_list) == 0:
        print(f"[WARN] Empty node list for week {w_int}, saving zeros.")
        week_matrix = np.zeros((len(companies), EMBED_DIM), dtype=np.float32)
        np.savez_compressed(save_path, embeddings=week_matrix)
        final_week_embeddings[w_int] = week_matrix
        continue
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    labels_arr = np.array([payload.get("node_classes", {}).get(n, 2) for n in node_list], dtype=np.int64)
    labels_t = torch.from_numpy(labels_arr).to(DEVICE)
    # Robust M_hat construction
    M_idx = payload.get("M_hat_indices", None)
    M_vals_raw = payload.get("M_hat_values", None)
    M_size = tuple(payload.get("M_hat_size", (len(node_list), len(node_list))))
    if len(M_size) != 2:
        M_size = (len(node_list), len(node_list))
    M_hat_sparse = None
    if M_idx is not None and M_vals_raw is not None:
        try:
            idx_np = np.array(M_idx)
            vals_np = np.array(M_vals_raw, dtype=np.float32)
            if vals_np.size == 0:
                raise ValueError("M_hat_values is empty")
            if idx_np.ndim == 1:
                if idx_np.size == 2 * vals_np.size:
                    idx_np = idx_np.reshape(2, -1)
                else:
                    raise ValueError("1D M_hat_indices with unexpected length")
            elif idx_np.ndim == 2 and idx_np.shape[0] == 2:
                pass
            elif idx_np.ndim == 2 and idx_np.shape[1] == 2:
                idx_np = idx_np.T
            else:
                idx_np = idx_np.T
                if idx_np.ndim != 2 or idx_np.shape[0] != 2:
                    raise ValueError(f"Unrecognized shape for M_hat_indices: {idx_np.shape}")
            i_tensor = torch.LongTensor(idx_np)
            v_tensor = torch.FloatTensor(vals_np.ravel())
            if i_tensor.shape[1] != v_tensor.numel():
                raise ValueError(f"M_hat indices count ({i_tensor.shape[1]}) != values count ({v_tensor.numel()})")
            M_hat_sparse = torch.sparse_coo_tensor(i_tensor, v_tensor, M_size).coalesce().to(DEVICE)
        except Exception as e:
            print(f"[WARN] Week {w_int}: failed to build M_hat_sparse ({e}). Continuing without motif matrix.", file=sys.stderr)
            M_hat_sparse = None
    try:
        with torch.no_grad():
            embeddings = model(labels_t, M_hat_sparse)
            emb = embeddings.cpu().numpy()
    except Exception as e:
        print(f"[ERROR] Week {w_int}: model forward failed ({e}). Saving zeros for this week.", file=sys.stderr)
        week_matrix = np.zeros((len(companies), EMBED_DIM), dtype=np.float32)
        np.savez_compressed(save_path, embeddings=week_matrix)
        final_week_embeddings[w_int] = week_matrix
        continue
    week_matrix = np.zeros((len(companies), EMBED_DIM), dtype=np.float32)
    for comp in companies:
        if comp in node_to_idx:
            week_matrix[company_index_map[comp], :] = emb[node_to_idx[comp]]
    final_week_embeddings[int(w_int)] = week_matrix
    np.savez_compressed(save_path, embeddings=week_matrix)

# ----------------------------
# LSTM-VAE model (same architecture used during training)
# ----------------------------
class LSTMVAE(Model):
    def __init__(self, window_size, n_features, latent_dim=4, beta=1.0, lstm_units=32, dropout=0.0, rec_dropout=0.0):
        super().__init__()
        self.beta = beta
        self.window_size = window_size
        self.encoder_inputs = Input(shape=(window_size, n_features))
        x = layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=rec_dropout)(self.encoder_inputs)
        self.z_mean = layers.Dense(latent_dim)(x)
        self.z_log_var = layers.Dense(latent_dim)(x)
        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_var])
        self.decoder_inputs = Input(shape=(latent_dim,))
        x_dec = layers.RepeatVector(window_size)(self.decoder_inputs)
        x_dec = layers.LSTM(lstm_units, return_sequences=True)(x_dec)
        decoder_outputs = layers.TimeDistributed(layers.Dense(n_features))(x_dec)
        self.decoder = Model(self.decoder_inputs, decoder_outputs)
    def sample(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sample(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

def reconstruction_errors(model, windows):
    recon = model.predict(windows, verbose=VAE_VERBOSE)
    if isinstance(recon, tuple):
        recon = recon[0]
    mse = np.mean((windows - recon) ** 2, axis=(1, 2))
    return mse

# ----------------------------
# Load embeddings tensor for VAE inference
# ----------------------------
emb_by_week = []
for w in weeks:
    fpath = os.path.join(EMB_DIR, f"week_{w:02d}_final.npz")
    if not os.path.exists(fpath):
        print(f"[WARN] Missing embeddings for week {w}, filling zeros.")
        emb = np.zeros((len(companies), EMBED_DIM), dtype=np.float32)
    else:
        emb = np.load(fpath)["embeddings"]
    emb_by_week.append(emb)
if len(emb_by_week) == 0:
    raise RuntimeError("[ERROR] No weekly embeddings found to run LSTM-VAE inference.")
emb_by_week = np.stack(emb_by_week, axis=0)

def build_company_windows(emb_by_week, companies, window_size):
    n_weeks, n_companies, embed_dim = emb_by_week.shape
    company_windows = {}
    for i, company in enumerate(companies):
        company_series = emb_by_week[:, i, :]
        if n_weeks - window_size + 1 <= 0:
            windows = np.empty((0, window_size, embed_dim), dtype=np.float32)
        else:
            windows = np.array([company_series[j:j + window_size] for j in range(n_weeks - window_size + 1)])
        if len(windows) > 0:
            company_windows[company] = windows
    return company_windows

company_windows = build_company_windows(emb_by_week, companies, WINDOW_SIZE)

# ----------------------------
# Prepare VAE weights path
# ----------------------------
if VAE_WEIGHTS_PATH is None:
    # find candidate files
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, "vae_run*_best.weights.h5")))
    if len(candidates) == 0:
        print(f"[WARN] No VAE weights found in {DATA_DIR} (pattern vae_run*_best.weights.h5). Provide VAE_WEIGHTS_PATH to proceed.")
        vae_weights_to_load = None
    else:
        vae_weights_to_load = candidates[0]
        print(f"[INFO] Auto-selected VAE weights: {vae_weights_to_load}")
else:
    vae_weights_to_load = VAE_WEIGHTS_PATH
    if not os.path.exists(vae_weights_to_load):
        print(f"[WARN] Explicit VAE_WEIGHTS_PATH {vae_weights_to_load} does not exist.")
        vae_weights_to_load = None

# ----------------------------
# Instantiate VAE model and load weights
# ----------------------------
vae = LSTMVAE(WINDOW_SIZE, EMBED_DIM, LATENT_DIM, BETA, lstm_units=LSTM_UNITS)

vae.build(input_shape=(None, WINDOW_SIZE, EMBED_DIM))
# compile
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# load weights
if vae_weights_to_load:
    try:
        vae.load_weights(vae_weights_to_load)
    except Exception as e:
        print(f"[WARN] Failed to load VAE weights: {e}")
else:
    print("[WARN] No VAE weights loaded; inference will use randomly initialized model.")

# ----------------------------
# Compute reconstruction errors per company
# ----------------------------
all_company_error_series = {}
for company, windows in company_windows.items():
    if len(windows) == 0:
        continue
    try:
        errors = reconstruction_errors(vae, windows)
    except Exception as e:
        print(f"[WARN] Reconstruction failed for company {company}: {e}")
        errors = np.array([np.nan] * len(windows))
    all_company_error_series[company] = errors

# ----------------------------
# Aggregate scores (same logic as training script)
# ----------------------------
mean_error_series_per_company = {}
for company, run_errors in all_company_error_series.items():
    # here we only have one run, so mean is the errors themselves
    mean_error_series_per_company[company] = run_errors

final_mean_errors = {}
for company, mean_series in mean_error_series_per_company.items():
    if len(mean_series) >= 10:
        top10_mean = np.mean(np.sort(mean_series)[-10:])
    else:
        top10_mean = np.nanmean(mean_series)
    final_mean_errors[company] = float(top10_mean) if not np.isnan(top10_mean) else float("nan")

sorted_final = sorted(final_mean_errors.items(), key=lambda x: (np.nan_to_num(x[1], nan=-np.inf)), reverse=True)
final_ranks = {nif: rank + 1 for rank, (nif, _) in enumerate(sorted_final)}

runs_df = pd.DataFrame({
    "nif": list(final_mean_errors.keys()),
    "score": list(final_mean_errors.values())
})
os.makedirs(DATA_DIR, exist_ok=True)
runs_df.to_csv(SCORES_CSV, index=False)

# =============================================================================
# SCORE AGGREGATION
# =============================================================================

score_files = [
    'ph_scores_triangles.csv',
    'ph_scores_pca.csv',
    'ph_scores_egodensity.csv',
    'lstm_vae_scores.csv',
    'motif_gcn_scores.csv'
]

score_df_names = [
    'PH-Triangles',
    'PH-PCA',
    'PH-EgoDensity',
    'LSTM-VAE',
    'CM-GCN-LSTM-VAE'
]

# Ensure companies are strings
companies = [str(nif) for nif in companies]

# Load all scores into a dict of Series indexed by NIF
score_dfs = {}
for name, file in zip(score_df_names, score_files):
    df = pd.read_csv(file)
    df['nif'] = df['nif'].astype(str)
    score_dfs[name] = df.set_index('nif')['score']

n_companies = len(companies)
records = []

for nif in companies:
    scores = [score_dfs[name].get(nif, 0.0) for name in score_df_names]
    normalized_ranks = []

    # Compute rank for each method and normalize
    for method_name, score in zip(score_df_names, scores):
        if score == 0.0:
            # If raw score is zero, normalized rank is forced to 0
            normalized_rank = 0.0
        else:
            series = score_dfs[method_name]
            # Rank descending: higher score = rank 1
            rank = series.rank(ascending=False, method='min').get(nif, n_companies)
            normalized_rank = (n_companies - rank) / (n_companies - 1)
        normalized_ranks.append(normalized_rank)

    # Final score: median of normalized ranks
    final_score = pd.Series(normalized_ranks).median()

    record = {'nif': nif, 'final_score': final_score}
    record.update({f"{name}_norm_rank": r for name, r in zip(score_df_names, normalized_ranks)})
    records.append(record)

# Save to DataFrame
result_df = pd.DataFrame(records)
result_df.to_csv('final_scores.csv', index=False)

print("\n=== Pipeline Complete ===")
print("Final results saved to: final_scores.csv")
