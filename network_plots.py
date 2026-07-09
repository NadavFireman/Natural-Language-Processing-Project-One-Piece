"""
network_plots.py
One Piece NLP Project — network visualization layer.

Two plotting helpers, kept out of features.py (which is about parsing and
feature engineering) so that drawing code lives on its own. Each function
computes what it needs internally (the name->bounty map, the graph layout,
node colours/sizes) and draws with matplotlib.

They keep the notebook's original 'inferno' colormap and honour whatever
matplotlib style is active (e.g. plt.style.use('ggplot')). Because the graphs
hide their axes, the active style's panel background (ggplot's grey #E5E5E5)
is painted back explicitly — otherwise matplotlib would leave it white.

Usage in the notebook, after the character graph exists:

    import features as F
    import network_plots as NP

    parsed_df = F.add_graph_features(parsed_df)      # already in your pipeline
    G  = F.build_character_graph(parsed_df)          # already in your pipeline

    NP.group_network(parsed_df)                      # crews / organizations
    NP.character_network(G, parsed_df)               # individual characters

Both return the graph object they drew (GG / H) so later cells can reuse it.
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx


def _bounty_map(parsed_df):
    """name -> bounty dict (the notebook's `bounty`)."""
    return dict(zip(parsed_df["name"], parsed_df["bounty"]))


def _keep_panel_background(ax):
    """Hide ticks and spines but KEEP the panel fill of the active style.

    `ax.axis('off')` would also drop the ggplot grey background, so instead we
    explicitly repaint the panel with the style's axes.facecolor and only
    remove ticks + spines."""
    ax.set_facecolor(plt.rcParams["axes.facecolor"])
    ax.patch.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ----------------------------------------------------------------------
# 1. Group / organization network
# ----------------------------------------------------------------------
def group_network(parsed_df, min_members=3, ax=None, show=True):
    """Crew / organization network.

    Node  = a group with >= `min_members` members.
    Edge  = two groups that share at least one character (weight = #shared).
    Size  = number of members; Colour = median log(bounty) of the members.

    Returns the built graph GG.
    """
    group_edges = Counter()
    group_members = Counter()
    for affs in parsed_df["graph_affiliations"]:
        affs = sorted(set(affs or []))
        for g in affs:
            group_members[g] += 1
        for a, b in combinations(affs, 2):
            group_edges[(a, b)] += 1

    GG = nx.Graph()
    for g, n in group_members.items():
        if n >= min_members:
            GG.add_node(g, size=n)
    for (a, b), w in group_edges.items():
        if a in GG and b in GG:
            GG.add_edge(a, b, weight=w)

    # keep only components with at least 3 groups
    keep = set()
    for comp in nx.connected_components(GG):
        if len(comp) >= 3:
            keep |= comp
    GG = GG.subgraph(keep).copy()

    bounty = _bounty_map(parsed_df)
    grp_bounty = {}
    for g in GG.nodes():
        members = [bounty[n] for n, affs in zip(parsed_df["name"], parsed_df["graph_affiliations"])
                   if g in (affs or []) and pd.notna(bounty.get(n))]
        grp_bounty[g] = np.log1p(np.median(members)) if members else np.nan

    vals = [v for v in grp_bounty.values() if not np.isnan(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)
    ncol = ["#cccccc" if np.isnan(grp_bounty[g])
            else plt.cm.inferno((grp_bounty[g] - vmin) / (vmax - vmin + 1e-9))
            for g in GG.nodes()]

    sizes = np.array([GG.nodes[g]["size"] for g in GG.nodes()], dtype=float)
    s = np.sqrt(sizes)
    nsz = 60 + 1500 * (s - s.min()) / (s.max() - s.min() + 1e-9)

    pos = nx.spring_layout(GG, seed=42, k=3.0 / np.sqrt(GG.number_of_nodes()),
                           iterations=250, weight="weight")
    xs = np.array([pos[n][0] for n in GG.nodes()])
    ys = np.array([pos[n][1] for n in GG.nodes()])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = ax.figure

    nx.draw_networkx_edges(GG, pos, ax=ax, alpha=0.15, width=0.6)
    nx.draw_networkx_nodes(GG, pos, ax=ax, node_color=ncol, node_size=nsz,
                           linewidths=0.3, edgecolors="white")
    top = sorted(GG.nodes(), key=lambda g: -GG.nodes[g]["size"])[:25]
    nx.draw_networkx_labels(GG, pos, {g: g for g in top}, font_size=8, ax=ax)

    ax.set_xlim(np.percentile(xs, [4, 96]))
    ax.set_ylim(np.percentile(ys, [6, 94]))

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, label="median log(bounty)")
    ax.set_title(f"Group network - {GG.number_of_nodes()} crews/organizations "
                 f"(size = members, colour = median bounty)")
    _keep_panel_background(ax)

    if show:
        plt.show()
    return GG


# ----------------------------------------------------------------------
# 2. Character network
# ----------------------------------------------------------------------
def character_network(G, parsed_df, k_core=3, ax=None, show=True):
    """Individual-character network.

    Draws the k-core of the giant component plus every bounty-bearing node.
    Size = log-degree; Colour = log(bounty) (grey = no bounty, faded).

    Returns the drawn subgraph H.
    """
    bounty = _bounty_map(parsed_df)

    giant = max(nx.connected_components(G), key=len)
    core = nx.k_core(G.subgraph(giant), k=k_core)
    keep = set(core.nodes()) | {n for n in giant if pd.notna(bounty.get(n))}
    H = G.subgraph(keep).copy()
    H.remove_nodes_from(list(nx.isolates(H)))

    logb = {n: (np.log1p(bounty[n]) if bounty.get(n) and not pd.isna(bounty[n]) else np.nan)
            for n in H.nodes()}
    vals = [v for v in logb.values() if not np.isnan(v)]
    vmin, vmax = (min(vals), max(vals)) if vals else (0, 1)

    degs = dict(H.degree())
    dmin, dmax = np.log1p(min(degs.values())), np.log1p(max(degs.values()))
    size = {n: 15 + 400 * (np.log1p(degs[n]) - dmin) / (dmax - dmin + 1e-9) for n in H.nodes()}

    pos = nx.spring_layout(H, seed=42, k=2.5 / np.sqrt(H.number_of_nodes()),
                           iterations=200, weight="weight")

    gray = [n for n in H.nodes() if np.isnan(logb[n])]
    col = [n for n in H.nodes() if not np.isnan(logb[n])]
    ccol = [plt.cm.inferno((logb[n] - vmin) / (vmax - vmin + 1e-9)) for n in col]

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
    else:
        fig = ax.figure

    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.04, width=0.3)
    nx.draw_networkx_nodes(H, pos, nodelist=gray, node_color="#d9d9d9",
                           node_size=[size[n] * 0.35 for n in gray], alpha=0.3,
                           linewidths=0, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=col, node_color=ccol,
                           node_size=[size[n] for n in col],
                           linewidths=0.2, edgecolors="white", ax=ax)

    top = sorted(col, key=lambda n: -logb[n])[:30]
    nx.draw_networkx_labels(H, pos, {n: n for n in top}, font_size=8, ax=ax)

    xs = np.array([pos[n][0] for n in H.nodes()])
    ys = np.array([pos[n][1] for n in H.nodes()])
    ax.set_xlim(np.percentile(xs, [2, 96]))
    ax.set_ylim(np.percentile(ys, [5, 94]))

    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, label="log(bounty)")
    ax.set_title(f"Character network - core + bounty characters ({H.number_of_nodes()} characters)")
    _keep_panel_background(ax)

    if show:
        plt.show()
    return H