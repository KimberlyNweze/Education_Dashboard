"""
visualisations.py
Source for all chart functions used in the LLM dashboard project.
Imports of this modules are used, rather than copying the fucntions multiple times.

Usage:
    from visualisations import (
        plot_timeline, plot_participation, plot_network,
        plot_stacked_bar, plot_heatmap, plot_episode_dashboard,
        render_visualization, COLORS, gini, to_seconds
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from collections import Counter
import networkx as nx
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Helper Functions 
def to_seconds(t):
    """Convert HH:MM:SS string to total seconds."""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def gini(values):
    """Participation equity score. 0 = equal, 1 = one person dominates."""
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * arr).sum()) / (n * arr.sum()) - (n + 1) / n


#Map LLM output strings to visualisation keys 

VIZ_KEY_MAP = {
    "timeline" : "timeline",
    "participation" : "participation",
    "participation chart": "participation",
    "network": "network",
    "network graph" : "network",
    "stacked bar" : "stacked_bar",
    "stacked" : "stacked_bar",
    "heatmap" : "heatmap",
}

def parse_viz_key(llm_string):
    """Normalise whatever the LLM returned to a key."""
    return VIZ_KEY_MAP.get(llm_string.lower().strip(), "timeline")


#Colour palette (for consistency across plots )
COLORS = {
    "challenge": "#E05C5C",   # red (challenge)
    "regulation": "#4A90D9",   # blue (regulation)
    "teacher": "#7BC67E",   # green (teacher speaker bars)
    "background": "#F8F9FB",
    "grid": "#E5E8EE",
}

LABEL_MAP = {
    "C" : "Cognitive", "E" : "Emotional",
    "M" : "Metacognitive","T" : "Technical",
    "MC": "Monitoring/Control",
    "TA": "Task Analysis",
    "RA": "Reflection/Adaptation",
    "none": "None",
}

def gini(values):
    """Participation equity. 0 = equal, 1 = one person dominates."""
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * arr).sum()) / (n * arr.sum()) - (n + 1) / n

print("Template and helpers ready ")



def plot_timeline(ep_df, session_id, ep_id, label="", ta_annotated=True, ax=None):
    """
    Line chart showing challenge and regulation density over time within an episode.

    Parameters
    ----------
    ep_df: pd.DataFrame — utterance rows for one episode
    session_id: int
    ep_id: int
    label: str — episode label for the title (e.g. 'HIGH')
    ta_annotated : bool  — whether TA column is reliable for this session
    ax: matplotlib Axes

    Returns
    -------
    fig, ax
    """
    ep_df = ep_df.copy().sort_values("start_sec").reset_index(drop=True)

    # Time from episode start in minutes
    t0 = ep_df["start_sec"].min()
    ep_df["minutes"] = (ep_df["start_sec"] - t0) / 60

    # Binary presence per utterance
    ep_df["challenge_any"]= ep_df[["C","E","M","T"]].any(axis=1).astype(float)
    reg_cols = ["MC","TA","RA"] if ta_annotated else ["MC","RA"]
    ep_df["regulation_any"]  = ep_df[reg_cols].any(axis=1).astype(float)

    # Rolling density, window is scales with episode length
    window = max(3, len(ep_df) // 5)
    ep_df["chal_density"] = ep_df["challenge_any"].rolling(window, min_periods=1, center=True).mean()
    ep_df["reg_density"]  = ep_df["regulation_any"].rolling(window, min_periods=1, center=True).mean()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLORS["background"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)

    # Shaded areas
    ax.fill_between(ep_df["minutes"], ep_df["chal_density"],
                    alpha=0.15, color=COLORS["challenge"])
    ax.fill_between(ep_df["minutes"], ep_df["reg_density"],
                    alpha=0.15, color=COLORS["regulation"])

    # Lines
    ax.plot(ep_df["minutes"], ep_df["chal_density"],color=COLORS["challenge"], linewidth=2.2,label="Challenge density", zorder=3)
    ax.plot(ep_df["minutes"], ep_df["reg_density"],color=COLORS["regulation"], linewidth=2.2,label="Regulation density", zorder=3)

    # Tick marks at each challenge utterance along x-axis
    chal_times = ep_df.loc[ep_df["challenge_any"] == 1, "minutes"]
    ax.scatter(chal_times, [0.02] * len(chal_times),
               marker="|", color=COLORS["challenge"],
               s=60, zorder=4, linewidth=1.5)

    ax.set_xlabel("Time into episode (minutes)", fontsize=10)
    ax.set_ylabel("Density (rolling avg)", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    ta_note = "" if ta_annotated else "  [TA not annotated]"
    ax.set_title(
        f"Timeline — {label} regulation  |  Session {session_id}, Episode {ep_id}{ta_note}",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    if standalone:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return fig, ax

print("plot_timeline() defined ")



def plot_participation(ep_df, session_id, ep_id, label="", ax=None):
    """
    Horizontal bar chart of speaking time per speaker with Gini annotation.

    Parameters
    ----------
    ep_df: pd.DataFrame — utterance rows for one episode
    session_id : int
    ep_id: int
    label: str
    ax: matplotlib Axes

    Returns
    -------
    fig, ax
    """
    ep_df = ep_df.copy()

    speak_time = ep_df.groupby("speaker")["duration"].sum().sort_values(ascending=True)
    speak_pct  = speak_time / speak_time.sum() * 100
    gini_score = gini(speak_time.values)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, max(3, len(speak_time) * 0.7)))
    else:
        fig = ax.get_figure()

    ax.set_facecolor(COLORS["background"])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, zorder=0)

    # Teachers gets a different colour from students
    bar_colors = [
        COLORS["teacher"] if str(spk).startswith("T0") else COLORS["regulation"]
        for spk in speak_pct.index
    ]

    bars = ax.barh(speak_pct.index, speak_pct.values,
                   color=bar_colors, edgecolor="white",
                   linewidth=0.8, height=0.55, zorder=3)

    # Percentage labels on each bar
    for bar, val in zip(bars, speak_pct.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    # Gini annotation (colour reflects equity level)
    if gini_score < 0.3:
        equity_label, gini_color = "High equity","#4CAF50"
    elif gini_score < 0.5:
        equity_label, gini_color = "Moderate equity","#FF9800"
    else:
        equity_label, gini_color = "Low equity","#F44336"

    ax.text(0.98, 0.05,
            f"Gini = {gini_score:.3f}\n{equity_label}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=gini_color,
                      alpha=0.15, edgecolor=gini_color))

    # Legend
    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS["regulation"], label="Student"),
            mpatches.Patch(color=COLORS["teacher"],    label="Teacher"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.9,
    )
    # Prevent the legend from getting clipped 
    plt.tight_layout()
    if standalone:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return fig, ax

print("plot_participation() defined ")



def plot_episode_dashboard(ep_df, session_id, ep_id, label="", ta_annotated=True):
    """
    Full episode dashboard: Timeline (top) + Participation (bottom-left)
    + regulatory summary panel (bottom-right).
    """
    fig = plt.figure(figsize=(13, 7), facecolor=COLORS["background"])
    gs  = GridSpec(2, 2, figure=fig, height_ratios=[1.4, 1], hspace=0.45, wspace=0.35)

    ax_time= fig.add_subplot(gs[0, :])   # full-width top row
    ax_part= fig.add_subplot(gs[1, 0])   # bottom left
    ax_stats= fig.add_subplot(gs[1, 1])   # bottom right with text summary

    plot_timeline(ep_df, session_id, ep_id,label=label, ta_annotated=ta_annotated, ax=ax_time)
    plot_participation(ep_df, session_id, ep_id, label=label, ax=ax_part)

    # Text summary panel
    ax_stats.set_facecolor(COLORS["background"])
    ax_stats.axis("off")

    reg_cols= ["MC","TA","RA"] if ta_annotated else ["MC","RA"]
    chal_rate= ep_df[["C","E","M","T"]].any(axis=1).mean()
    reg_rate= ep_df[reg_cols].any(axis=1).mean()
    duration= (ep_df["end_sec"].max() - ep_df["start_sec"].min()) / 60
    speak_time = ep_df.groupby("speaker")["duration"].sum()
    gini_score = gini(speak_time.values)

    dom_reg  = ep_df[reg_cols].sum().idxmax()  if ep_df[reg_cols].sum().sum() > 0 else "none"
    dom_chal = ep_df[["C","E","M","T"]].sum().idxmax()   if ep_df[["C","E","M","T"]].sum().sum() > 0 else "none"

    lines = [
        ("Episode summary", True),
        (f"Duration      {duration:.1f} min",           False),
        (f"Turns         {len(ep_df)}",                 False),
        (f"Speakers      {ep_df['speaker'].nunique()}", False),
        ("", False),
        ("Regulation", True),
        (f"  Rate        {reg_rate:.0%}",               False),
        (f"  Dominant    {LABEL_MAP.get(dom_reg, dom_reg)}", False),
        ("", False),
        ("Challenge", True),
        (f"  Rate        {chal_rate:.0%}",              False),
        (f"  Dominant    {LABEL_MAP.get(dom_chal, dom_chal)}", False),
        ("", False),
        ("Participation", True),
        (f"  Gini        {gini_score:.3f}",             False),
    ]

    y = 0.97
    for text, bold in lines:
        ax_stats.text(
            0.05, y, text,
            transform=ax_stats.transAxes,
            fontsize=10 if bold else 9.5,
            fontweight="bold" if bold else "normal",
            color="#333" if bold else "#555",
            fontfamily="monospace", va="top"
        )
        y -= 0.07

    ta_note = "" if ta_annotated else "\n TA not annotated for this session"
    fig.suptitle(
        f"Episode Dashboard — {label} regulation  |  "
        f"Session {session_id}, Episode {ep_id}{ta_note}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.show()
    return fig

print("plot_episode_dashboard() defined")



def plot_network(ep_df, session_id, ep_id, label="", ax=None):
    ep_df = ep_df.copy().sort_values("start_sec").reset_index(drop=True)
    speakers = ep_df["speaker"].dropna()
    transitions = Counter(zip(speakers.iloc[:-1], speakers.iloc[1:]))
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()
    ax.set_facecolor(COLORS["background"])
    if len(transitions) == 0:
        ax.text(0.5, 0.5, "Not enough turns", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(f"Network - {label} | Session {session_id}, Ep {ep_id}",
                     fontsize=11, fontweight="bold")
        return (fig, ax)
    G = nx.DiGraph()
    G.add_nodes_from(ep_df["speaker"].dropna().unique().tolist())
    for (src, tgt), freq in transitions.items():
        G.add_edge(src, tgt, weight=freq)
    # Spring layout
    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_colors = [
        COLORS["teacher"] if str(n).startswith("T0") else COLORS["regulation"]
        for n in G.nodes()
    ]
    # Normalise widths(all edges on the same scale)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    mn, mx  = min(weights), max(weights)
    def nw(w):
        return 1.5 if mx == mn else 1.5 + ((w - mn) / (mx - mn)) * 4.5
    edge_widths = [nw(w) for w in weights]
    # Draw nodes big enough for labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1600, edgecolors="white", linewidths=2)
    labels = {n: str(n)[:11] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7.5, font_color="white", font_weight="bold")
    # Edges one at a time so widths apply correctly
    for (u, v), lw in zip(G.edges(), edge_widths):
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)], width=lw,
            alpha=0.75, edge_color="#444444",
            arrows=True, arrowsize=20,
            connectionstyle="arc3,rad=0.2",
            min_source_margin=30, min_target_margin=30,
        )
    # Edge labels with white background so they never overlap arrows
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={e: G[e[0]][e[1]]["weight"] for e in G.edges()},
        ax=ax, font_size=8, label_pos=0.35,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none")
    )
    ax.legend(handles=[
        mpatches.Patch(color=COLORS["regulation"], label="Student"),
        mpatches.Patch(color=COLORS["teacher"],    label="Teacher"),
    ], loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Network - {label} regulation  |  Session {session_id}, Episode {ep_id}",
        fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")
    if standalone:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return fig, ax

print("plot_network()")


def plot_stacked_bar(ep_df, session_id, ep_id, label="",
                     ta_annotated=True, n_segments=5, ax=None):
    """
    Stacked bar with mutually exclusive categories so bars always sum to 100%.
    Categories: Challenge-only | Regulation-only | Both | Neither
    """
    ep_df = ep_df.copy().sort_values("start_sec").reset_index(drop=True)
    n = len(ep_df)
    n_segments = min(n_segments, max(1, n // 2))
    ep_df["segment"] = pd.cut(ep_df.index, bins=n_segments,
                               labels=[f"Seg {i+1}" for i in range(n_segments)])
    reg_cols = ["MC","TA","RA"] if ta_annotated else ["MC","RA"]
    ep_df["has_chal"] = ep_df[["C","E","M","T"]].any(axis=1)
    ep_df["has_reg"]= ep_df[reg_cols].any(axis=1)
    # Mutually exclusive (sums up to 1)
    ep_df["chal_only"] = (ep_df["has_chal"] & ~ep_df["has_reg"]).astype(float)
    ep_df["reg_only"]= (~ep_df["has_chal"] &  ep_df["has_reg"]).astype(float)
    ep_df["both"]= (ep_df["has_chal"] &  ep_df["has_reg"]).astype(float)
    ep_df["neither"]= (~ep_df["has_chal"] & ~ep_df["has_reg"]).astype(float)
    seg = ep_df.groupby("segment", observed=True)[
        ["chal_only","reg_only","both","neither"]].mean()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()
    ax.set_facecolor(COLORS["background"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, zorder=0)
    x      = np.arange(len(seg))
    width  = 0.55
    bottom = np.zeros(len(seg))
    layers = [
        ("chal_only", COLORS["challenge"],  "Challenge only"),
        ("reg_only",  COLORS["regulation"], "Regulation only"),
        ("both", "#9B59B6", "Both"),
        ("neither", "#CCCCCC", "Neither"),
    ]
    for col, color, lbl in layers:
        vals = seg[col].values
        ax.bar(x, vals, width, bottom=bottom, color=color, label=lbl,
               zorder=3, edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.08:
                ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(seg.index, fontsize=9)
    ax.set_xlabel("Episode segment", fontsize=10)
    ax.set_ylabel("Proportion of utterances", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ta_note = "" if ta_annotated else "  [TA not annotated]"
    ax.set_title(
        f"Stacked Bar - {label} regulation  |  Session {session_id}, Episode {ep_id}{ta_note}",
        fontsize=11, fontweight="bold", pad=10)
    if standalone:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return fig, ax

print("plot_stacked_bar() ")


def plot_heatmap(ep_df, session_id, ep_id, label="", ta_annotated=True, ax=None):
    """
    Heatmap of dialogue act frequency per speaker.
    Cell annotation: "X% (count/total)" which makes the denominator explicit
    so readers understand why two speakers can share the same raw count
    but show different percentages.
    """
    ep_df = ep_df.copy()
    ep_df["Question"] = ep_df["content"].str.contains(r"\?").astype(int)
    act_cols = ["Question","C","E","M","T","MC","RA"]
    act_labels = {
        "Question": "Question",
        "C":"Cognitive\nchallenge",
        "E":"Emotional\nchallenge",
        "M":"Metacog.\nchallenge",
        "T":"Technical\nchallenge",
        "MC":"Monitoring/\nControl",
        "RA":"Reflection/\nAdaptation",
    }
    if ta_annotated:
        ep_df["TA"] = ep_df["TA"].fillna(0)
        act_cols.insert(-1, "TA")
        act_labels["TA"] = "Task\nAnalysis"
    ep_df = ep_df.dropna(subset=["speaker"])
    heatmap_data= ep_df.groupby("speaker")[act_cols].sum()
    speaker_totals = ep_df.groupby("speaker").size()
    heatmap_norm= heatmap_data.div(speaker_totals, axis=0)
    standalone = ax is None
    n_rows, n_cols = heatmap_norm.shape
    if standalone:
        fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), max(3, n_rows * 0.9)))
    else:
        fig = ax.get_figure()
    ax.set_facecolor(COLORS["background"])
    im = ax.imshow(heatmap_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([act_labels[c] for c in act_cols], fontsize=8, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(heatmap_norm.index.tolist(), fontsize=9)
    # Annotation: "pct (raw/total)" — explicit denominator removes ambiguity
    for i, speaker in enumerate(heatmap_norm.index):
        total = int(speaker_totals[speaker])
        for j, col in enumerate(act_cols):
            val = heatmap_norm.values[i, j]
            raw = int(heatmap_data.values[i, j])
            text_color = "white" if val > 0.55 else "#333333"
            ax.text(j, i, f"{val:.0%}\n({raw}/{total})",
                    ha="center", va="center", fontsize=7.5, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Proportion of speaker\'s turns", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ta_note = "" if ta_annotated else "  [TA not annotated]"
    ax.set_title(
        f"Heatmap - {label} regulation  |  Session {session_id}, Episode {ep_id}{ta_note}",
        fontsize=11, fontweight="bold", pad=12)
    if standalone:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return fig, ax

print("plot_heatmap() ")



def render_visualization(viz_key, ep_df, session_id, ep_id,
                         label="", ta_annotated=True):
    """
    Render the correct chart given a canonical viz_key and episode data.

    Parameters
    ----------
    viz_key : str  — one of: timeline, participation, network, stacked_bar, heatmap
    ep_df: pd.DataFrame — raw utterance rows for the episode
    session_id: int
    ep_id: int
    label: str  — episode label for chart title (e.g. HIGH)
    ta_annotated: bool — whether TA column is reliable for this session

    Returns
    -------
    fig, ax
    """
    # Ensure time columns are present
    if "start_sec" not in ep_df.columns:
        ep_df = ep_df.copy()
        ep_df["start_sec"] = ep_df["start"].apply(to_seconds)
        ep_df["end_sec"]   = ep_df["end"].apply(to_seconds)
        ep_df["duration"]  = ep_df["end_sec"] - ep_df["start_sec"]
        ep_df["word_count"]= ep_df["content"].str.split().str.len()

    dispatch = {
        "timeline": lambda: plot_timeline(ep_df, session_id, ep_id, label, ta_annotated),
        "participation" : lambda: plot_participation(ep_df, session_id, ep_id, label),
        "network": lambda: plot_network(ep_df, session_id, ep_id, label),
        "stacked_bar": lambda: plot_stacked_bar(ep_df, session_id, ep_id, label, ta_annotated),
        "heatmap": lambda: plot_heatmap(ep_df, session_id, ep_id, label, ta_annotated),
    }

    if viz_key not in dispatch:
        raise ValueError(f"Unknown viz_key: {viz_key!r}. Must be one of: {list(dispatch.keys())}")

    return dispatch[viz_key]()
