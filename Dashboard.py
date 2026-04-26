"""
dashboard.py  —  Collaborative Learning Analytics Dashboard
Requires: dataset.csv, episode_features.csv, visualisations.py, .env (contains API key)
"""

import os, io, json, time, warnings
import matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np, pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

from visualisations import (
    render_visualization, parse_viz_key,
    plot_timeline, COLORS, gini, to_seconds,
)

#Page configeration 
st.set_page_config(
    page_title="Collaborative Learning Dashboard",layout="wide",
    initial_sidebar_state="expanded",
)
# Cascading Stlye Sheet (custom uniform look across the dashboard )
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; letter-spacing: -0.02em; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #e0e4f0 !important; }
[data-testid="metric-container"] {
    background: #f7f8fc; border: 1px solid #e4e8f2;
    border-radius: 10px; padding: 12px 16px;
}
.llm-panel {
    background: linear-gradient(135deg, #f0f4ff 0%, #f7f0ff 100%);
    border-left: 4px solid #4a6cf7; border-radius: 0 12px 12px 0;
    padding: 18px 22px; margin-top: 12px;
}
.llm-panel h4 { font-family: 'DM Serif Display', serif; color: #1a1f3c; margin: 0 0 8px 0; font-size: 1rem; }
.llm-panel p  { color: #3a3f5c; margin: 0; font-size: 0.92rem; line-height: 1.6; }
.reasoning-pill {
    background: #fff8e6; border: 1px solid #f5c842; border-radius: 8px;
    padding: 10px 16px; margin-top: 10px; font-size: 0.85rem; color: #5c4a00;
}
.deep-reasoning {
    background: #f0fff4; border: 1px solid #68d391; border-radius: 8px;
    padding: 14px 18px; margin-top: 10px; font-size: 0.88rem; color: #1a4731;
    line-height: 1.7;
}
.viz-badge {
    display: inline-block; background: #4a6cf7; color: white !important;
    font-weight: 500; font-size: 0.78rem; padding: 4px 12px;
    border-radius: 20px; margin-bottom: 10px;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.section-rule { border: none; border-top: 2px solid #e4e8f2; margin: 28px 0 20px 0; }
.compare-header {
    background: #0f1117; color: #e0e4f0 !important; border-radius: 8px;
    padding: 8px 16px; font-size: 0.85rem; font-weight: 500;
    margin-bottom: 8px; text-align: center;
}
.flag-card {
    background: #fff5f5; border-left: 3px solid #e05c5c; border-radius: 0 8px 8px 0;
    padding: 8px 14px; margin: 4px 0; font-size: 0.85rem; color: #4a1010;
}
.flag-card-yellow {
    background: #fffbf0; border-left: 3px solid #f5c842; border-radius: 0 8px 8px 0;
    padding: 8px 14px; margin: 4px 0; font-size: 0.85rem; color: #4a3800;
}
.ep-chip {
    display: inline-block; border-radius: 6px; padding: 3px 10px;
    font-size: 0.75rem; font-weight: 500; margin: 2px; cursor: default;
}
.framework-box {
    background: #f8f9ff; border: 1px solid #dde3ff; border-radius: 10px;
    padding: 16px 20px; margin-top: 8px; font-size: 0.85rem; color: #2a2f5c;
    line-height: 1.8;
}
.framework-box h5 { color: #4a6cf7; margin: 0 0 6px 0; font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)



# Load transcript + episode feature datasets and generate timing metrics
@st.cache_data
def load_data():
    df= pd.read_csv("dataset.csv")
    ep= pd.read_csv("episode_features.csv")
    df["start_sec"] = df["start"].apply(to_seconds)
    df["end_sec"]= df["end"].apply(to_seconds)
    df["duration"]= df["end_sec"] - df["start_sec"]
    df["word_count"] = df["content"].str.split().str.len()
    ta_unknown = df.groupby("session")["TA"].apply(lambda x: x.isnull().all())
    return df, ep, ta_unknown[ta_unknown].index.tolist()

df, ep_features, TA_UNKNOWN = load_data()


# Groq LLM
# Initialise and cache the Groq API client using the GROQ_API_KEY stored in the .env file (returns None if no key is available)
@st.cache_resource
def get_groq_client():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

groq_client = get_groq_client()
MODEL = "llama-3.3-70b-versatile"

PEDAGOGICAL_FRAMEWORK = """
You are an expert learning analytics assistant helping teachers understand
collaborative learning dynamics in small-group discussions.

VISUALISATION SELECTION CRITERIA

1. TIMELINE — ONLY when: regulation_rate > 0.8 AND challenge_rate == 0.0 AND total_duration_sec > 120
2. PARTICIPATION CHART — ONLY when: gini_coefficient strictly greater than 0.6 AND teacher speaker present
3. NETWORK GRAPH — ONLY when: n_speakers >= 3 AND n_turns >= 8
4. STACKED BAR — ONLY when: challenge_rate > 0.1 AND regulation_rate > 0.1 AND n_turns >= 6
5. HEATMAP — ONLY when: n_speakers >= 3 AND gini_coefficient < 0.6 AND (challenge_rate > 0 OR variation in speaker labels)

STRICT DECISION PRIORITY — apply in order, first match wins:
1. gini_coefficient > 0.6 AND teacher present → PARTICIPATION CHART
2. challenge_rate > 0.1 AND regulation_rate > 0.1 → STACKED BAR
3. regulation_rate > 0.8 AND challenge_rate == 0.0 AND total_duration_sec > 120 → TIMELINE
4. n_speakers >= 3 AND n_turns >= 8 AND gini_coefficient < 0.6 → HEATMAP
5. n_speakers >= 3 AND n_turns >= 8 → NETWORK GRAPH
6. Default → TIMELINE

CRITICAL RULES:
- Apply thresholds EXACTLY. 0.592 is NOT greater than 0.6. Never round thresholds.
- Only cite numbers that appear verbatim in EPISODE FEATURES. Never approximate.
"""

VIZ_FRAMEWORK_NOTES = {
    "timeline": {
        "title": "Timeline",
        "when": (
            "Timeline is used when the episode has high regulation (>80% of turns) with little or no challenge present, "
            "AND the episode is long enough for temporal progression to be meaningful (>120 seconds). "
            "In these episodes, the most informative question for a teacher is not whether regulation is happening "
            "(this is already known) but whether it is sustained evenly across the episode or whether it fluctuates. "
            "A group that regulates consistently throughout a task is demonstrating qualitatively different behaviour "
            "from a group that bursts into regulation only at certain moments. "
            "The Timeline makes this visible in a way that a single rate statistic cannot."
        ),
        "theory": (
            "Grounded in Socially Shared Regulation of Learning (SSRL) theory (Järvelä & Hadwin, 2013; Hadwin, Järvelä, & Miller, 2018), "
            "which distinguishes between regulation that is sustained and co-constructed across an episode versus regulation "
            "that is reactive and episodic. Järvelä et al. (2019) emphasise the importance of capturing the sequential and "
            "temporal characteristics of regulation, arguing that point-in-time or aggregate measures miss the dynamic nature "
            "of how groups regulate their learning. Martinez-Maldonado et al. (2019) demonstrated that timeline "
            "visualisations in teacher-facing dashboards helped teachers make faster and better-informed decisions about "
            "when to intervene with a group."
        ),
        "look_for": (
            "A flat blue line across the whole episode indicates consistent monitoring and control — the group is "
            "regulating steadily, which is generally positive and suggests task engagement. "
            "A blue line that drops off sharply in the middle or at the end suggests regulation collapsed — the group "
            "may have lost focus or drifted off-task. "
            "A red spike (challenge) followed closely by a blue rise (regulation) is the ideal pattern: "
            "it means the group encountered difficulty and self-corrected. "
            "A red spike with no subsequent blue is the most concerning signal — this is where teacher intervention is most needed."
        ),
        "references": (
            '<a href="https://doi.org/10.1891/1945-8959.12.3.267" target="_blank">Järvelä & Hadwin (2013)</a> · '
            '<a href="https://psycnet.apa.org/record/2017-45259-006" target="_blank">Hadwin et al (2018)</a> · '
            '<a href="https://doi.org/10.1007/s11412-019-09313-2" target="_blank">Järvelä et al. (2019)</a> · '
            '<a href="https://www.researchgate.net/publication/337301299" target="_blank">Martinez-Maldonado et al. (2019)</a>'
        ),
    },
    "participation": {
        "title": "Participation Chart",
        "when": (
            "Participation Chart was selected when the Gini coefficient exceeds 0.6 and a teacher is present in the episode. "
            "A Gini above 0.6 indicates that speaking time is very unequally distributed, with one person is carrying the "
            "conversation while others are largely silent. This is particularly important to flag when the dominant "
            "speaker is the teacher, because this pattern suggests the teacher may be over-scaffolding. "
            "If the dominant speaker is a student, it may indicate a status problem within the group."
        ),
        "theory": (
            "Rooted in research on participation equity in collaborative learning. Strauß & Rummel (2021) used the "
            "Gini coefficient specifically as a threshold measure for triggering adaptive prompts when participation "
            "became unequal (Gini ≥ 0.5), demonstrating its validity as an actionable signal. "
            "Janssen et al. (2012) established that unequal participation in CSCL is associated with lower-quality "
            "knowledge co-construction outcomes. Van Leeuwen et al. (2019) found that teachers using dashboards were "
            "better able to detect participation problems when speaking time was visualised directly."
        ),
        "look_for": (
            "A teacher bar substantially longer than all student bars is the primary signal of over-scaffolding. "
            "If the teacher accounts for more than 60% of speaking time, students are likely not building their own "
            "regulatory capacity. Among student bars, near-zero bars indicate participants effectively disengaged "
            "from the conversation. The Gini annotation box (green/amber/red) gives an at-a-glance equity verdict."
        ),
        "references": (
            '<a href="https://doi.org/10.1007/s11412-021-09340-y" target="_blank">Strauß & Rummel (2021)</a> · '
            '<a href="https://doi.org/10.1007/s11412-019-09299-x" target="_blank">Van Leeuwen et al. (2019)</a> · '
            '<a href="https://doi.org/10.1016/j.compedu.2006.01.004" target="_blank">Janssen et al. (2012)</a>'
        ),
    },
    "network": {
        "title": "Network Graph",
        "when": (
            "Network Graph was selected when an episode has 3 or more speakers and at least 8 turns, "
            "and when the Gini coefficient is below 0.6. "
            "The key question this visualisation answers is not how much each person talks, but who responds "
            "to whom — two episodes can have identical Gini scores but completely different conversation structures. "
            "The network graph makes structural differences visible in a way that bar charts cannot."
        ),
        "theory": (
            "Grounded in Social Network Analysis (SNA) applied to CSCL. "
            "Rienties et al. (2018) used SNA to monitor online collaborative learning and found that teacher-centred "
            "interaction patterns were associated with lower collaborative quality, and that SNA-informed interventions "
            "significantly increased student-to-student interaction. "
            "The distinction between IRE patterns and dialogic patterns is well established in classroom discourse "
            "research (Mercer, 2000) and is directly readable from the network structure."
        ),
        "look_for": (
            "All arrows pointing to or from a single teacher node signals an Initiation-Response-Evaluation (IRE) pattern — "
            "students are not building on each other's ideas, only responding to the teacher. "
            "Dense bidirectional arrows between student nodes indicate genuine peer dialogue. "
            "A student node with arrows going out but none coming in is an isolated contributor — "
            "they are speaking but nobody is responding. "
            "Edge numbers show how many times one speaker was immediately followed by another in the transcript - "
            "thicker arrows indicate more frequent transitions and larger nodes indicate more total turns taken by that speaker."
            "Note that these counts reflect turn frequency, not speaking time or content quality,"
            "so a high number between two speakers means they exchanged turns often but says nothing about the substance of those exchanges."
        ),
        "references": (
            '<a href="https://doi.org/10.1371/journal.pone.0194777" target="_blank">Rienties et al. (2018)</a> · '
            '<a href="https://doi.org/10.1016/j.caeo.2022.100073" target="_blank">Kaliisa et al. (2022)</a> · '
            '<a href="https://doi.org/10.4324/9780203464984" target="_blank">Mercer (2000)</a>'
        ),
    },
    "stacked_bar": {
        "title": "Stacked Bar",
        "when": (
            "Stacked Bar was selected when both challenge and regulation are present at meaningful levels "
            "(challenge rate > 10% and regulation rate > 10%), and the episode has at least 6 turns. "
            "When both signals are present, the most pedagogically important question is how their balance "
            "shifts across the episode — a dynamic invisible in a single rate number."
        ),
        "theory": (
            "Grounded in the SSRL framework (Hadwin, Järvelä, & Miller, 2011, 2018), which describes regulation "
            "as occurring in response to challenge. Challenge episodes are identified as the critical unit of analysis "
            "for understanding when and how groups regulate (Järvelä et al., 2013). "
            "The Suraworachet et al. (2024) dataset specifically models challenge moments as distinct episodes "
            "where regulatory response can be observed."
        ),
        "look_for": (
            "Challenge (red) dominating early segments followed by regulation (blue) later likely means the group self-regulated. "
            "The purple 'Both' category is most analytically interesting: co-occurring challenge and regulation "
            "indicates the group is simultaneously struggling and self-correcting. "
            "Challenge persisting across all segments without regulation = group is stuck, intervention needed. "
            "Grey 'Neither' segments = neutral content discussion."
        ),
        "references": (
            '<a href="https://www.researchgate.net/publication/313369294" target="_blank">Hadwin, Järvelä, & Miller (2011)</a> · '
            '<a href="https://doi.org/10.1891/1945-8959.12.3.267" target="_blank">Järvelä & Hadwin (2013)</a> · '
            '<a href="https://doi.org/10.1145/3636555.3636905" target="_blank">Suraworachet et al. (2024)</a>'
        ),
    },
    "heatmap": {
        "title": "Heatmap",
        "when": (
            "Heatmap was selected when there are 3 or more speakers and the Gini coefficient is below 0.6, "
            "but you want to understand not just how much each speaker contributes but what kind of contributions. "
            "Two students can have identical speaking time but one may raise all the challenges while the other "
            "does all the monitoring — a critical distinction for understanding regulatory load distribution."
        ),
        "theory": (
            "Grounded in research on dialogic roles in collaborative learning (Mercer, 2000). "
            "In the SSRL framework (Hadwin et al., 2011), regulation is most effective when genuinely shared "
            "across the group. A heatmap showing one student performing all monitoring is diagnostic of "
            "co-regulation rather than shared regulation — a less robust form of group self-management. "
            "Van Leeuwen et al. (2019) and Kaliisa et al. (2022) both found that discourse-level visualisations "
            "showing the nature of contributions were rated by teachers as more actionable than participation-level views alone."
        ),
        "look_for": (
            "Dark cells in Monitoring/Control for only one student = that student carries the regulatory load. "
            "A row of near-zero cells across all categories = that speaker's turns are short acknowledgements, not substantive contributions. "
            "Dark challenge columns concentrated in one speaker = that person is the source of difficulty. "
            "Cell annotations show percentage and raw count/total (e.g. 80% (4/5)) — "
            "interpret high percentages from very few turns (e.g. 100% (1/1)) cautiously."
        ),
        "references": (
            '<a href="https://doi.org/10.4324/9780203464984" target="_blank">Mercer (2000)</a> · '
            '<a href="https://www.researchgate.net/publication/313369294" target="_blank">Hadwin et al. (2011)</a> · '
            '<a href="https://doi.org/10.1007/s11412-019-09299-x" target="_blank">Van Leeuwen et al. (2019)</a> · '
            '<a href="https://doi.org/10.1016/j.caeo.2022.100073" target="_blank">Kaliisa et al. (2022)</a>'
        ),
    },
}

PROMPT_TEMPLATE = """{framework}

TASK: Respond ONLY with valid JSON (no markdown fences):
{{
  "selected_visualisation": "<Timeline|Participation Chart|Network Graph|Stacked Bar|Heatmap>",
  "reason": "<1-2 sentences citing specific feature values>",
  "teacher_explanation": "<2-3 plain-language sentences ending with one actionable suggestion>"
}}

{features}"""

DEEP_REASONING_PROMPT = """You previously selected {viz} for this episode.

A teacher is asking: "Why did you choose this visualisation over the others?"

Episode features for context:
{features}

Please give a detailed explanation (4-6 sentences) that:
1. Explains which specific feature values triggered this choice
2. Explains why the other 4 visualisations would be less informative for this episode
3. Connects to the pedagogical purpose — what will the teacher actually learn from this viz?
4. Only mention one of the five possible visualisations when talking about alternatives: Timeline, Participation Chart, Network Graph, Stacked Bar, or Heatmap.

Use plain language. No jargon. No JSON — just a clear paragraph."""


# Helper Functions 
def fmt_dur(s):
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s" if m else f"{sec}s"

def build_features_text(session_id, ep_id, ep_df_raw):
    r = ep_features[(ep_features["session"]==session_id)&(ep_features["ep"]==ep_id)].iloc[0]
    speak = ep_df_raw.groupby("speaker")["duration"].sum().sort_values(ascending=False)
    total = speak.sum()
    spk_lines = "\n".join(
        f"    {s} ({'teacher' if str(s).startswith('T0') else 'student'}): {v}s ({v/total*100:.0f}%)"
        for s,v in speak.items()
    )
    ta_note = "TA annotated." if session_id not in TA_UNKNOWN else "TA NOT annotated."
    return (f"EPISODE FEATURES\nSession:{int(r.session)} Episode:{int(r.ep)} {ta_note}\n"
            f"TEMPORAL duration={r.total_duration_sec}s turns={int(r.n_turns)} "
            f"avg_turn={r.avg_turn_length_sec}s latency={r.avg_response_latency}s\n"
            f"PARTICIPATION speakers={int(r.n_speakers)} gini={r.gini_coefficient}\n{spk_lines}\n"
            f"CONVERSATIONAL questions={int(r.question_count)} q_density={r.question_density} "
            f"avg_utt={r.avg_utt_length_words}w\n"
            f"CHALLENGE any={r.challenge_rate} C={r.C_rate} E={r.E_rate} M={r.M_rate} "
            f"T={r.T_rate} dominant={r.dominant_challenge}\n"
            f"REGULATION any={r.regulation_rate} MC={r.MC_rate} TA={r.TA_rate} "
            f"RA={r.RA_rate} dominant={r.dominant_regulation}")

#Sends episode features to Groq LLM and validates returned JSON response
def call_llm(session_id, ep_id, ep_df_raw):
    if groq_client is None:
        return {"selected_visualisation":"Timeline","reason":"No API key.",
                "teacher_explanation":"Configure GROQ_API_KEY in .env.","status":"no_api_key"}
    prompt = PROMPT_TEMPLATE.format(
        framework=PEDAGOGICAL_FRAMEWORK,
        features=build_features_text(session_id, ep_id, ep_df_raw)
    )
    VALID = {"timeline","participation chart","network graph","stacked bar","heatmap"}
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system","content":"Respond with valid JSON only. No markdown fences."},
                    {"role":"user","content":prompt}
                ],
                max_tokens=512, temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw[4:].strip() if raw.startswith("json") else raw.strip()
            result = json.loads(raw)
            missing = {"selected_visualisation","reason","teacher_explanation"} - set(result.keys())
            if missing: raise ValueError(f"Missing keys: {missing}")
            if result["selected_visualisation"].lower() not in VALID:
                raise ValueError(f"Unknown viz: {result['selected_visualisation']}")
            result["status"]    = "ok"
            result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            return result
        except Exception as e:
            if attempt < 2: time.sleep(1)
            else:
                return {"selected_visualisation":"Timeline","reason":f"LLM error: {e}",
                        "teacher_explanation":"Could not generate explanation.","status":"error"}

def call_deep_reasoning(viz_name, features_text):
    if groq_client is None:
        return "API key not configured."
    prompt = DEEP_REASONING_PROMPT.format(viz=viz_name, features=features_text)
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are a learning analytics expert. Respond in plain prose, no JSON."},
                {"role":"user","content":prompt}
            ],
            max_tokens=400, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate deep explanation: {e}"
# Stores model outputs locally for reproducibility and evaluation
def log_llm_result(session_id, ep_id, result, features_text, deep_reasoning=None):
    """Append LLM call to llm_log.json for reproducibility and evaluation."""
    log_path = "llm_log.json"
    entry = {
        "timestamp" : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session": session_id,
        "ep": ep_id,
        "selected_visualisation": result.get("selected_visualisation"),
        "reason": result.get("reason"),
        "teacher_explanation": result.get("teacher_explanation"),
        "status": result.get("status"),
        "features_sent" : features_text,
        "deep_reasoning": deep_reasoning,
        "model" : MODEL,
    }
    try:
        existing = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                existing = json.load(f)
        existing = [e for e in existing
                    if not (e["session"] == session_id and e["ep"] == ep_id)]
        existing.append(entry)
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass  # prevents logging from crashing the app

# Save each LLM decision to llm_log.json.
#Existing entries for the same episode are updated.
def export_session_csv(session_id):
    """Build CSV of all LLM results for a session. Returns bytes for download."""
    rows = []
    sess_ep = ep_features[ep_features["session"] == session_id]
    for _, row in sess_ep.iterrows():
        ep_id  = int(row["ep"])
        result = st.session_state.get(f"llm_{session_id}_{ep_id}", {})
        rows.append({
            "session": session_id,
            "ep": ep_id,
            "n_turns": int(row["n_turns"]),
            "n_speakers": int(row["n_speakers"]),
            "regulation_rate": row["regulation_rate"],
            "challenge_rate": row["challenge_rate"],
            "gini_coefficient": row["gini_coefficient"],
            "llm_run": bool(result),
            "selected_visualisation": result.get("selected_visualisation", ""),
            "reason": result.get("reason", ""),
            "teacher_explanation": result.get("teacher_explanation", ""),
            "status": result.get("status", ""),
            "timestamp": result.get("timestamp", ""),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


VIZ_ICONS= {"timeline":"📈","participation":"🎤","network":"🕸️","stacked_bar":"📊","heatmap":"🔥"}
VIZ_COLORS= {"timeline":"#4A90D9","participation":"#E05C5C",
               "network":"#8B7EC8","stacked_bar":"#9B59B6","heatmap":"#E8913A"}
VIZ_LABELS= {"timeline":"Timeline","participation":"Participation", "network":"Network","stacked_bar":"Stacked Bar","heatmap":"Heatmap"}

FLAG_RULES = [
    ("participation","🔴 Participation issue",
     lambda r: r["gini_coefficient"]>0.6,
     "Gini > 0.6 — one speaker dominates"),
    ("stacked_bar","🟠 Challenge + regulation co-occurring",
     lambda r: r["challenge_rate"]>0.1 and r["regulation_rate"]>0.1,
     "Both challenge and regulation elevated"),
    ("heatmap","🟡 Varied speaker roles",
     lambda r: r["n_speakers"]>=3 and r["challenge_rate"]>0 and r["gini_coefficient"]<0.6,
     "Multiple speakers with varied contributions"),
    ("timeline","🔵 Sustained regulation",
     lambda r: r["regulation_rate"]>0.8 and r["challenge_rate"]==0,
     "High regulation throughout, no challenge"),
]


#Sidebar Pannel 
with st.sidebar:
    st.markdown("## Learning Dashboard")
    st.markdown("<hr style='border-color:#1e2130;margin:8px 0 16px'>", unsafe_allow_html=True)

    # Switch to Episode Detail if navigating from a chip click
    if "nav_ep" in st.session_state:
        default_page_idx = 2
    else:
        default_page_idx = 0

    page = st.radio("Page", [
        "🏠  Session Overview",
        "📋  Session Summary",
        "🔍  Episode Detail",
    ], index=default_page_idx, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2130;margin:12px 0'>", unsafe_allow_html=True)

    sessions = sorted(df["session"].unique().tolist())

    # Respect chip-click navigation (open epsiode in anotehr tab)
    default_session = st.session_state.pop("nav_session", None)
    default_ep_nav= st.session_state.pop("nav_ep", None)

    session_index = sessions.index(default_session) if default_session in sessions else 0
    selected_session = st.selectbox(
        "Session", sessions,
        index=session_index,
        format_func=lambda x: f"Session {x}" + (" ⚠️" if x in TA_UNKNOWN else ""),
    )

    eps_in_session = sorted(
        ep_features[ep_features["session"]==selected_session]["ep"].unique().tolist()
    )
    ep_index = eps_in_session.index(default_ep_nav) if default_ep_nav in eps_in_session else 0
    selected_ep = st.selectbox(
        "Episode", eps_in_session,
        index=ep_index,
        format_func=lambda x: f"Episode {x}",
    )

    if selected_session in TA_UNKNOWN:
        st.warning("TA labels missing for this session.")

    st.markdown("<hr style='border-color:#1e2130;margin:12px 0'>", unsafe_allow_html=True)
    st.caption(f"Groq API: {'🟢 Connected' if groq_client else '🔴 No API key'}")
    st.caption("28 sessions · 882 episodes · 10,799 utterances")

    # Handle query-param chip navigation
    params = st.query_params
    if params.get("page") == "detail":
        try:
            st.session_state["nav_session"] = int(params.get("session", selected_session))
            st.session_state["nav_ep"]      = int(params.get("ep", selected_ep))
            st.query_params.clear()
            st.rerun()
        except (ValueError, KeyError):
            st.query_params.clear()



#  PAGE 1 — SESSION OVERVIEW

if "Overview" in page:
    st.markdown(f"# Session {selected_session}")
    sess_df = df[df["session"]==selected_session]
    sess_ep = ep_features[ep_features["session"]==selected_session]
    if selected_session in TA_UNKNOWN:
        st.warning(" TA (Task Analysis) not annotated for this session.")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Episodes",       len(sess_ep))
    c2.metric("Utterances",     f"{len(sess_df):,}")
    c3.metric("Total duration", fmt_dur(sess_df["duration"].sum()))
    c4.metric("Participants",   sess_df["speaker"].nunique())
    c5.metric("Avg regulation", f"{sess_ep['regulation_rate'].mean():.0%}")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### Episodes")

    def reg_b(r): return "🔴 High" if r>=0.7 else "🟠 Moderate" if r>=0.3 else "⚪ Low"
    def chal_b(r): return "🔴 High" if r>=0.3 else "🟠 Moderate" if r>=0.1 else "⚪ Low"

    rows = [{"Ep":int(r["ep"]),"Duration":fmt_dur(r["total_duration_sec"]),
             "Turns":int(r["n_turns"]),"Speakers":int(r["n_speakers"]),
             "Regulation":reg_b(r["regulation_rate"]),"Challenge":chal_b(r["challenge_rate"]),
             "Gini":f"{r['gini_coefficient']:.2f}",
             "Dom. Reg":r["dominant_regulation"],"Dom. Chal":r["dominant_challenge"]}
            for _,r in sess_ep.iterrows()]
    st.dataframe(pd.DataFrame(rows).set_index("Ep"), use_container_width=True, height=380)
    st.caption("Regulation & Challenge — 🔴 High (≥70%) · 🟠 Moderate (30–70%) · ⚪ Low (<30%) ")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### Regulation & Challenge by Episode")
    ch1, ch2 = st.columns(2)
    
# Create a styled bar chart used for session-overview comparisons 
    def ep_bar(ax, values, eps, color_fn, ylabel, title, legend_handles):
        ax.bar(eps, values, color=[color_fn(v) for v in values], edgecolor="white", linewidth=0.4)
        ax.set_facecolor("#F8F9FB"); ax.set_xlabel("Episode",fontsize=8)
        ax.set_ylabel(ylabel,fontsize=8); ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.tick_params(labelsize=7); ax.grid(axis="y",color="#E5E8EE",linewidth=0.6)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.legend(handles=legend_handles, fontsize=7, framealpha=0.8)
        ax.set_title(title, fontsize=9, pad=6)

    with ch1:
        fig, ax = plt.subplots(figsize=(5.5,3)); fig.patch.set_facecolor("#F8F9FB")
        ep_bar(ax, sess_ep["regulation_rate"].values, sess_ep["ep"].values,
               lambda v: "#4A90D9" if v>=0.7 else "#F5A623" if v>=0.3 else "#CCCCCC",
               "Regulation rate", "Regulation rate per episode",
               [mpatches.Patch(color="#4A90D9",label="High (≥70%)"),
                mpatches.Patch(color="#F5A623",label="Moderate"),
                mpatches.Patch(color="#CCCCCC",label="Low (<30%)")])
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with ch2:
        fig, ax = plt.subplots(figsize=(5.5,3)); fig.patch.set_facecolor("#F8F9FB")
        ep_bar(ax, sess_ep["challenge_rate"].values, sess_ep["ep"].values,
               lambda v: "#E05C5C" if v>=0.3 else "#E8913A" if v>=0.1 else "#CCCCCC",
               "Challenge rate", "Challenge rate per episode",
               [mpatches.Patch(color="#E05C5C",label="High (≥30%)"),
                mpatches.Patch(color="#E8913A",label="Moderate (10–30%)"),
                mpatches.Patch(color="#CCCCCC",label="Low (<10%)")])
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### Participation across session")
    speak = sess_df.groupby("speaker")["duration"].sum().sort_values(ascending=False).reset_index()
    speak["Share"]= (speak["duration"]/speak["duration"].sum()*100).round(1).astype(str)+"%"
    speak["Role"]= speak["speaker"].apply(lambda x: "Teacher" if str(x).startswith("T0") else "Student")
    speak = speak.rename(columns={"speaker":"Speaker","duration":"Speaking time (s)"})
    st.dataframe(speak[["Speaker","Role","Speaking time (s)","Share"]].set_index("Speaker"), use_container_width=True)


#  PAGE 2 — SESSION SUMMARY

elif "Summary" in page:
    st.markdown(f"# Session {selected_session} — Summary")
    sess_ep = ep_features[ep_features["session"]==selected_session].copy()
    sess_ep["too_small"] = (sess_ep["n_turns"] < 3) | (sess_ep["n_speakers"] < 2)

    if selected_session in TA_UNKNOWN:
        st.warning("TA labels not available for this session.")

    # Flagged Patterns from data exploration 
    st.markdown("### 🚩 Flagged Patterns")
    all_flags = []
    for viz_key, flag_label, condition, detail in FLAG_RULES:
        flagged = sess_ep[sess_ep.apply(condition, axis=1)]
        if len(flagged) > 0:
            ep_list = ", ".join([f"Ep {int(e)}" for e in flagged["ep"].tolist()])
            all_flags.append((flag_label, len(flagged), ep_list, detail))

    if all_flags:
        fc1, fc2 = st.columns(2)
        for i, (flag_label, count, ep_list, detail) in enumerate(all_flags):
            col = fc1 if i % 2 == 0 else fc2
            css_class = "flag-card" if "🔴" in flag_label else "flag-card-yellow"
            with col:
                st.markdown(f"""
<div class="{css_class}">
  <strong>{flag_label}</strong> — {count} episode(s)<br>
  <small>{detail}</small><br>
  <small style="opacity:0.7">{ep_list}</small>
</div>""", unsafe_allow_html=True)
    else:
        st.success("No significant patterns flagged for this session.")

    #LLM Visualisation by Episode 
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### LLM Visualisation by Episode")

    # Count how many episodes have LLM results
    llm_run_count = sum(
        1 for ep_id in sess_ep["ep"]
        if f"llm_{selected_session}_{int(ep_id)}" in st.session_state
    )
    total_valid = int((~sess_ep["too_small"]).sum())
    progress_pct = llm_run_count / total_valid if total_valid > 0 else 0

    st.info(
        f" **{llm_run_count} of {total_valid} valid episodes have LLM results** for this session. "
        "Go to **Episode Detail** and click ▶ Run LLM to populate more. "
        "Click any coloured chip to jump directly to that episode."
    )

    # Progress bar
    prog_col, dl_col = st.columns([3, 1])
    with prog_col:
        st.progress(progress_pct,
                    text=f"Session {selected_session} coverage: "
                         f"{llm_run_count}/{total_valid} episodes run "
                         f"({progress_pct*100:.0f}%)")
    with dl_col:
        csv_bytes = export_session_csv(selected_session)
        st.download_button(
            label="⬇️ Export CSV",
            data=csv_bytes,
            file_name=f"session_{selected_session}_llm_results.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download all LLM results for this session as a CSV",
        )

    sv1, sv2 = st.columns([2, 1])
    with sv1:
        chips_html = ""
        llm_viz_counts = {}

        for _, row in sess_ep.iterrows():
            ep_id     = int(row["ep"])
            too_small = row["too_small"]
            llm_result = st.session_state.get(f"llm_{selected_session}_{ep_id}", {})
            llm_viz    = parse_viz_key(llm_result.get("selected_visualisation", "")) if llm_result else None

            if too_small:
                # Dark grey — permanently disabled, not clickable (because not long enough)
                chips_html += (
                    f'<span class="ep-chip" '
                    f'style="background:#d1d5db;color:#6b7280;cursor:not-allowed;" '
                    f'title="Too small to visualise (< 3 turns or < 2 speakers)">'
                    f'Ep {ep_id} —</span>'
                )
            elif llm_viz:
                # Coloured — LLM result exists, clickable
                color = VIZ_COLORS.get(llm_viz, "#888")
                icon  = VIZ_ICONS.get(llm_viz, "📊")
                label = VIZ_LABELS.get(llm_viz, llm_viz)
                llm_viz_counts[llm_viz] = llm_viz_counts.get(llm_viz, 0) + 1
                chips_html += (
                    f'<a href="?page=detail&session={selected_session}&ep={ep_id}" '
                    f'style="text-decoration:none;">'
                    f'<span class="ep-chip" '
                    f'style="background:{color};color:white;'
                    f'border:2px solid {color};cursor:pointer;" '
                    f'title="{label} — click to view Episode {ep_id}">'
                    f'{icon} Ep {ep_id}</span></a>'
                )
            else:
                # Light grey dashed — valid but LLM not yet run
                chips_html += (
                    f'<span class="ep-chip" '
                    f'style="background:#f3f4f6;color:#9ca3af;'
                    f'border:1px dashed #d1d5db;" '
                    f'title="LLM not yet run — go to Episode Detail to generate">'
                    f'Ep {ep_id}</span>'
                )

        st.markdown(chips_html, unsafe_allow_html=True)
        st.caption(
            " Coloured = LLM result (click to open) · "
            " Dashed = not yet run · "
            " Solid grey = too small"
        )

    with sv2:
        if llm_viz_counts:
            st.markdown("**LLM selections so far**")
            for vk, count in sorted(llm_viz_counts.items(), key=lambda x: -x[1]):
                icon  = VIZ_ICONS.get(vk, "📊")
                label = VIZ_LABELS.get(vk, vk)
                color = VIZ_COLORS.get(vk, "#888")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                    f'<span style="font-size:1.1rem">{icon}</span>'
                    f'<span style="flex:1;font-size:0.85rem">{label}</span>'
                    f'<span style="background:{color};color:white;border-radius:12px;'
                    f'padding:1px 10px;font-size:0.78rem;font-weight:500">{count}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("**No LLM results yet**")
            st.caption("Run the LLM on individual episodes to see the breakdown here.")

    # Pedagogical framework reference 
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### Pedagogical Framework Reference")
    st.caption("Rationale behind each visualisation type — click to expand.")

    for vk, info in VIZ_FRAMEWORK_NOTES.items():
        with st.expander(f"{info['title']}"):
            refs     = info.get("references", "")
            ref_html = (f'<h5 style="margin-top:10px">Key references</h5>'
                        f'<p style="font-size:0.82rem;color:#666">{refs}</p>') if refs else ""
            st.markdown(f"""
<div class="framework-box">
  <h5>When to use</h5>
  <p>{info['when']}</p>
  <h5 style="margin-top:10px">Theoretical grounding</h5>
  <p>{info['theory']}</p>
  <h5 style="margin-top:10px">What to look for</h5>
  <p>{info['look_for']}</p>
  {ref_html}
</div>""", unsafe_allow_html=True)

    # Full episode table 
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    with st.expander(" Full episode data", expanded=False):
        display_cols = ["ep","n_turns","n_speakers","regulation_rate","challenge_rate", "gini_coefficient","dominant_regulation","dominant_challenge"]
        st.dataframe(
            sess_ep[display_cols].rename(columns={
                "ep":"Ep","n_turns":"Turns","n_speakers":"Speakers",
                "regulation_rate":"Reg rate","challenge_rate":"Chal rate",
                "gini_coefficient":"Gini","dominant_regulation":"Dom reg",
                "dominant_challenge":"Dom chal",
            }).set_index("Ep"),
            use_container_width=True,
        )


#  PAGE 3 — EPISODE DETAIL

elif "Detail" in page:
    ep_row = ep_features[
        (ep_features["session"]==selected_session) & (ep_features["ep"]==selected_ep)
    ]
    if ep_row.empty:
        st.error("Episode not found."); st.stop()

    r     = ep_row.iloc[0]
    ep_df = df[(df["session"]==selected_session)&(df["ep"]==selected_ep)].copy()
    ta_ok = selected_session not in TA_UNKNOWN

    st.markdown(f"# Session {selected_session} &nbsp;·&nbsp; Episode {selected_ep}")
    if not ta_ok:
        st.warning("TA labels not available for this session.")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Duration", fmt_dur(r["total_duration_sec"]))
        st.metric("Turns",    int(r["n_turns"]))
    with c2:
        st.metric("Speakers",  int(r["n_speakers"]))
        st.metric("Questions", f"{int(r['question_count'])} ({r['question_density']:.0%})")
    with c3:
        g= r["gini_coefficient"]
        eq = "🟢 High equity" if g<0.3 else "🟡 Moderate" if g<0.6 else "🔴 Low equity"
        st.metric("Gini", f"{g:.3f}")
        st.markdown(f"<small>{eq}</small>", unsafe_allow_html=True)
        reg= r["regulation_rate"]
        rl= "🟢 High" if reg>=0.7 else "🟡 Moderate" if reg>=0.3 else "🔴 Low"
        st.metric("Regulation", f"{reg:.0%}")
        st.markdown(f"<small>{rl}</small>", unsafe_allow_html=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### LLM Visualisation Selection")

    min_turns= int(r["n_turns"])
    min_speakers = int(r["n_speakers"])
    too_small= min_turns < 3 or min_speakers < 2

    if too_small:
        st.warning(
            f" Episode has {min_turns} turn(s) and {min_speakers} speaker(s) — "
            "too small for a meaningful visualisation. "
            "Try an episode with at least 3 turns and 2 speakers."
        )
    else:
        cache_key= f"llm_{selected_session}_{selected_ep}"
        deep_cache_key = f"deep_{selected_session}_{selected_ep}"
        features_text= build_features_text(selected_session, selected_ep, ep_df)

        col_btn1, col_btn2, _ = st.columns([1,1,4])
        with col_btn1:
            run_llm = st.button("▶ Run LLM", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("↺ Reset", use_container_width=True):
                for k in [cache_key, deep_cache_key]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()

        if run_llm:
            with st.spinner("Calling Groq API..."):
                st.session_state[cache_key] = call_llm(selected_session, selected_ep, ep_df)
            log_llm_result(selected_session, selected_ep,
                           st.session_state[cache_key], features_text)
            if deep_cache_key in st.session_state:
                del st.session_state[deep_cache_key]

        if cache_key in st.session_state:
            result= st.session_state[cache_key]
            viz_key = parse_viz_key(result["selected_visualisation"])
            icon= VIZ_ICONS.get(viz_key, "📊")

            if result.get("status") == "error":
                st.error(f"LLM error: {result['reason']}")
            elif result.get("status") == "no_api_key":
                st.info("No API key — showing default Timeline.")

            st.markdown(f"""
<div class="llm-panel">
  <h4>{icon} Recommended: {result['selected_visualisation']}</h4>
  <p>{result['teacher_explanation']}</p>
</div>
<div class="reasoning-pill">
  <strong>Why this visualisation:</strong> {result['reason']}
</div>""", unsafe_allow_html=True)

            st.markdown("")
            if st.button("💬 Why did you choose this visualisation?", use_container_width=False):
                with st.spinner("Generating detailed explanation..."):
                    deep = call_deep_reasoning(result["selected_visualisation"], features_text)
                    st.session_state[deep_cache_key] = deep
                    log_llm_result(selected_session, selected_ep,
                                   result, features_text, deep_reasoning=deep)

            if deep_cache_key in st.session_state:
                st.markdown(f"""
<div class="deep-reasoning">
  <strong>🔍 Detailed reasoning:</strong><br><br>
  {st.session_state[deep_cache_key]}
</div>""", unsafe_allow_html=True)

            if viz_key in VIZ_FRAMEWORK_NOTES:
                info = VIZ_FRAMEWORK_NOTES[viz_key]
                with st.expander(f"Pedagogical framework — {info['title']}", expanded=False):
                    refs= info.get("references", "")
                    ref_html = f'<h5 style="margin-top:10px">Key references</h5><p style="font-size:0.82rem;color:#666">{refs}</p>' if refs else ""
                    st.markdown(f"""
<div class="framework-box">
  <h5>When to use</h5><p>{info['when']}</p>
  <h5 style="margin-top:10px">Theoretical grounding</h5><p>{info['theory']}</p>
  <h5 style="margin-top:10px">What to look for in this chart</h5><p>{info['look_for']}</p>
  {ref_html}
</div>""", unsafe_allow_html=True)

            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

            #Comparison section
            st.markdown("**Compare visualisations**")
            comp_col1, comp_col2 = st.columns(2)

            all_viz_options = {
                "Timeline": "timeline",
                "Participation Chart": "participation",
                "Network Graph" : "network",
                "Stacked Bar": "stacked_bar",
                "Heatmap": "heatmap",
            }

            with comp_col1:
                left_choice = st.selectbox(
                    "Left chart",
                    options=list(all_viz_options.keys()),
                    index=list(all_viz_options.values()).index(viz_key),
                    key=f"left_viz_{cache_key}",
                )
            with comp_col2:
                # Default right to Timeline unless LLM chose Timeline
                default_right = "Timeline" if viz_key != "timeline" else "Participation Chart"
                right_choice = st.selectbox(
                    "Right chart",
                    options=list(all_viz_options.keys()),
                    index=list(all_viz_options.keys()).index(default_right),
                    key=f"right_viz_{cache_key}",
                )

            show_compare= st.toggle("Show side-by-side comparison", value=False)

            if show_compare:
                if left_choice == right_choice:
                    st.warning("Both charts are the same — select different visualisations to compare.")
                else:
                    l_key = all_viz_options[left_choice]
                    r_key = all_viz_options[right_choice]
                    l_icon= VIZ_ICONS.get(l_key, "📊")
                    r_icon= VIZ_ICONS.get(r_key, "📊")
                    l_col, r_col = st.columns(2)
                    with l_col:
                        st.markdown(f'<div class="compare-header">{l_icon} {left_choice}</div>',
                                    unsafe_allow_html=True)
                        l_fig, _ = render_visualization(l_key, ep_df, selected_session,
                                                        selected_ep, label="", ta_annotated=ta_ok)
                        st.pyplot(l_fig, use_container_width=True); plt.close(l_fig)
                    with r_col:
                        st.markdown(f'<div class="compare-header">{r_icon} {right_choice}</div>',
                                    unsafe_allow_html=True)
                        r_fig, _ = render_visualization(r_key, ep_df, selected_session,
                                                        selected_ep, label="", ta_annotated=ta_ok)
                        st.pyplot(r_fig, use_container_width=True); plt.close(r_fig)
            else:
                # Single viz — LLM choice
                st.markdown(f'<div class="viz-badge">{icon} {result["selected_visualisation"]}</div>',
                            unsafe_allow_html=True)
                viz_fig, _ = render_visualization(viz_key, ep_df, selected_session,
                                                  selected_ep, label="", ta_annotated=ta_ok)
                st.pyplot(viz_fig, use_container_width=True); plt.close(viz_fig)

        else:
            st.info("Click **▶ Run LLM** to select and render the most appropriate visualisation.")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### Episode features")
    ft1, ft2 = st.columns(2)
    with ft1:
        st.caption("Temporal & conversational")
        st.dataframe(pd.DataFrame({
            "Feature":["Duration","Turns","Avg turn","Avg latency","Questions","Q density","Avg utt. length"],
            "Value":[fmt_dur(r["total_duration_sec"]),int(r["n_turns"]),
                     f"{r['avg_turn_length_sec']}s",f"{r['avg_response_latency']}s",
                     int(r["question_count"]),f"{r['question_density']:.1%}",
                     f"{r['avg_utt_length_words']:.1f} words"],
        }).set_index("Feature"), use_container_width=True)
    with ft2:
        st.caption("Challenge & regulation")
        st.dataframe(pd.DataFrame({
            "Feature":["Challenge (any)","  C","  E","  M","  T","Dom. challenge",
                       "Regulation (any)","  MC","  TA","  RA","Dom. regulation"],
            "Value":[f"{r['challenge_rate']:.1%}",f"{r['C_rate']:.1%}",f"{r['E_rate']:.1%}",
                     f"{r['M_rate']:.1%}",f"{r['T_rate']:.1%}",r["dominant_challenge"],
                     f"{r['regulation_rate']:.1%}",f"{r['MC_rate']:.1%}",
                     f"{r['TA_rate']:.1%}" if ta_ok else "Unknown",
                     f"{r['RA_rate']:.1%}",r["dominant_regulation"]],
        }).set_index("Feature"), use_container_width=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("### Participation")
    speak= ep_df.groupby("speaker")["duration"].sum().sort_values(ascending=False)
    total = speak.sum()
    pc1, pc2 = st.columns([2,1])
    with pc1:
        st.dataframe(pd.DataFrame([{
            "Speaker":s,"Role":"🎓 Teacher" if str(s).startswith("T0") else "👤 Student",
            "Time (s)":int(v),"Share":f"{v/total*100:.1f}%",
            "Turns":int((ep_df["speaker"]==s).sum())} for s,v in speak.items()
        ]).set_index("Speaker"), use_container_width=True)
    with pc2:
        g = r["gini_coefficient"]
        st.metric("Gini", f"{g:.3f}")
        if g<0.3:st.success("🟢 High equity")
        elif g<0.6: st.warning("🟡 Moderate")
        else:st.error("🔴 Low equity")

    with st.expander("Raw transcript", expanded=False):
        t = ep_df[["start","end","speaker","content","C","E","M","T","MC","TA","RA"]].copy()
        t.index = range(1, len(t)+1); t.index.name = "#"
        st.dataframe(t, use_container_width=True, height=380)
