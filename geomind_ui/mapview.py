"""Plan-view map of every dataset; selecting a dataset zooms close to it."""
import numpy as np
import plotly.graph_objects as go

from . import catalog as C

_DIM = 0.22


def _lines(rings):
    xs, ys = [], []
    for r in rings:
        xs += list(r[:, 0]) + [None]
        ys += list(r[:, 1]) + [None]
    return xs, ys


def make(cat, focus=None, highlight=(), layers=("survey", "horizon", "polygons", "faults", "wells"), targets=(), revision=0, zoom_to_results=True):
    """focus: item (zoom target) or None; highlight: ids drawn emphasised."""
    fig = go.Figure()
    hl = set(highlight) | ({focus["id"]} if focus else set())
    survey = cat["survey"]
    items = cat["items"]
    # seismic footprint (one drawn: all volumes share the survey)
    vols = [i for i in items if i["geom"] == "volume" and i.get("bbox")]
    if vols and "survey" in layers:
        b = vols[0]["bbox"]
        fig.add_trace(go.Scatter(x=[b[0], b[2], b[2], b[0], b[0]], y=[b[1], b[1], b[3], b[3], b[1]], mode="lines", fill="toself", fillcolor="rgba(120,150,180,.10)",
                                 line=dict(color="rgba(90,120,150,.55)", width=1, dash="dot"), name="Seismic survey", hoverinfo="name", showlegend=False))
    # horizon
    for i in items:
        if i["geom"] == "horizon" and "horizon" in layers:
            pts = C.horizon_points(i.get("name"))
            if pts is not None:
                x, y, t = pts
                on = i["id"] in hl or not hl
                fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=4, color=t, colorscale="Viridis", reversescale=True, opacity=0.55 if on else _DIM,
                                                                                    colorbar=dict(title="TWT ms", thickness=10, len=0.5, x=1.0) if on else None),
                                           name=i["label"], hovertemplate="TWT %{marker.color:.0f} ms<extra>" + i["label"] + "</extra>", showlegend=False))
    # polygons
    merged = {}
    for i in items:
        if i.get("kind") == "polygon" and "polygons" in layers:
            if i["id"] in hl:
                x, y = _lines(i["rings"])
                col = C.CLS_COLOR.get(i["cls"], "#00877e")
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=_rgba(col, .45), line=dict(color=col, width=2.5), name=i["label"],
                                         hoverinfo="name", showlegend=False))
            else:
                merged.setdefault(i["cls"], []).extend(i["rings"])
    for cls, rings in merged.items():
        x, y = _lines(rings)
        col = C.CLS_COLOR.get(cls, "#00877e")
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=_rgba(col, .18 if not hl else .08), line=dict(color=_rgba(col, .7 if not hl else .35), width=1),
                                 name=cls, hoverinfo="name", showlegend=False))
    # faults
    dim = []
    for i in items:
        if i.get("kind") == "fault" and "faults" in layers:
            if i["id"] in hl:
                x, y = _lines(i["rings"])
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C.CLS_COLOR["fault"], width=4), name=f"{i['label']} ({i.get('time_ms')} ms)",
                                         hoverinfo="name", showlegend=False))
            else:
                dim.extend(i["rings"])
    if dim:
        x, y = _lines(dim)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=_rgba(C.CLS_COLOR["fault"], .85 if not hl else .3), width=2), name="faults", hoverinfo="name", showlegend=False))
    # wells
    wells = [i for i in items if i["geom"] == "well" and i.get("bbox")]
    if wells and "wells" in layers:
        sel = [w for w in wells if w["id"] in hl]
        rest = [w for w in wells if w["id"] not in hl]
        for grp, big in ((rest, False), (sel, True)):
            if grp:
                fig.add_trace(go.Scatter(x=[w["bbox"][0] for w in grp], y=[w["bbox"][1] for w in grp], mode="markers+text", text=[w["label"] for w in grp], textposition="top center",
                                         marker=dict(size=16 if big else 10, color="#1b2b3a", symbol="diamond", line=dict(color="#fff", width=1.5), opacity=1 if (big or not hl) else .5),
                                         hovertext=[w["label"] for w in grp], hoverinfo="text", showlegend=False))
    # Structured results are points; no invented boundary or extent.
    if targets:
        fig.add_trace(go.Scatter(x=[p["x"] for p in targets], y=[p["y"] for p in targets],
                                 text=[p["label"] for p in targets], mode="markers+text", textposition="top center",
                                 marker=dict(size=18, color="#00877e", symbol="circle-open", line=dict(width=3)),
                                 name="Copilot result", hovertemplate="%{text}<br>X %{x}<br>Y %{y}<extra></extra>"))
    # window
    win = C.focus_box(focus, survey) if focus else (C._pad(survey, 0.06) if survey else None)
    if targets and not focus and zoom_to_results:
        a = np.array([[p["x"], p["y"]] for p in targets])
        bounds = [a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()]
        win = C.focus_box({"bbox": bounds}, survey)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#ffffff", plot_bgcolor="#f8fafb", dragmode="pan", showlegend=False,
                      uirevision=f"{revision}:{focus['id'] if focus else 'all'}", height=560, hoverlabel=dict(bgcolor="#fff"))
    ax = dict(showgrid=True, gridcolor="rgba(120,150,180,.25)", zeroline=False, showticklabels=True, tickfont=dict(size=10, color="#6b7c8d"), ticks="")
    fig.update_xaxes(**ax, title="X · dataset coordinates")
    fig.update_yaxes(**ax, title="Y · dataset coordinates", scaleanchor="x", scaleratio=1)
    if win:
        fig.update_xaxes(range=[win[0], win[2]])
        fig.update_yaxes(range=[win[1], win[3]])
    return fig


def _rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
