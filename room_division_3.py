"""
ILP hotel planning voor september 2025 met walk-ins.

Belangrijk:
- Binaire variabelen x[g,r] voor kamerkeuze per gast.
- Binaire variabelen y[g] voor acceptatie van walk-ins.
- Bezettingsrestricties per kamer per dag.
- Lock-restricties voor in-house of T-1 gasten.
- Doel: max geaccepteerde walk-ins + tevredenheid - move penalty.

Benodigd:
- pulp  (pip install pulp)
- plotly
"""

from __future__ import annotations
from datetime import date, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st
from itertools import combinations
from room_division_2 import simulate_variant
# ----------------------------- Data -----------------------------

@dataclass
class Room:
    id: str
    type: str
    floor: int
    elev: bool

@dataclass
class Guest:
    name: str
    need: str
    start: date
    end: date           # checkout, exclusief
    pref_floor: int
    near_elev: bool
    prebooked: bool
    planned_room: str | None   # pre-assigned bij prebooked
    walk_in: bool


def daterange(d0: date, d1: date):
    """Genereer opeenvolgende dagen [d0, d1)."""
    d = d0
    while d < d1:
        yield d
        d += timedelta(days=1)


def overlaps(a0: date, a1: date, b0: date, b1: date) -> bool:
    """True als intervallen [a0,a1) en [b0,b1) elkaar snijden."""
    return a0 < b1 and b0 < a1


# ----------------------- Synthetic scenario ---------------------

def generate_rooms() -> List[Room]:
    """Maak de 8 kamers volgens jouw schema."""
    return [
        Room("11", "single", 1, True),
        Room("12", "double", 1, False),
        Room("21", "single", 2, True),
        Room("22", "double", 2, False),
        Room("31", "single", 3, True),
        Room("32", "double", 3, False),
        Room("41", "single", 4, True),
        Room("42", "double", 4, False),
    ]


def generate_prebooked_with_gaps(rooms: List[Room], start: date, end: date,
                                 gap_p: float = 0.4, cancel_p: float = 0.15,
                                 seed: int = 42) -> List[Guest]:
    """Genereer prebooked gasten met gaten en annuleringen zoals in simulatie 2."""
    rng = random.Random(seed)
    guests: List[Guest] = []
    gid = 1
    for r in rooms:
        cur = start
        while cur < end:
            stay = rng.randint(2, 6)
            g = Guest(
                name=f"G{gid:02d}",
                need=rng.choice(["single", "double"]),
                start=cur,
                end=min(cur + timedelta(days=stay), end),
                pref_floor=rng.randint(1, 4),
                near_elev=rng.choice([True, False]),
                prebooked=True,
                planned_room=r.id,
                walk_in=False,
            )
            if rng.random() > cancel_p:
                guests.append(g)
                cur = g.end
            else:
                cur = min(cur + timedelta(days=stay), end)
            if rng.random() < gap_p:
                cur = min(cur + timedelta(days=rng.choice([0, 1])), end)
            gid += 1
    return guests


def generate_walkins(n: int, start: date, end: date, seed: int = 42) -> List[Guest]:
    """Genereer walk-ins met willekeurige aankomst en verblijfsduur."""
    rng = random.Random(seed + 1)
    guests: List[Guest] = []
    for k in range(n):
        stay = rng.randint(1, 4)
        arr = start + timedelta(days=rng.randint(0, (end - start).days - 1))
        guests.append(
            Guest(
                name=f"W{k+1}",
                need=rng.choice(["single", "double"]),
                start=arr,
                end=min(arr + timedelta(days=stay), end),
                pref_floor=rng.randint(1, 4),
                near_elev=rng.choice([True, False]),
                prebooked=False,
                planned_room=None,
                walk_in=True,
            )
        )
    return guests


def assert_no_double_occupancy(df, period_start, period_end, rooms):
    """Stop als er echte dubbele bezetting is per dag×kamer."""
    from datetime import timedelta
    days = []
    d = period_start
    while d < period_end:
        days.append(d)
        d += timedelta(days=1)

    conflicts = []
    for rid in [r.id for r in rooms]:
        for t in days:
            gasten = df[
                (df["Assigned_room"] == rid) &
                (df["Start"] <= t) &
                (df["Eind"] > t)          # end = checkout, exclusief
            ]["Gast"].tolist()
            if len(gasten) > 1:
                conflicts.append((t, rid, gasten))
    if conflicts:
        st.info(f"Double occupancy: {len(conflicts)} conflicts, e.g. {conflicts}")
        # raise ValueError(f"Double occupancy: {len(conflicts)} conflicts, e.g. {conflicts[:3]}")

def true_occupancy(df, period_start, period_end, rooms) -> float:
    """Echte bezetting: tel unieke dag×kamer-slots met een gast."""
    from datetime import timedelta
    total_slots = ((period_end - period_start).days) * len(rooms)
    occupied = 0
    d = period_start
    while d < period_end:
        occ_d = df.dropna(subset=["Assigned_room"])
        occ_d = occ_d[(occ_d["Start"] <= d) & (occ_d["Eind"] > d)]
        occupied += occ_d.drop_duplicates(["Assigned_room"]).shape[0]
        d += timedelta(days=1)
    return occupied / total_slots if total_slots else 0.0

# ----------------------------- ILP ------------------------------
def build_and_solve_ilp(rooms, prebooked, walkins,
                        period_start, period_end,
                        move_penalty: float = 0.25,
                        accept_reward: float = 1.0):
    """
    ILP met dag×kamer-slot variabelen z[g,r,t] om dubbele bezetting uit te sluiten.
    """
    all_guests = prebooked + walkins
    rooms_by_id = {r.id: r for r in rooms}
    days = list(daterange(period_start, period_end))  # t zijn dagen [start, end)

    def sat(g, r):
        s = 0
        s += 1 if r.type == g.need else 0
        s += 1 if r.floor == g.pref_floor else 0
        s += 1 if r.elev == g.near_elev else 0
        return s

    satisfaction = {(g.name, r.id): sat(g, rooms_by_id[r.id]) for g in all_guests for r in rooms}

    # occ[g,t] = 1 als g op dag t in-house is
    occ = {(g.name, t): 1 if (g.start < t + timedelta(days=1) and t < g.end) else 0
           for g in all_guests for t in days}

    def is_locked(g):
        return (g.start <= period_start < g.end) or (g.start == period_start + timedelta(days=1))

    locked = {g.name: is_locked(g) and g.prebooked and g.planned_room is not None for g in all_guests}

    m = pulp.LpProblem("Hotel_ILP", pulp.LpMaximize)

    # Toewijzing per gast-kamer
    x = pulp.LpVariable.dicts("x", [(g.name, r.id) for g in all_guests for r in rooms],
                              lowBound=0, upBound=1, cat="Binary")
    # Acceptatie walk-ins
    y = pulp.LpVariable.dicts("y", [g.name for g in all_guests if g.walk_in],
                              lowBound=0, upBound=1, cat="Binary")
    # Dag×kamer bezetting
    z = pulp.LpVariable.dicts("z", [(g.name, r.id, t) for g in all_guests for r in rooms for t in days],
                              lowBound=0, upBound=1, cat="Binary")

    # 1) Eén kamer per gast
    for g in all_guests:
        if g.walk_in:
            m += pulp.lpSum(x[(g.name, r.id)] for r in rooms) == y[g.name]
        else:
            m += pulp.lpSum(x[(g.name, r.id)] for r in rooms) == 1

    # 2) Type-match
    for g in all_guests:
        for r in rooms:
            if rooms_by_id[r.id].type != g.need:
                m += x[(g.name, r.id)] == 0

    # 3) Lock in-house of T-1 prebooked
    for g in all_guests:
        if locked[g.name]:
            pr = next((pg.planned_room for pg in prebooked if pg.name == g.name), None)
            if pr:
                for r in rooms:
                    m += x[(g.name, r.id)] == (1 if r.id == pr else 0)

    # 4) Link z aan x en aan aanwezigheid occ
    #    z[g,r,t] = 1 alleen als x[g,r]=1 én gast op dag t aanwezig is
    # for g in all_guests:
    #     for r in rooms:
    #         for t in days:
    #             m += z[(g.name, r.id, t)] <= x[(g.name, r.id)]
    #             m += z[(g.name, r.id, t)] <= occ[(g.name, t)]
    #             m += z[(g.name, r.id, t)] >= x[(g.name, r.id)] + occ[(g.name, t)] - 1

    # 5) Kamercapaciteit per dag: som_g z[g,r,t] ≤ 1
    # for r in rooms:
    #     for t in days:
    #         m += pulp.lpSum(z[(g.name, r.id, t)] for g in all_guests) <= 1

    for r in rooms:
        for g1, g2 in combinations(all_guests, 2):
            if overlaps(g1.start, g1.end, g2.start, g2.end):
                m += x[(g1.name, r.id)] + x[(g2.name, r.id)] <= 1

    # 6) Move-indicator
    moved = {}
    for g in prebooked:
        if g.planned_room:
            mv = pulp.LpVariable(f"moved_{g.name}", 0, 1, cat="Binary")
            moved[g.name] = mv
            m += mv >= 1 - x[(g.name, g.planned_room)]

    # Doel
    obj = pulp.lpSum(accept_reward * y[gn] for gn in y.keys())
    obj += pulp.lpSum(x[(g.name, r.id)] * satisfaction[(g.name, r.id)] for g in all_guests for r in rooms)
    obj -= pulp.lpSum(move_penalty * moved[gn] for gn in moved.keys())
    m += obj

    _ = m.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract
    rows = []
    for g in all_guests:
        rid = None
        for r in rooms:
            if pulp.value(x[(g.name, r.id)]) > 0.5:
                rid = r.id
                break
        rows.append({
            "Gast": g.name,
            "Walk_in": g.walk_in,
            "Prebooked": g.prebooked,
            "Start": g.start,
            "Eind": g.end,
            "Nachten": (g.end - g.start).days,
            "Need": g.need,
            "Pref_floor": g.pref_floor,
            "Near_elev": g.near_elev,
            "Planned_room": g.planned_room,
            "Assigned_room": rid,
            "Moved": (rid is not None and g.prebooked and g.planned_room and rid != g.planned_room),
            "Accepted": (not g.walk_in) or (g.walk_in and (g.name in y and pulp.value(y[g.name]) > 0.5)),
            "Satisfaction_0_3": satisfaction.get((g.name, rid), 0) if rid else 0,
        })
    df = pd.DataFrame(rows).sort_values(["Start", "Assigned_room", "Gast"]).reset_index(drop=True)

    # Safety check
    assert_no_double_occupancy(df, period_start, period_end, rooms)

    # Metrics op echte slots
    occ_rate = true_occupancy(df, period_start, period_end, rooms)
    metrics = {
        "walkins_created": sum(1 for g in walkins),
        "walkins_accepted": int(df.query("Walk_in and Accepted").shape[0]),
        "accept_rate_pct": round(100 * df.query("Walk_in").Accepted.mean(), 1) if df.query("Walk_in").shape[0] else 0.0,
        "moves": int(df["Moved"].sum()),
        "occupancy_pct": round(100 * occ_rate, 1),
        "satisfaction_avg": round(float(df["Satisfaction_0_3"].mean()), 2) if not df.empty else 0.0,
        "objective_value": round(pulp.value(obj), 3),
        "solver_status": pulp.LpStatus[m.status],
    }
    return df, metrics

# --------------------------- Plotly -----------------------------

def gantt_plot(df: pd.DataFrame, title: str = "Kamerbezetting • ILP • Sep 2025") -> go.Figure:
    """Maak een Gantt met Plotly. Hatches via pattern shape op Walk-in."""
   
    dff = df.dropna(subset=["Assigned_room"]).copy()
    dff["Assigned_room"] = dff["Assigned_room"].astype(str)   # forceer string
    dff["Type"] = dff.apply(lambda r: "Walk-in" if r["Walk_in"] else "Prebooked", axis=1)

    rooms_order = sorted(dff["Assigned_room"].unique())  # alleen echte kamers

    
    

     
    # Maak een kolom voor de kleurcategorie
    def assign_color(row):
        if row["Happy_floor"]:
            return "green"
        elif not row["Happy_floor"] and row["Shuffled"]:
            return "orange"
        else:
            return "red"
    
    dff["Start"] = pd.to_datetime(dff["Start"])
    dff["Eind"] = pd.to_datetime(dff["Eind"])
    # Maak einde iets eerder zodat blokken aansluiten
    dff["Eind_plot"] = dff["Eind"] - pd.Timedelta(seconds=1)
    dff["Kamer"]= dff["Assigned_room"]
    dff["Floor"] = dff["Kamer"].str[0].astype(int)
    dff["Room_type"] = dff["Kamer"].str[-1].astype(int).map(lambda x: "single" if x % 2 == 1 else "double")
    dff["Happy_floor"] = dff["Pref_floor"] == dff["Floor"]
    dff["Happy_room"] = dff["Need"] == dff["Room_type"]    
    dff["Shuffled"] = dff["Walk_in"]  # Walk-in gasten zijn de 'shuffled' gasten
    # Satisfactiescore (voorbeeld: alleen floor afstand)
    dff["Satisfaction_score"] = 100 - ((dff["Floor"] - dff["Pref_floor"]).abs() / 3) * 100
    dff["ColorCategory"] = dff.apply(assign_color, axis=1)

    fig = px.timeline(
        dff,
        x_start="Start",
        x_end="Eind_plot",
        y="Assigned_room",
        color="ColorCategory",
        text="Gast",
        hover_data={
            "Gast": True, "Kamer": True, "Planned_room": True,
            "Start": True, "Eind": True,
            "Need": True, "Pref_floor": True, "Near_elev": True,
            "Happy_floor": True, "Happy_room": True, "Walk_in": True
        },
        color_discrete_map={
            "green": "green",
            "orange": "orange",
            "red": "red"
        }
    )

    fig.update_traces(marker=dict(line=dict(color="black", width=1)))
    fig.update_traces(width= .5, offset = -0.25)
    # Y-as categorisch + volgorde
    fig.update_yaxes(
        type="category",
        # categoryorder="array",
        categoryarray=rooms_order,
        autorange="reversed",
        # showgrid=True,
        #gridwidth=1
    )

    # X-as met daggrid
    fig.update_xaxes(showgrid=True, dtick="D1", gridwidth=1)

    # Strakke blokken
    fig.update_layout(bargap=0, bargroupgap=0, height=700, title="Kamerindeling gasten")

    # Verticale lijnen per dag
    for d in pd.date_range(df["Start"].min(), df["Eind"].max(), freq="D"):
        fig.add_vline(x=d, line_width=1, line_color="rgba(0,0,0,0.15)")

    # Horizontale lijnen TUSSEN kamers
    for i in range(len(rooms_order) - 1):
        y_pos = i + 0.5
        fig.add_shape(
            type="line",
            x0=df["Start"].min(), x1=df["Eind"].max(),
            y0=y_pos, y1=y_pos,
            xref="x", yref="y",
            line=dict(color="rgba(0,0,0,0.25)", width=1)
        )

    return fig




def main():
    PERIOD_START = date(2025, 9, 1)
    PERIOD_END   = date(2025, 10, 1)  # exclusief

    rooms = generate_rooms()
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1: 
        
        gap_p = st.slider("Gap probability", 0.0, 1.0, 0.4, 0.05)
    with col2:
        cancel_p = st.slider("Cancellation prob.", 0.0, 1.0, 0.15, 0.05)
    with col3:
        move_penalty = st.slider("Move penalty", 0.0, 2.0, 0.25, 0.05)
    with col4:
        accept_reward = st.slider("Accept reward", 0.0, 5.0, 2.0, 0.1)
    with col5:
        n_walkins = st.slider("Number of walk-ins", 0, 20, 1, 1)
    prebooked = generate_prebooked_with_gaps(rooms, PERIOD_START, PERIOD_END,
                                             gap_p, cancel_p, seed=42)
    # gelijk aan simulatie 2 orde van grootte
    walkins = generate_walkins(n_walkins, start=PERIOD_START, end=PERIOD_END, seed=42)

    df_ilp, metrics = build_and_solve_ilp(
        rooms, prebooked, walkins,
        PERIOD_START, PERIOD_END,
        move_penalty,     # zwaarder maakt minder herplaatsingen
        accept_reward      # hoger stimuleert acceptatie van walk-ins
    )

    st.write(metrics)
    st.write(df_ilp)
    totaal_nachten=df_ilp["Nachten"].sum()
    st.metric("Totaal nachten", totaal_nachten)
    totaal_mogelijk = len(rooms) * (PERIOD_END - PERIOD_START).days
    st.metric("Bezettinsgraad (%)", f"{100*totaal_nachten/totaal_mogelijk:.1f}")

    days = list(daterange(PERIOD_START, PERIOD_END))
    total_slots = len(days) * len(rooms)

    occupied_slots = 0
    for _, g in df_ilp.iterrows():
        for d in daterange(g["Start"], g["Eind"]):
            if pd.notna(g["Assigned_room"]):
                occupied_slots += 1

    occupancy = occupied_slots / total_slots
    st.metric("Bezettingsgraad (%)", f"{occupancy*100:.1f}")

    # Plotly Gantt
    fig = gantt_plot(df_ilp, "Kamerbezetting • ILP • September 2025")
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)
    # Optioneel export
    # fig.write_html("hotel_ilp_gantt_sep2025.html")
    # df_ilp.to_csv("hotel_ilp_assignments_sep2025.csv", index=False)


# ---------------------------- Run -------------------------------

if __name__ == "__main__":
    main()