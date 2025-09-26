from datetime import date, timedelta
import random
import pulp as pl

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    st.set_page_config(page_title="Kamerindeling", layout="wide")
except Exception:
    pass


# ---------- Helpers ----------
def overlap(g, h):
    return not (g["end"] <= h["start"] or h["end"] <= g["start"])

def force_rooms(prob, guests, rooms, x, fixed_map):
    name_to_idx = {g["name"]: i for i, g in enumerate(guests)}
    room_to_idx = {str(r["id"]): i for i, r in enumerate(rooms)}
    for guest_name, room_id in fixed_map.items():
        gi = name_to_idx[guest_name]
        ri_target = room_to_idx[str(room_id)]
        prob += x[(gi, ri_target)] == 1
        for rj in range(len(rooms)):
            if rj != ri_target:
                prob += x[(gi, rj)] == 0

def compat_cost(g, r):
    W_IMPOSSIBLE = 10_000
    W_UPGRADE    = 5
    W_FLOOR_DIFF = 1
    W_ELEV_MISS  = 2
    if g["need"] == r["type"]:
        base = 0
    else:
        if g["need"] == "single" and r["type"] == "double":
            base = W_UPGRADE
        else:
            return W_IMPOSSIBLE
    base += W_FLOOR_DIFF * abs(g["pref_floor"] - r["floor"])
    if g["near_elev"] and not r["elev"]:
        base += W_ELEV_MISS
    return base

def random_guests(n=30, start_date=date(2025, 9, 1), end_date=date(2025, 9, 30)):
    guests = []
    for i in range(n):
        name = f"G{i+1}"
        need = random.choice(["single", "double"])
        stay_length = random.randint(2, 5)  # nachten
        start_offset = random.randint(0, (end_date - start_date).days - stay_length)
        start = start_date + timedelta(days=start_offset)
        end = start + timedelta(days=stay_length)
        pref_floor = random.randint(1, 4)
        near_elev = random.choice([True, False])
        guests.append({
            "name": name, "need": need,
            "start": start, "end": end,
            "pref_floor": pref_floor, "near_elev": near_elev
        })
    return guests

def not_so_random_guests(rooms, shuffle_factor, start_date=date(2025, 9, 1), end_date=date(2025, 9, 30)):
    """Genereer gasten die bij de kamers passen, met een deel 'walk-ins' op basis van shuffle_factor."""
    #st.metric("Percentage walk-in gasten/boekingen", f"{shuffle_factor*100:.0f}%")

    guests = []
    i = 1

    for r in rooms:
        start_date_iter = start_date
        while start_date_iter < end_date:
            # KANS PER GAST
            is_shuffled = random.random() < shuffle_factor

            # Voorkeuren bepalen
            if is_shuffled:
                need = random.choice(["single", "double"])
                pref_floor = random.randint(1, 4)
                near_elev = random.choice([True, False])
            else:
                need = r["type"]
                pref_floor = r["floor"]
                near_elev = r["elev"]
            if is_shuffled:
                name = f"W{i}"
            else:
                name = f"G{i}"
            stay_length = random.randint(2, 6)  # nachten
            start = start_date_iter
            end = start + timedelta(days=stay_length)

            guests.append({
                "name": name,
                "need": need,
                "start": start,
                "end": end,
                "pref_floor": pref_floor,
                "near_elev": near_elev,
                "geplande_room": r["id"],
                "shuffled": is_shuffled
            })

            in_between = 0  # eventueel random pauze tussen verblijven
            start_date_iter = end + timedelta(days=in_between)
            i += 1

    # Willekeurige volgorde van gasten
    random.shuffle(guests)

    # TELLEN PER GAST
    total_guests = len(guests)
    total_shuffled = sum(g["shuffled"] for g in guests)

    t1,t2,t3=st.columns(3)
    with t1:
        st.metric("Aantal gasten", f"{total_guests}")
    with t2:
        st.metric("Aantal walk in gasten (W)", f"{total_shuffled}")
    with t3:
        st.metric("Waargenomen walk-in %", f"{(100*total_shuffled/total_guests):.1f}%")
    
    return guests

# ---------- App ----------
def main():
    st.title("Kamerindeling met ILP + Gantt")

    # Kamers (id: eerste digit = verdieping, laatste digit: oneven=single, even=double)
    rooms = [
        {"id": "11", "type": "single", "floor": 1, "elev": True},
        {"id": "12", "type": "double", "floor": 1, "elev": False},
        {"id": "21", "type": "single", "floor": 2, "elev": True},
        {"id": "22", "type": "double", "floor": 2, "elev": False},
        {"id": "31", "type": "single", "floor": 3, "elev": True},
        {"id": "32", "type": "double", "floor": 3, "elev": False},
        {"id": "41", "type": "single", "floor": 4, "elev": True},
        {"id": "42", "type": "double", "floor": 4, "elev": False},
    ]

    # Gasten (random demo)
    shuffle_factor = st.slider("Shuffle factor", 0, 100, 10, 5)/100
    guests = not_so_random_guests(rooms, shuffle_factor)
  
    # ---------- ILP ----------
    prob = pl.LpProblem("RoomAssignment", pl.LpMinimize)

    # Beslissingsvariabelen
    x = {}
    cost = {}
    for gi, g in enumerate(guests):
        for ri, r in enumerate(rooms):
            c = compat_cost(g, r)
            x[(gi, ri)] = pl.LpVariable(f"x_{gi}_{ri}", lowBound=0, upBound=1, cat=pl.LpBinary)
            cost[(gi, ri)] = c

    # Optioneel: vaste kamers afdwingen
    # force_rooms(prob, guests, rooms, x, {"G7": "22"})  # voorbeeld

    # Niet-toewijzen toestaan met penalty
    PENALTY_UNASSIGNED = 10_000
    u = {gi: pl.LpVariable(f"u_{gi}", lowBound=0, upBound=1, cat=pl.LpBinary)
        for gi in range(len(guests))}

    # Doel: plaatsingskosten + straf voor niet-plaatsen
    prob += (
        pl.lpSum(cost[(gi, ri)] * x[(gi, ri)] for gi in range(len(guests)) for ri in range(len(rooms)))
        + pl.lpSum(PENALTY_UNASSIGNED * u[gi] for gi in range(len(guests)))
    )

    # Elke gast ofwel in 1 kamer, of onassigned
    for gi in range(len(guests)):
        prob += pl.lpSum(x[(gi, ri)] for ri in range(len(rooms))) + u[gi] == 1

    # Geen overlap in dezelfde kamer
    for ri in range(len(rooms)):
        for gi in range(len(guests)):
            for hj in range(gi + 1, len(guests)):
                if overlap(guests[gi], guests[hj]):
                    prob += x[(gi, ri)] + x[(hj, ri)] <= 1

    prob.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[prob.status]
    st.write("Solver status:", status)
    if status not in ("Optimal", "Not Solved"):  # CBC geeft vaak 'Optimal'
        st.error(f"Optimalisatie niet gelukt: {status}")
        return

    # ---------- Oplossing uitlezen ----------
    assignment = {}
    unassigned = []
    for gi, g in enumerate(guests):
        if pl.value(u[gi]) > 0.5:
            unassigned.append(g)
            continue
        for ri, r in enumerate(rooms):
            if pl.value(x[(gi, ri)]) > 0.5:
                assignment[g["name"]] = rooms[ri]["id"]
                break

    if unassigned:
        st.subheader(f"Niet geplaatst {len(pd.DataFrame(unassigned))}/{len(guests)} gasten")
        st.dataframe(pd.DataFrame(unassigned))

    if not assignment:
        st.warning("Geen toewijzingen gevonden.")
        return

    # ---------- DataFrame + features ----------
    guest_map = {g["name"]: g for g in guests}
    data = [{
        "Gast": name,
        "Kamer": str(room),
        "Start": guest_map[name]["start"],
        "Eind":  guest_map[name]["end"],
        "Need":  guest_map[name]["need"],
        "Pref_floor":  guest_map[name]["pref_floor"],
        "Near_elev":  guest_map[name]["near_elev"],
        "Shuffled":  guest_map[name]["shuffled"],
         "Geplande room":  guest_map[name]["geplande_room"],
        
    } for name, room in assignment.items()]
    df = pd.DataFrame(data)

    # Afgeleide kolommen
    df["Floor"] = df["Kamer"].str[0].astype(int)
    df["Room_type"] = df["Kamer"].str[-1].astype(int).map(lambda x: "single" if x % 2 == 1 else "double")
    df["Happy_floor"] = df["Pref_floor"] == df["Floor"]
    df["Happy_room"] = df["Need"] == df["Room_type"]

    # Satisfactiescore (voorbeeld: alleen floor afstand)
    df["Satisfaction_score"] = 100 - ((df["Floor"] - df["Pref_floor"]).abs() / 3) * 100

    st.metric("Gemiddelde satisfactiescore", f"{df['Satisfaction_score'].mean():.2f}")
    # st.dataframe(df.sort_values(["Kamer", "Start"]))

    # ---------- Timeline ----------
    df["Start"] = pd.to_datetime(df["Start"])
    df["Eind"] = pd.to_datetime(df["Eind"])
    # Maak einde iets eerder zodat blokken aansluiten
    df["Eind_plot"] = df["Eind"] - pd.Timedelta(seconds=1)

    # Alleen gebruikte kamers tonen en netjes ordenen
    df["Kamer"] = df["Kamer"].astype(str)
    df = df.sort_values(by=["Kamer"])

    rooms_order = df["Kamer"].unique().tolist()
    
    # Maak een kolom voor de kleurcategorie
    def assign_color(row):
        if row["Happy_floor"]:
            return "green"
        elif not row["Happy_floor"] and row["Shuffled"]:
            return "orange"
        else:
            return "red"

    df["ColorCategory"] = df.apply(assign_color, axis=1)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Eind_plot",
        y="Kamer",
        color="ColorCategory",
        text="Gast",
        hover_data={
            "Gast": True, "Kamer": True, "Geplande room": True,
            "Start": True, "Eind": True,
            "Need": True, "Pref_floor": True, "Near_elev": True,
            "Happy_floor": True, "Happy_room": True, "Shuffled": True
        },
        color_discrete_map={
            "green": "green",
            "orange": "orange",
            "red": "red"
        }
    )

    # Border (outline) om elk blok
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

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # if st.button("GO"):
    main()
