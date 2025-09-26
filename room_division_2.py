# Simulate hotel bookings with prebooked guests until Sep 1 and walk-ins from Sep 1, 2025.
import random
from datetime import date, timedelta
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
import streamlit as st



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import pandas as pd


import plotly.express as px


# ------------------- Helpers -------------------
def daterange(start: date, end: date):
    d = start
    while d < end:
        yield d
        d += timedelta(days=1)

def overlaps(a_start, a_end, b_start, b_end):
    # stay ranges are [start, end) with end as checkout date
    return a_start < b_end and b_start < a_end

def generate_prebooked(rooms: List[Dict[str, Any]], start: date, end: date) -> List[Dict[str, Any]]:
    """Generate back-to-back prebooked guests per room covering the whole period."""
    guests = []
    gid = 1
    for r in rooms:
        cur = start
        while cur < end:
            stay_length = rng.randint(2, 6)  # nights
            g = {
                "name": f"G{gid}",
                "need": rng.choice(["single", "double"]),
                "pref_floor": rng.randint(1, 4),
                "near_elev": rng.choice([True, False]),
                "start": cur,
                "end": min(cur + timedelta(days=stay_length), end),
                "assigned_room": r["id"],     # pre-assigned
                "planned_room": r["id"],
                "prebooked": True,
                "walk_in": False,
                "moved": False,
            }
            guests.append(g)
            cur = g["end"]  # next guest starts when this one checks out
            gid += 1
    return guests

def add_walkins(guests: List[Dict[str, Any]], rooms: List[Dict[str, Any]], start: date, end: date, ratio: float) -> List[Dict[str, Any]]:
    """Create additional walk-in guests proportional to prebooked volume and try to assign them."""
    total_pre = len(guests)
    n_walkin = max(1, int(total_pre * ratio))
    gid_start = max(int(g["name"][1:]) for g in guests) + 1 if guests else 1
    room_by_id = {r["id"]: r for r in rooms}

    # Build current schedule dict per room -> list of (start, end, guest_index)
    schedule = defaultdict(list)
    for i, g in enumerate(guests):
        schedule[g["assigned_room"]].append((g["start"], g["end"], i))
    for rid in schedule:
        schedule[rid].sort()

    def room_free(rid, s, e):
        for (as_, ae_, _) in schedule.get(rid, []):
            if overlaps(as_, ae_, s, e):
                return False
        return True

    # Quick index of guests by date to check "cannot move if in-house or within 1 day"
    def locked_guest_indices(on_date: date):
        locked = set()
        for idx, g in enumerate(guests):
            if g["start"] <= on_date < g["end"]:
                locked.add(idx)  # in-house
            elif g["start"] == on_date + timedelta(days=1):
                locked.add(idx)  # arriving within 1 day
        return locked

    def try_assign_walkin(wg: Dict[str, Any]) -> bool:
        # 1) Try any available room matching need first
        candidate_rids = [r["id"] for r in rooms if r["type"] == wg["need"]]
        # Sort by preference closeness
        def score(rid):
            r = room_by_id[rid]
            s = 0
            s += 1 if r["floor"] == wg["pref_floor"] else 0
            s += 1 if r["elev"] == wg["near_elev"] else 0
            return -s  # sort ascending, higher match first
        candidate_rids.sort(key=score)

        # a) direct fit
        for rid in candidate_rids:
            if room_free(rid, wg["start"], wg["end"]):
                wg["assigned_room"] = rid
                schedule[rid].append((wg["start"], wg["end"], len(guests)))
                schedule[rid].sort()
                return True

        # b) Try to reshuffle future guests more than 1 day away
        lock = locked_guest_indices(wg["start"])
        # Build list of conflicting stays by candidate rooms
        for rid in candidate_rids:
            # Try to move conflicting future guests to other rooms of same type to make space
            conflicts = [t for t in schedule[rid] if overlaps(t[0], t[1], wg["start"], wg["end"])]
            if not conflicts:
                continue
            # Check if all conflicts are movable
            movable = [idx for (_, _, idx) in conflicts if idx not in lock]
            if len(movable) != len(conflicts):
                continue  # someone is locked, cannot use this room

            # Try to rehome conflicts one by one
            moved_list = []
            ok = True
            for (_, _, idx) in conflicts:
                gconf = guests[idx]
                # Remove from its schedule temporarily
                schedule[rid] = [t for t in schedule[rid] if t[2] != idx]

                # find other room of same need
                placed = False
                for alt in candidate_rids:
                    if alt == rid:
                        continue
                    if room_free(alt, gconf["start"], gconf["end"]):
                        schedule[alt].append((gconf["start"], gconf["end"], idx))
                        schedule[alt].sort()
                        moved_list.append((idx, gconf["assigned_room"], alt))
                        gconf["assigned_room"] = alt
                        gconf["moved"] = True
                        placed = True
                        break
                if not placed:
                    ok = False
                    # rollback any moves
                    for midx, prev, new in moved_list:
                        schedule[new] = [t for t in schedule[new] if t[2] == midx] + [t for t in schedule[new] if t[2] != midx]
                        schedule[new] = [t for t in schedule[new] if t[2] != midx]  # remove
                        schedule[prev].append((guests[midx]["start"], guests[midx]["end"], midx))
                        schedule[prev].sort()
                        guests[midx]["assigned_room"] = prev
                        guests[midx]["moved"] = False
                    # put original conflicts back
                    for (s,e,idx2) in conflicts:
                        if (s,e,idx2) not in schedule[rid]:
                            schedule[rid].append((s,e,idx2))
                            schedule[rid].sort()
                    break
            if ok:
                # place walk-in in rid
                wg["assigned_room"] = rid
                schedule[rid].append((wg["start"], wg["end"], len(guests)))
                schedule[rid].sort()
                # mark moved_list already reflected
                return True
        return False

    new_walkins = []
    for k in range(n_walkin):
        stay_length = rng.randint(1, 4)
        # arrivals between start and end-1
        arr = start + timedelta(days=rng.randint(0, (end - start).days - 1))
        wg = {
            "name": f"G{gid_start + k}W",
            "need": rng.choice(["single", "double"]),
            "pref_floor": rng.randint(1, 4),
            "near_elev": rng.choice([True, False]),
            "start": arr,
            "end": min(arr + timedelta(days=stay_length), end),
            "assigned_room": None,
            "planned_room": None,
            "prebooked": False,
            "walk_in": True,
            "moved": False,
        }
        placed = try_assign_walkin(wg)
        wg["accepted"] = placed
        new_walkins.append(wg)
        if placed:
            guests.append(wg)

    return guests, new_walkins

def satisfaction(g, room_by_id):
    r = room_by_id[g["assigned_room"]] if g["assigned_room"] else None
    score = 0
    if r:
        score += 1 if r["type"] == g["need"] else 0
        score += 1 if r["floor"] == g["pref_floor"] else 0
        score += 1 if r["elev"] == g["near_elev"] else 0
    return score

def simulate(shuffle_ratio: float = 0.25):
    pre = generate_prebooked(rooms, start_period, end_period + timedelta(days=1))
    all_guests, walkins = add_walkins(pre, rooms, start_period, end_period + timedelta(days=1), shuffle_ratio)
    room_by_id = {r["id"]: r for r in rooms}
    # Build dataframe
    rows = []
    for g in all_guests:
        rows.append({
            "Gast": g["name"],
            "Walk_in": g["walk_in"],
            "Prebooked": g["prebooked"],
            "Start": g["start"],
            "Eind": g["end"],
            "Nachten": (g["end"] - g["start"]).days,
            "Need": g["need"],
            "Pref_floor": g["pref_floor"],
            "Near_elev": g["near_elev"],
            "Planned_room": g["planned_room"],
            "Assigned_room": g["assigned_room"],
            "Moved": g["moved"],
            "Satisfaction_0_3": satisfaction(g, room_by_id),
            "Happy_need": room_by_id[g["assigned_room"]]["type"] == g["need"] if g["assigned_room"] else False,
            "Happy_floor": room_by_id[g["assigned_room"]]["floor"] == g["pref_floor"] if g["assigned_room"] else False,
            "Happy_elev": room_by_id[g["assigned_room"]]["elev"] == g["near_elev"] if g["assigned_room"] else False,
        })
    df = pd.DataFrame(rows).sort_values(["Start", "Assigned_room", "Gast"]).reset_index(drop=True)

    # Key metrics
    total_nights = sum((g["end"] - g["start"]).days for g in all_guests if g["assigned_room"])
    hotel_nights = len(list(daterange(start_period, end_period + timedelta(days=1)))) * len(rooms)
    occupancy = total_nights / hotel_nights if hotel_nights else 0

    accepted_walkins = sum(1 for w in walkins if w["accepted"])
    acceptance_rate = accepted_walkins / len(walkins) if walkins else 0

    moves = sum(1 for g in all_guests if g["moved"])

    metrics = {
        "total_guests": len(all_guests),
        "prebooked": sum(g["prebooked"] for g in all_guests),
        "walkins_created": len(walkins),
        "walkins_accepted": accepted_walkins,
        "walkin_acceptance_rate": round(100 * acceptance_rate, 1),
        "moves": moves,
        "occupancy_rate": round(100 * occupancy, 1),
        "satisfaction_avg": round(df["Satisfaction_0_3"].mean(), 2) if not df.empty else 0.0,
    }

    return df, metrics


# Rerun with gaps and cancellations so walk-ins have a chance
def generate_prebooked_with_gaps(rooms, start, end, gap_p=0.3, cancel_p=0.1):
    guests = []
    gid = 1
    for r in rooms:
        cur = start
        while cur < end:
            stay_length = rng.randint(2, 6)
            g = {
                "name": f"G{gid}",
                "need": rng.choice(["single", "double"]),
                "pref_floor": rng.randint(1, 4),
                "near_elev": rng.choice([True, False]),
                "start": cur,
                "end": min(cur + timedelta(days=stay_length), end),
                "assigned_room": r["id"],
                "planned_room": r["id"],
                "prebooked": True,
                "walk_in": False,
                "moved": False,
            }
            # possible cancellation
            if rng.random() > cancel_p:
                guests.append(g)
                cur = g["end"]
            else:
                # canceled, no occupancy
                cur = min(cur + timedelta(days=stay_length), end)
            # possible gap night
            if rng.random() < gap_p:
                gap = rng.choice([0,1])
                cur = min(cur + timedelta(days=gap), end)
            gid += 1
    return guests

def simulate_variant(shuffle_ratio: float = 0.25, gap_p=0.3, cancel_p=0.1):

    """_summary_


    Returns:
        _type_: _description_
    """

    pre = generate_prebooked_with_gaps(rooms, start_period, end_period + timedelta(days=1), gap_p, cancel_p)
    all_guests, walkins = add_walkins(pre, rooms, start_period, end_period + timedelta(days=1), shuffle_ratio)
    room_by_id = {r["id"]: r for r in rooms}
    rows = []
    for g in all_guests:
        rows.append({
            "Gast": g["name"],
            "Walk_in": g["walk_in"],
            "Prebooked": g["prebooked"],
            "Start": g["start"],
            "Eind": g["end"],
            "Nachten": (g["end"] - g["start"]).days,
            "Need": g["need"],
            "Pref_floor": g["pref_floor"],
            "Near_elev": g["near_elev"],
            "Planned_room": g["planned_room"],
            "Assigned_room": g["assigned_room"],
            "Moved": g["moved"],
            "Satisfaction_0_3": (1 if rooms[0] else 0),  # placeholder, recalc below
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Satisfaction_0_3"] = [
            (1 if (rid and next(r for r in rooms if r["id"]==rid)["type"] == need) else 0)
            + (1 if (rid and next(r for r in rooms if r["id"]==rid)["floor"] == pf) else 0)
            + (1 if (rid and next(r for r in rooms if r["id"]==rid)["elev"] == ne) else 0)
            for rid, need, pf, ne in zip(df["Assigned_room"], df["Need"], df["Pref_floor"], df["Near_elev"])
        ]
    total_nights = 0
    for _, g in df.iterrows():
        if pd.notna(g["Assigned_room"]):
            total_nights += (g["Eind"] - g["Start"]).days
    hotel_nights = len(list(daterange(start_period, end_period + timedelta(days=1)))) * len(rooms)
    occupancy = total_nights / hotel_nights if hotel_nights else 0
    accepted_walkins = sum(1 for w in walkins if w.get("accepted"))
    metrics = {
        "total_guests": len(df),
        "prebooked": int(df["Prebooked"].sum()) if not df.empty else 0,
        "walkins_created": len(walkins),
        "walkins_accepted": accepted_walkins,
        "walkin_acceptance_rate": round(100 * accepted_walkins / len(walkins), 1) if walkins else 0.0,
        "moves": int(df["Moved"].sum()) if not df.empty else 0,
        "occupancy_rate": round(100 * occupancy, 1),
        "satisfaction_avg": round(df["Satisfaction_0_3"].mean(), 2) if not df.empty else 0.0,
    }
    return df.sort_values(["Start","Assigned_room","Gast"]).reset_index(drop=True), metrics



def plot_chart_mpl(df_sim2):
    # Keep only rows with an assigned room
    gdf = df_sim2.dropna(subset=["Assigned_room"]).copy()

    # Sort by room then start date
    gdf = gdf.sort_values(["Assigned_room","Start"]).reset_index(drop=True)

    # Prepare room order
    rooms_sorted = sorted(gdf["Assigned_room"].unique())
    room_to_y = {rid: i for i, rid in enumerate(rooms_sorted)}

    # Convert to durations in days for plotting
    gdf["start_num"] = mdates.date2num(gdf["Start"])
    gdf["end_num"]   = mdates.date2num(gdf["Eind"])
    gdf["duration"]  = gdf["end_num"] - gdf["start_num"]

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Choose hatches to distinguish prebooked vs walk-in without setting colors
    for _, row in gdf.iterrows():
        y = room_to_y[row["Assigned_room"]]
        hatch = "//" if row["Walk_in"] else ""
        ax.broken_barh([(row["start_num"], row["duration"])], (y - 0.4, 0.8), hatch=hatch)

    # Formatting
    ax.set_yticks(range(len(rooms_sorted)))
    ax.set_yticklabels(rooms_sorted)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kamer")
    ax.set_title("Gantt chart kamerbezetting • September 2025 • Variant")

    # X axis as dates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

    # Grid to improve readability
    ax.grid(True, axis="x", linestyle=":", linewidth=0.7)

    # Legend using proxy artists
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(hatch="", label="Prebooked"),
        Patch(hatch="//", label="Walk-in"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    # out_path = "/mnt/data/hotel_gantt_sep2025_variant.png"
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # out_path

    st.pyplot(fig)

def plot_chart_plotly(df):
 # ---------- Timeline ----------
    df["Start"] = pd.to_datetime(df["Start"])
    df["Eind"] = pd.to_datetime(df["Eind"])
    # Maak einde iets eerder zodat blokken aansluiten
    df["Eind_plot"] = df["Eind"] - pd.Timedelta(seconds=1)

    # Alleen gebruikte kamers tonen en netjes ordenen
    df["Kamer"] = df["Assigned_room"].astype(str)
    df = df.sort_values(by=["Kamer"])

    rooms_order = df["Kamer"].unique().tolist()
    # Afgeleide kolommen
    df["Floor"] = df["Kamer"].str[0].astype(int)
    df["Room_type"] = df["Kamer"].str[-1].astype(int).map(lambda x: "single" if x % 2 == 1 else "double")
    df["Happy_floor"] = df["Pref_floor"] == df["Floor"]
    df["Happy_room"] = df["Need"] == df["Room_type"]    
    df["Shuffled"] = df["Walk_in"]  # Walk-in gasten zijn de 'shuffled' gasten
    # Satisfactiescore (voorbeeld: alleen floor afstand)
    df["Satisfaction_score"] = 100 - ((df["Floor"] - df["Pref_floor"]).abs() / 3) * 100


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


def main():
    # ------------------- Setup -------------------
    

    # Run one simulation with 25% walk-ins
    df_sim, metrics = simulate(shuffle_ratio=0.25)

    st.write(df_sim)
    st.write(metrics)


    # plot_chart_mpl(df_sim)
    plot_chart_plotly(df_sim)
    df_sim2, metrics2 = simulate_variant(shuffle_ratio=0.35, gap_p=0.4, cancel_p=0.15)


    st.write(df_sim2)
    st.write(metrics2)

    # plot_chart_mpl(df_sim2)
    plot_chart_plotly(df_sim2)

if __name__ == "__main__":
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

    start_period = date(2025, 9, 1)
    end_period = date(2025, 9, 30)  # exclusive in our loops

    rng = random.Random(42)  # deterministic for reproducibility

    st.title("Kamerindeling hotel met prebooked en walk-in gasten")
    main()
