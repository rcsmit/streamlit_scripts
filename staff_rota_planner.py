"""
Staff rota planner.

Generates a weekly staff schedule with Google OR-Tools CP-SAT, using the same
kind of constraints found in a real hospitality rota: fixed vacation/comp-day
entries (FERIE / RECUPERO), automatically-decided rest days (RIPOSO) subject
to a minimum-rest-days rule, a maximum-consecutive-working-days rule, split
shifts (two blocks in one day with a minimum break in between), an exact
weekly contract-hours target per person, hour-by-hour minimum staffing
coverage (defined independently for every day of the week, with any hours
beyond the minimum steered towards the morning rather than the afternoon),
staff relationships ("these two must always work the same shift", "these two
should work complementary/alternating hours"), and individual shift or
day-off requests.

Dependencies: streamlit, pandas, ortools

---
Provided "AS IS", without warranty of any kind, express or implied, including
but not limited to warranties of merchantability, fitness for a particular
purpose, and non-infringement. This tool produces a mathematically-consistent
schedule given the rules you configure - it does not know your local labour
law, collective agreements, or anything you haven't told it. Always review a
generated schedule before relying on it operationally.
---
"""

from __future__ import annotations

# version : 20260730-120000 - Initial version: CP-SAT rota model, staff/shift/fixed-day/coverage editors, results grid, CSV + Excel export
# version : 20260730-153000 - Per-day coverage defaults, same-shift & complementary staff rules, shift/day-off request system
# version : 20260730-171500 - Max closing shifts/week per person, optional fixed 8h split (2x4h, 1-2h break) + 2 days off contract type
# version : 20260730-182000 - Added Day x Hour "staff working per hour" table to the results tab
# version : 20260730-193000 - Italian staff names, 40h/split-shifts for everyone, min-4-people 9-13 & 15-20 coverage rule, removed Excel export (xlsxwriter dropped)
# version : 20260730-203000 - Weekly contract hours now a hard exact constraint (not a soft target); extra coverage beyond the minimum is steered towards 10-15h rather than the afternoon
# version : 20260730-211500 - Optional max-break-between-split-shifts cap; sidebar "reset all to code defaults" button
# version : 20260730-221500 - Shift catalog capped at 6h max (removed 8h shifts, enforced on edit); Min rest days/week restricted to a 1-or-2 choice
# version : 20260730-230000 - Added a "Quick set: days off for everyone" bulk control in the Staff tab (1 or 2, applied to all staff at once)
# version : 20260730-234500 - With exactly 1 day off + 40h target: enforce 3x8h + 2x6h + 1x4h day mix. Added a hard "maximum staff per hour" table (Coverage tab)
# version : 20260731-001500 - Coverage defaults: 9h min lowered to 3; max-staff-per-hour set to 2/3/1/1 for 8h/9h/20h/21h (10-19h unchanged at 8)
# version : 20260731-010000 - Added an "About" tab (first tab) with the manual/blog-post write-up, including a bordered "which engine and why" callout
# version : 20260731-013000 - Removed the overstaffed-hours badge (not useful); understaffed-hours expander now auto-opens and highlights the gap rows
# version : 20260731-020000 - Fixed the FERIE/RECUPERO infeasibility: weekly hours target now drops by 8h per FERIE/RECUPERO day instead of clamping to max-possible
# version : 20260731-022000 - Understaffed-hours expander/table now only shown when gaps exist; otherwise a plain "All hours comply" message
# version : 20260731-023000 - CSV export now matches the on-screen grid (names as rows, days as columns) instead of a long one-row-per-day format
# version : 20260731-030000 - Fixed a bug where Fixed-days/Rules/Requests edits (e.g. setting RIPOSO) could occasionally revert: sync functions rebuilt those tables from scratch on every rerun of the whole app, not just when the staff list changed
# version : 20260731-040000 - Added an "Advanced" tab (last tab): technical reference for every config constant, an explanation of the objective weights, and an algorithm-details box
# version : 20260731-050000 - Advanced tab now also explains the coverage min/max overrides. Performance: dict-based lookups instead of repeated .loc calls, cached shifts-covering-hour table, itertuples instead of iterrows. Added an input-validation pass (duplicate names, bad hours, bad shifts, min>max coverage, duplicate pairs) before solving
# version : 20260731-053000 - Requests tab's ShiftCode dropdown now shows "CODE (start-end)" (e.g. "M6 (9-13)"); sidebar now expanded by default
# version : 20260731-060000 - Sidebar reverted to collapsed by default. Coverage tab: added an Arrivals & Departures table that can derive the minimum-coverage grid (departures->morning, arrivals->evening, positive departures-arrivals surplus->afternoon cleaning window)
# version : 20260731-063000 - Arrivals & Departures section moved to the top of the Coverage tab; derivation simplified to day/total*100 (vs. an even 1/7 share) scaling a single baseline staff count, replacing the per-staff-member ratios
# version : 20260731-070000 - The existing minimum/maximum coverage tables are now always leading during derivation: the ratio-derived value is clamped to never go below the existing minimum or above the existing maximum, so a 0-arrivals/departures day keeps its configured floor instead of dropping
# version : 20260731-071500 - Arrivals & Departures table now defaults to 20 for every day/metric instead of 0
# version : 20260731-074500 - Removed the manual "Baseline staff" field - each window's baseline is now auto-derived from the current average of its own cells (a perfectly even arrivals/departures week leaves coverage unchanged). Button renamed to "Apply". Added a "verify before using" disclaimer under the schedule and an AS-IS disclaimer in the module docstring
# version : 20260731-081500 - Fixed the Coverage tab's Quick-fill buttons not clearing their editor's cached widget state before rerunning (same class of bug as the earlier RIPOSO issue, now also affecting min/max coverage edits). "Limit max break" checkbox now backed by session state with a real default, resettable from the sidebar. Understaffed-hours message reworded to "X hour blocks are understaffed"
# version : 20260731-090000 - Arrivals & Departures: Departures now the first row, Arrivals second. Every data_editor table in the app is now wrapped in an st.form with an explicit Save/Apply button, so editing a cell no longer triggers a full-app rerun (and the "enter it twice" symptom) on every keystroke - edits are batched locally until the button is pressed
# version : 20260731-093000 - Fixed limit_max_break default (True). Every Save/Apply button now toasts a confirmation on submit, distinguishing a genuine change from clicking Save with nothing new (the honest limit: st.form can't show a live "unsaved changes" state before submit, since it doesn't rerun the script while you're still editing)
# version : 20260731-101500 - DEFAULT_STAFF now carries a per-person FixedDays dict (seeds the Fixed-days grid at startup, supports multiple pre-set days off). New "Days off together" preference (checkbox, only meaningful with 2 days off/week): soft-rewards the 2 rest days being adjacent, or penalises adjacency if unchecked - weighted by WEIGHT_DAYS_OFF_TOGETHER
# version : 20260731-104500 - Added a "deploying your own copy" box to the Advanced tab: GitHub repo, editing constants, requirements.txt, deploying via share.streamlit.io
# version : 20260731-110000 - Fixed a KeyError crash for existing sessions/saved files missing newer staff columns (e.g. DaysOffTogether): staff_df schema is now migrated on every load via ensure_staff_columns. Renamed "Kader" to "Box" in the About tab for consistency with the Advanced tab
# version : 20260731-112000 - Filled in each default staff member's FixedDays with their given day off (RIPOSO): Alessia/Chiara/Ilaria/Nadia->Friday, Bruno/Federico->Wednesday, Davide/Lorenzo->Tuesday, Elena/Giulia->Thursday, Marco->Monday
# version : 20260731-120000 - A "Prefer shift"/"Avoid shift" request now overrules a RIPOSO fixed day (opens that day back up for scheduling) - FERIE/RECUPERO remain untouchable since those are real absences, not just a scheduling default. Re-fixed the About-tab ordering regression (now first again)
# version : 20260731-124500 - Added DEFAULT_ARRIVALS_DEPARTURES and DEFAULT_REQUESTS dicts (both empty by default, so behaviour is unchanged unless filled in) to pre-seed those two tables. Default week-start date is now the upcoming Monday strictly after today, not the current week's Monday
# version : 20260731-130000 - Fixed DEFAULT_REQUESTS silently failing to apply: bare shift codes (e.g. "M2") are now auto-translated to the "CODE (start-end)" format the Requests dropdown actually requires, without changing DEFAULT_REQUESTS itself
# version : 20260731-133000 - If DEFAULT_ARRIVALS_DEPARTURES has been customised away from the flat default, the minimum-coverage table is now derived from it automatically at startup, instead of requiring a manual "Apply" click just to reflect your own configured defaults
current_version = "20260731-133000"

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

# ============================================================================
# CONFIGURATION BLOCK - all tunable parameters live here
# ============================================================================


DAYS: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

OPEN_HOUR: int = 8
CLOSE_HOUR: int = 22
HOURS: list[int] = list(range(OPEN_HOUR, CLOSE_HOUR))  # hour blocks 8-9 .. 21-22

FIXED_STATUS_OPTIONS: list[str] = ["", "RIPOSO", "FERIE", "RECUPERO"]
REQUEST_TYPE_OPTIONS: list[str] = ["Prefer day off", "Prefer shift", "Avoid shift"]
PRIORITY_OPTIONS: list[str] = ["Low", "Medium", "High"]
PRIORITY_WEIGHTS: dict[str, int] = {"Low": 1, "Medium": 3, "High": 6}
REST_DAYS_OPTIONS: list[int] = [1, 2]  # everyone gets either 1 or 2 fixed days off per week

CONTRACT_TYPE_FLEXIBLE: str = "Flexible"
CONTRACT_TYPE_FIXED_SPLIT: str = "Fixed 8h split (2x4h, 1-2h break) + 2 days off/week"
CONTRACT_TYPE_OPTIONS: list[str] = [CONTRACT_TYPE_FLEXIBLE, CONTRACT_TYPE_FIXED_SPLIT]
FIXED_SPLIT_SHIFT_HOURS: int = 4          # each half of the fixed-split day
FIXED_SPLIT_MIN_BREAK: int = 1            # minimum break between the two 4h blocks
FIXED_SPLIT_MAX_BREAK: int = 2            # maximum break between the two 4h blocks
FIXED_SPLIT_MIN_REST_DAYS: int = 2        # the "2 days off/week" bundled into this contract type

DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK: int = 1  # 0 = unlimited. A "closing" shift is whichever
                                                # shift(s) in the catalog end at the latest hour.

DEFAULT_MIN_BREAK_HOURS: int = 1          # minimum gap between two blocks of a split shift
DEFAULT_MAX_BREAK_HOURS_OPTION: int = 2   # default value shown when "limit max break" is switched on
DEFAULT_MAX_DAILY_HOURS: int = 9          # hard cap on paid hours per person per day
HOURS_PER_FERIE_RECUPERO_DAY: int = 8     # each FERIE/RECUPERO day lowers that week's hours target by this much
DEFAULT_SOLVER_TIME_LIMIT_SEC: int = 20
STRICT_WEEKLY_HOURS: bool = True  # if True, everyone's weekly hours must equal ContractHours exactly (hard constraint)

# objective weights: understaffing is penalised far more than everything else,
# so the solver only sacrifices coverage when there is truly no other option.
WEIGHT_UNDERSTAFF: int = 200
WEIGHT_OVERSTAFF: int = 5          # default overstaffing penalty, outside the morning/afternoon windows below
REQUEST_WEIGHT_SCALE: int = 10     # multiplies PRIORITY_WEIGHTS for shift/day-off requests
WEIGHT_DAYS_OFF_TOGETHER: int = 15  # soft preference weight for someone with 2 days off/week
                                     # wanting them consecutive (or, if unchecked, kept apart)

# With exact weekly hours enforced, any hours beyond the minimum coverage need
# have to go *somewhere*. These two windows let extra coverage be steered
# towards the morning rather than the afternoon: overstaffing is cheap (weight
# 1) between 10-15h, and deliberately expensive (weight 10) in the afternoon.
MORNING_PRIORITY_HOURS: set[int] = set(range(10, 15))   # 10,11,12,13,14
AFTERNOON_HOURS: set[int] = set(range(15, 20))          # 15,16,17,18,19
WEIGHT_OVERSTAFF_MORNING: int = 1
WEIGHT_OVERSTAFF_AFTERNOON: int = 10


def overstaff_weight_for_hour(hour: int) -> int:
    if hour in MORNING_PRIORITY_HOURS:
        return WEIGHT_OVERSTAFF_MORNING
    if hour in AFTERNOON_HOURS:
        return WEIGHT_OVERSTAFF_AFTERNOON
    return WEIGHT_OVERSTAFF


MAX_SHIFT_LENGTH_HOURS: int = 6  # hard cap on how long any single shift block can be


# Coverage rule: minimum 4 people between 9-13 and between 15-20, every day of
# the week. Hours outside those windows default to a lower baseline - edit
# freely per day/hour in the Coverage tab. Hour 9 is overridden down to 3
# (rather than the general peak value of 4) to match the max-staff cap below.
_BASELINE_MIN = 1
_PEAK_MIN = 4
_PEAK_HOURS = set(range(9, 13)) | set(range(15, 20))  # 9,10,11,12 and 15,16,17,18,19
_MIN_OVERRIDES: dict[int, int] = {9: 3}
DEFAULT_HOURLY_MIN_BY_DAY: dict[str, dict[int, int]] = {
    day: {
        hour: _MIN_OVERRIDES.get(hour, _PEAK_MIN if hour in _PEAK_HOURS else _BASELINE_MIN)
        for hour in HOURS
    }
    for day in DAYS
}

# Hard ceiling on staff per hour (hard constraint, not just a soft overstaffing
# penalty). Generous by default (8) so it rarely binds, except for a few hours
# that are explicitly tightened: quiet open (8h), a tighter 9h cap matching the
# 9h minimum above, and a quiet close (20h, 21h).
DEFAULT_MAX_STAFF_PER_HOUR: int = 8
_MAX_STAFF_OVERRIDES: dict[int, int] = {8: 2, 9: 3, 20: 1, 21: 1}

# Arrivals/departures-driven coverage: three windows reusing the same
# morning/afternoon/evening split as the peak-hours coverage rule above.
# - Morning (checkout window) coverage is driven by departures.
# - Evening (check-in window) coverage is driven by arrivals.
# - Afternoon (the gap in between) only gets a coverage bump when there are
#   more departures than arrivals that day (rooms need cleaning before the
#   next arrivals; if arrivals >= departures there's no such surplus).
ARRIVAL_DEPARTURE_MORNING_HOURS: set[int] = set(range(9, 13))    # 9,10,11,12
ARRIVAL_DEPARTURE_AFTERNOON_HOURS: set[int] = set(range(13, 15))  # 13,14 (the cleaning gap)
ARRIVAL_DEPARTURE_EVENING_HOURS: set[int] = set(range(15, 20))    # 15,16,17,18,19
DEFAULT_ARRIVALS_DEPARTURES_PER_DAY: int = 20  # starting value for every day in the Arrivals & Departures table

# Optional per-day overrides, e.g. {"Monday": {"Arrivals": 25, "Departures": 18}, ...}.
# Only the days/metrics you fill in are used; anything left out falls back to
# DEFAULT_ARRIVALS_DEPARTURES_PER_DAY. Leave this empty ({}) to keep the current
# flat default for every day.
# DEFAULT_ARRIVALS_DEPARTURES: dict[str, dict[str, int]] = {}
DEFAULT_ARRIVALS_DEPARTURES: dict[str, dict[str, int]] = { "Monday": {"Arrivals": 34, "Departures": 16},
                                                            "Tuesday": {"Arrivals": 18, "Departures": 24},
                                                            "Wednesday": {"Arrivals": 25, "Departures": 26},
                                                            "Thursday": {"Arrivals": 23, "Departures": 24},
                                                            "Friday": {"Arrivals": 14, "Departures": 30},
                                                            "Saturday": {"Arrivals": 35, "Departures": 45},     
                                                            "Sunday": {"Arrivals": 26, "Departures": 14},}


# Optional pre-set requests, e.g.
# [{"Name": "C", "Day": "Friday", "RequestType": "Prefer day off", "ShiftCode": "", "Priority": "High"}, ...]
# Leave this empty ([]) to keep the Requests tab starting blank, as now.
# DEFAULT_REQUESTS: list[dict] = []
DEFAULT_REQUESTS: list[dict] = [{"Name": "C", "Day": "Monday", "RequestType": "Prefer shift", "ShiftCode": "M2", "Priority": "High"}, 
    {"Name": "C", "Day": "Monday", "RequestType": "Prefer shift", "ShiftCode": "A3", "Priority": "High"}, 
    {"Name": "C", "Day": "Wednesday", "RequestType": "Prefer shift", "ShiftCode": "A9", "Priority": "High"}, 
    {"Name": "C", "Day": "Thursday", "RequestType": "Prefer shift", "ShiftCode": "M2", "Priority": "High"}, 
    {"Name": "C", "Day": "Thursday", "RequestType": "Prefer shift", "ShiftCode": "A6", "Priority": "High"}, ]
DEFAULT_SHIFT_CATALOG: list[tuple[str, int, int]] = [
    ("M1", 8, 12), ("M2", 9, 13), ("M3", 9, 15), ("M4", 8, 14),
    ("M5", 10, 14), ("M6", 10, 16), ("M7", 11, 15),
    ("A1", 13, 17), ("A2", 13, 19), ("A3", 14, 18), ("A4", 14, 20),
    ("A6", 15, 19), ("A7", 15, 20),("A8", 16, 19),("A9", 16, 20), ("10", 16, 22),
    ("E1", 18, 22),
]

# Names below follow the classic 21-letter Italian alphabet order (which has no
# native words/names starting with H, so it's skipped): A, B, C, D, E, F, G, I, L, M, N.
DEFAULT_STAFF: list[dict] = [
    {"Name": "C", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Tuesday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "T", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Wednesday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "So", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Friday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "I", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Tuesday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "M", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Thursday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "E", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Wednesday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "F", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Thursday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "Ma", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Friday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "R", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Tuesday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "D", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Monday": "RIPOSO"}, "DaysOffTogether": True},
    {"Name": "Si", "ContractHours": 40, "MinRestDays": 1, "MaxConsecDays": 6, "SplitAllowed": True,
     "ContractType": CONTRACT_TYPE_FLEXIBLE, "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
     "FixedDays": {"Tuesday": "RIPOSO"}, "DaysOffTogether": True},
]

# Default staff relationships (both fully editable in the "Rules" tab):
# - "same shift" pairs must always work identical shift blocks on days neither
#   has a fixed FERIE/RECUPERO entry.
# - "complementary" pairs are softly discouraged from working the same hour at
#   the same time (higher weight = stronger preference for alternating cover).
#   Sitemanager & Temaleader
DEFAULT_SAME_SHIFT_PAIRS: list[dict] = [
    {"Name1": "C", "Name2": "Si"},
]
DEFAULT_COMPLEMENTARY_PAIRS: list[dict] = [
    {"Name1": "C", "Name2": "T", "Weight": 15},
]

# ============================================================================
# DATA BUILDERS
# ============================================================================

def build_default_shift_df() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_SHIFT_CATALOG, columns=["Code", "Start", "End"])


def build_default_staff_df() -> pd.DataFrame:
    # FixedDays is deliberately excluded here: it's a one-time seed for the Fixed-days
    # grid (see build_default_fixed_df below), not an ongoing editable column - a
    # dict-per-cell column doesn't serialise through st.data_editor anyway.
    rows = [{k: v for k, v in person.items() if k != "FixedDays"} for person in DEFAULT_STAFF]
    return pd.DataFrame(rows)


STAFF_COLUMN_DEFAULTS: dict[str, object] = {
    "ContractHours": 30,
    "MinRestDays": 1,
    "MaxConsecDays": 6,
    "SplitAllowed": True,
    "ContractType": CONTRACT_TYPE_FLEXIBLE,
    "MaxClosingShiftsPerWeek": DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK,
    "DaysOffTogether": True,
}


def ensure_staff_columns(staff_df: pd.DataFrame) -> pd.DataFrame:
    """Add any staff column that's been introduced since a session (or an old saved
    copy of this app) was last touched, with a sensible default - prevents a KeyError
    whenever a newer version adds a column that an existing staff_df doesn't have yet."""
    for column, default in STAFF_COLUMN_DEFAULTS.items():
        if column not in staff_df.columns:
            staff_df[column] = default
    return staff_df


def build_default_fixed_df(names: list[str]) -> pd.DataFrame:
    """Seed the fixed-status grid from each default staff member's own `FixedDays`
    dict (set in DEFAULT_STAFF), e.g. {"Monday": "FERIE"} for someone with a known
    holiday that week. Names not found in DEFAULT_STAFF (e.g. added later via the
    Staff tab) simply start blank."""
    fixed_days_lookup = {person["Name"]: person.get("FixedDays", {}) for person in DEFAULT_STAFF}
    rows = []
    for name in names:
        fixed_days = fixed_days_lookup.get(name, {})
        rows.append({"Name": name, **{day: fixed_days.get(day, "") for day in DAYS}})
    return pd.DataFrame(rows)


def build_default_coverage_df() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        row = {"Day": day}
        for hour in HOURS:
            row[str(hour)] = DEFAULT_HOURLY_MIN_BY_DAY[day][hour]
        rows.append(row)
    return pd.DataFrame(rows)


def build_default_max_staff_df() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        row = {"Day": day}
        for hour in HOURS:
            row[str(hour)] = _MAX_STAFF_OVERRIDES.get(hour, DEFAULT_MAX_STAFF_PER_HOUR)
        rows.append(row)
    return pd.DataFrame(rows)


def build_default_arrivals_departures_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Departures",
         **{day: DEFAULT_ARRIVALS_DEPARTURES.get(day, {}).get("Departures", DEFAULT_ARRIVALS_DEPARTURES_PER_DAY)
            for day in DAYS}},
        {"Metric": "Arrivals",
         **{day: DEFAULT_ARRIVALS_DEPARTURES.get(day, {}).get("Arrivals", DEFAULT_ARRIVALS_DEPARTURES_PER_DAY)
            for day in DAYS}},
    ])


def _window_baseline(coverage_df: pd.DataFrame, hours: set[int]) -> int:
    """The 'usual' staffing level for a window, taken as the average of whatever's
    currently set across all 7 days for those hours. An average day is one where
    arrivals/departures are equal every day - in that case every day scores exactly
    this baseline back, i.e. the table comes out unchanged."""
    values = [
        int(coverage_df.loc[coverage_df["Day"] == day, str(hour)].values[0])
        for day in DAYS for hour in hours
    ]
    return max(1, round(sum(values) / len(values)))


def _scaled_staff(day_value: float, total_value: float, baseline_staff: int, num_days: int = len(DAYS)) -> int:
    """A day's share of the total (across `num_days` days), as a percentage, compared
    against an even 1/num_days share - then that ratio scales the baseline staffing
    level up or down. A day exactly at the average share gets exactly `baseline_staff`."""
    if total_value <= 0 or num_days <= 0:
        return baseline_staff
    even_share_pct = 100 / num_days
    day_share_pct = day_value / total_value * 100
    return max(1, round(baseline_staff * day_share_pct / even_share_pct))


def derive_coverage_from_arrivals(
    coverage_df: pd.DataFrame,
    arrivals_departures_df: pd.DataFrame,
    max_staff_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Adjust the morning/evening (and, when there's a cleaning surplus, afternoon)
    coverage minimums from arrivals/departures counts. Each day is scaled relative to
    its own share of the week's total (day/total*100 against an even 1/7 share). The
    baseline for each window is derived automatically from the average of what's
    already set there - there's nothing to configure: an average day (equal
    arrivals/departures every day) leaves the table unchanged.

    The existing minimum and maximum tables are always leading: the ratio-derived
    value is clamped to never go below whatever minimum is already set for that cell,
    nor above the maximum if one is given - so even a day with 0 arrivals/departures
    still gets at least the minimum you already configured, it's never overwritten
    down to something lower. Hours outside the three windows are left untouched."""
    result = coverage_df.copy()
    ad = arrivals_departures_df.set_index("Metric")
    arrivals = {day: int(ad.loc["Arrivals", day]) for day in DAYS}
    departures = {day: int(ad.loc["Departures", day]) for day in DAYS}
    total_arrivals = sum(arrivals.values())
    total_departures = sum(departures.values())
    surpluses = {day: max(0, departures[day] - arrivals[day]) for day in DAYS}
    total_surplus = sum(surpluses.values())
    days_with_surplus = sum(1 for s in surpluses.values() if s > 0)

    morning_baseline = _window_baseline(coverage_df, ARRIVAL_DEPARTURE_MORNING_HOURS)
    evening_baseline = _window_baseline(coverage_df, ARRIVAL_DEPARTURE_EVENING_HOURS)
    afternoon_baseline = _window_baseline(coverage_df, ARRIVAL_DEPARTURE_AFTERNOON_HOURS)

    def apply_clamped(day: str, hour: int, derived_value: int) -> None:
        floor = int(coverage_df.loc[coverage_df["Day"] == day, str(hour)].values[0])
        value = max(derived_value, floor)
        if max_staff_df is not None:
            ceiling = int(max_staff_df.loc[max_staff_df["Day"] == day, str(hour)].values[0])
            value = min(value, ceiling)
        result.loc[result["Day"] == day, str(hour)] = value

    for day in DAYS:
        morning_min = _scaled_staff(departures[day], total_departures, morning_baseline)
        evening_min = _scaled_staff(arrivals[day], total_arrivals, evening_baseline)
        for hour in ARRIVAL_DEPARTURE_MORNING_HOURS:
            apply_clamped(day, hour, morning_min)
        for hour in ARRIVAL_DEPARTURE_EVENING_HOURS:
            apply_clamped(day, hour, evening_min)

        if surpluses[day] > 0 and total_surplus > 0:
            # compared against the average *among days that have a surplus at all*,
            # not against 1/7 - otherwise a surplus concentrated on 1-2 days gets
            # wildly over-scaled just because most days have none
            afternoon_min = _scaled_staff(surpluses[day], total_surplus, afternoon_baseline, num_days=days_with_surplus)
            for hour in ARRIVAL_DEPARTURE_AFTERNOON_HOURS:
                apply_clamped(day, hour, afternoon_min)
        # no surplus that day: departures and arrivals are roughly back-to-back (no
        # cleaning gap to staff for) - afternoon hours are left at whatever they were

    return result


def build_default_same_shift_df() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_SAME_SHIFT_PAIRS, columns=["Name1", "Name2"])


def build_default_complementary_df() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_COMPLEMENTARY_PAIRS, columns=["Name1", "Name2", "Weight"])


def build_default_requests_df() -> pd.DataFrame:
    if not DEFAULT_REQUESTS:
        return pd.DataFrame(columns=["Name", "Day", "RequestType", "ShiftCode", "Priority"])
    # The Requests tab's ShiftCode dropdown only accepts the full "CODE (start-end)"
    # display string (that's what its SelectboxColumn options list contains), so a
    # bare code like "M2" in DEFAULT_REQUESTS would silently fail to match and get
    # blanked out. Auto-translate bare codes here rather than requiring you to write
    # out the display format (and re-edit it every time the catalog's hours change).
    shift_hours = {code: (start, end) for code, start, end in DEFAULT_SHIFT_CATALOG}
    rows = []
    for request in DEFAULT_REQUESTS:
        row = dict(request)
        code = row.get("ShiftCode", "")
        if code and " (" not in code and code in shift_hours:
            start, end = shift_hours[code]
            row["ShiftCode"] = f"{code} ({start}-{end})"
        rows.append(row)
    return pd.DataFrame(rows, columns=["Name", "Day", "RequestType", "ShiftCode", "Priority"])


def sync_fixed_df(fixed_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Keep the fixed-status grid aligned with the current staff list.
    Returns the same object untouched if the staff list hasn't actually changed,
    so an unrelated rerun (e.g. editing another tab) can't disturb in-progress
    edits in this one - rebuilding a fresh DataFrame every rerun is what caused
    cells to occasionally appear reset."""
    existing_names = fixed_df["Name"].tolist() if "Name" in fixed_df.columns else []
    if existing_names == names:
        return fixed_df
    existing = fixed_df.set_index("Name") if "Name" in fixed_df.columns else pd.DataFrame()
    rows = []
    for name in names:
        if name in existing.index:
            row = {"Name": name}
            for day in DAYS:
                row[day] = existing.loc[name, day] if day in existing.columns else ""
            rows.append(row)
        else:
            rows.append({"Name": name, **{day: "" for day in DAYS}})
    return pd.DataFrame(rows)


def sync_pair_df(pair_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Drop relationship rows that reference a staff member who no longer exists.
    No-op if every row already references a current name, to avoid rebuilding
    the DataFrame (and disturbing its widget state) on unrelated reruns."""
    if pair_df.empty:
        return pair_df
    mask = pair_df["Name1"].isin(names) & pair_df["Name2"].isin(names)
    if mask.all():
        return pair_df
    return pair_df[mask].reset_index(drop=True)


def sync_requests_df(requests_df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Drop requests that reference a staff member who no longer exists.
    No-op if every row already references a current name, to avoid rebuilding
    the DataFrame (and disturbing its widget state) on unrelated reruns."""
    if requests_df.empty:
        return requests_df
    mask = requests_df["Name"].isin(names)
    if mask.all():
        return requests_df
    return requests_df[mask].reset_index(drop=True)


# ============================================================================
# CP-SAT MODEL
# ============================================================================

def shifts_covering_hour(shift_catalog: list[tuple[str, int, int]], hour: int) -> list[str]:
    return [code for code, start, end in shift_catalog if start <= hour < end]


def incompatible_pairs(
    shift_catalog: list[tuple[str, int, int]],
    min_break: int,
    max_break: int | None = None,
) -> list[tuple[str, str]]:
    """Pairs of shifts that cannot both be worked by the same person on the same day:
    they overlap or touch (gap < min_break), or - if max_break is set - the gap between
    them is larger than that cap (so a split shift can't have a huge dead stretch)."""
    lookup = {code: (start, end) for code, start, end in shift_catalog}
    codes = list(lookup.keys())
    incompat = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            c1, c2 = codes[i], codes[j]
            s1, e1 = lookup[c1]
            s2, e2 = lookup[c2]
            gap = (s2 - e1) if s1 <= s2 else (s1 - e2)
            if gap < min_break:
                incompat.append((c1, c2))
            elif max_break is not None and gap > max_break:
                incompat.append((c1, c2))
    return incompat


@dataclass
class SolveResult:
    status_name: str
    feasible: bool
    schedule_df: pd.DataFrame | None
    total_understaffing: int
    total_overstaffing: int


def validate_inputs(
    staff_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    max_staff_df: pd.DataFrame | None,
    same_shift_df: pd.DataFrame,
    complementary_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Catch obviously-inconsistent input before handing it to the solver, where a
    contradiction would otherwise only surface as an opaque INFEASIBLE/UNKNOWN status.
    Returns (errors, warnings) - errors should block solving, warnings are just FYI."""
    errors: list[str] = []
    warnings: list[str] = []

    names = staff_df["Name"].tolist()
    duplicate_names = {n for n in names if names.count(n) > 1}
    if duplicate_names:
        errors.append(f"Duplicate staff name(s): {', '.join(sorted(duplicate_names))}. Names must be unique.")

    bad_hours = staff_df[staff_df["ContractHours"] <= 0]
    if not bad_hours.empty:
        errors.append(f"Contract hours must be positive: {', '.join(bad_hours['Name'].tolist())}.")

    bad_shifts = shift_df[shift_df["End"] <= shift_df["Start"]]
    if not bad_shifts.empty:
        errors.append(f"Shift(s) with end time not after start time: {', '.join(bad_shifts['Code'].tolist())}.")

    if max_staff_df is not None:
        violations = []
        for day in DAYS:
            for hour in HOURS:
                min_needed = int(coverage_df.loc[coverage_df["Day"] == day, str(hour)].values[0])
                max_allowed = int(max_staff_df.loc[max_staff_df["Day"] == day, str(hour)].values[0])
                if min_needed > max_allowed:
                    violations.append(f"{day} {hour}h ({min_needed} > {max_allowed})")
        if violations:
            shown = ", ".join(violations[:5])
            more = f", and {len(violations) - 5} more" if len(violations) > 5 else ""
            errors.append(f"Minimum coverage exceeds the maximum for {len(violations)} hour block(s): {shown}{more}.")

    for label, pair_df in (("same-shift", same_shift_df), ("complementary", complementary_df)):
        if pair_df.empty:
            continue
        self_paired = pair_df[pair_df["Name1"] == pair_df["Name2"]]
        if not self_paired.empty:
            warnings.append(f"{len(self_paired)} {label} row(s) pair someone with themselves - ignored.")
        pairs_only = pair_df[["Name1", "Name2"]].apply(lambda r: tuple(sorted(r)), axis=1)
        dupes = pairs_only[pairs_only.duplicated()]
        if not dupes.empty:
            warnings.append(f"{len(dupes)} duplicate {label} pair(s) - harmless, but can be tidied up.")

    return errors, warnings


def solve_schedule(
    staff_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    same_shift_df: pd.DataFrame,
    complementary_df: pd.DataFrame,
    requests_df: pd.DataFrame,
    max_staff_df: pd.DataFrame | None = None,
    min_break_hours: int = DEFAULT_MIN_BREAK_HOURS,
    max_break_hours: int | None = None,
    max_daily_hours: int = DEFAULT_MAX_DAILY_HOURS,
    strict_weekly_hours: bool = STRICT_WEEKLY_HOURS,
    solver_time_limit_sec: int = DEFAULT_SOLVER_TIME_LIMIT_SEC,
) -> SolveResult:
    """Build and solve the CP-SAT rota model. Returns a long-format schedule
    (one row per person per day) plus solver diagnostics."""

    shift_catalog = list(shift_df.itertuples(index=False, name=None))
    shift_lookup = {code: (start, end) for code, start, end in shift_catalog}
    shift_codes = list(shift_lookup.keys())
    incompat = incompatible_pairs(shift_catalog, min_break_hours, max_break_hours)

    names = staff_df["Name"].tolist()
    split_allowed = dict(zip(staff_df["Name"], staff_df["SplitAllowed"]))
    contract_type = dict(zip(staff_df["Name"], staff_df.get("ContractType", CONTRACT_TYPE_FLEXIBLE)))
    max_closing = dict(zip(staff_df["Name"], staff_df.get("MaxClosingShiftsPerWeek", 0)))

    # Plain dict lookups instead of repeated pandas .loc calls inside the hot loops below -
    # a DataFrame .loc lookup re-does index alignment on every call, which adds up fast
    # once you're calling it thousands of times while building the model.
    fixed: dict[tuple[str, str], str] = {
        (row.Name, day): getattr(row, day) for row in fixed_df.itertuples() for day in DAYS
    }

    def is_off_day(name: str, day: str) -> bool:
        return fixed.get((name, day), "") in ("FERIE", "RECUPERO")

    coverage_min: dict[tuple[str, int], int] = {
        (day, hour): int(coverage_df.loc[coverage_df["Day"] == day, str(hour)].values[0])
        for day in DAYS for hour in HOURS
    }
    coverage_max: dict[tuple[str, int], int] | None = None
    if max_staff_df is not None:
        coverage_max = {
            (day, hour): int(max_staff_df.loc[max_staff_df["Day"] == day, str(hour)].values[0])
            for day in DAYS for hour in HOURS
        }

    closing_hour = max(end for _, _, end in shift_catalog)
    closing_codes = [code for code, (start, end) in shift_lookup.items() if end == closing_hour]
    four_hour_codes = [code for code, (start, end) in shift_lookup.items() if end - start == FIXED_SPLIT_SHIFT_HOURS]
    six_hour_codes = [code for code, (start, end) in shift_lookup.items() if end - start == 6]
    all_durations = {end - start for _, start, end in shift_catalog}
    # cache which shift codes cover each hour once, rather than recomputing per (day, hour) pair
    covering_by_hour: dict[int, list[str]] = {hour: shifts_covering_hour(shift_catalog, hour) for hour in HOURS}
    # The "3x8h + 2x6h + 1x4h" day-type balancing rule below only makes algebraic
    # sense when every shift is 4h or 6h long and two shifts can't be combined
    # into anything but 8h (i.e. max_daily_hours forbids a 4h+6h=10h day) - both
    # true by default. If the catalog or the daily-hours cap has been changed
    # away from that, the rule is skipped rather than silently misapplied.
    day_type_rule_valid = all_durations.issubset({4, 6}) and max_daily_hours < 10

    def fixed_split_gap(c1: str, c2: str) -> int:
        s1, e1 = shift_lookup[c1]
        s2, e2 = shift_lookup[c2]
        return (s2 - e1) if s1 <= s2 else (s1 - e2)

    model = cp_model.CpModel()
    x: dict[tuple[str, str, str], cp_model.IntVar] = {}
    y: dict[tuple[str, str], cp_model.IntVar] = {}

    # Requests overrule a RIPOSO fixed day (but never FERIE/RECUPERO - those are real
    # absences, not just a scheduling default). If someone has a "Prefer shift" or
    # "Avoid shift" request on a day currently marked RIPOSO, that day opens back up
    # for scheduling instead of being hard-forced to zero hours.
    riposo_overridden_by_request: set[tuple[str, str]] = set()
    for row in requests_df.itertuples():
        name, day, rtype = getattr(row, "Name", None), getattr(row, "Day", None), getattr(row, "RequestType", None)
        if name not in names or day not in DAYS or rtype not in ("Prefer shift", "Avoid shift"):
            continue
        if fixed.get((name, day), "") == "RIPOSO":
            riposo_overridden_by_request.add((name, day))

    for name in names:
        is_fixed_split = contract_type.get(name) == CONTRACT_TYPE_FIXED_SPLIT
        for day in DAYS:
            status = fixed.get((name, day), "")
            if status in ("FERIE", "RECUPERO"):
                y[name, day] = model.NewConstant(0)
                continue

            for code in shift_codes:
                x[name, day, code] = model.NewBoolVar(f"x_{name}_{day}_{code}")
            day_shifts = [x[name, day, c] for c in shift_codes]

            y[name, day] = model.NewBoolVar(f"y_{name}_{day}")
            model.AddMaxEquality(y[name, day], day_shifts)

            max_shifts_today = 2 if split_allowed.get(name, True) else 1
            model.Add(sum(day_shifts) <= max_shifts_today)

            for c1, c2 in incompat:
                model.Add(x[name, day, c1] + x[name, day, c2] <= 1)

            daily_hours = sum(x[name, day, c] * (shift_lookup[c][1] - shift_lookup[c][0]) for c in shift_codes)
            model.Add(daily_hours <= max_daily_hours)

            if status == "RIPOSO" and (name, day) not in riposo_overridden_by_request:
                model.Add(sum(day_shifts) == 0)

            if is_fixed_split:
                # only 4h blocks are allowed, and a working day is exactly two of them
                for code in shift_codes:
                    if code not in four_hour_codes:
                        model.Add(x[name, day, code] == 0)
                model.Add(sum(x[name, day, c] for c in four_hour_codes) == 2 * y[name, day])
                for i in range(len(four_hour_codes)):
                    for j in range(i + 1, len(four_hour_codes)):
                        c1, c2 = four_hour_codes[i], four_hour_codes[j]
                        gap = fixed_split_gap(c1, c2)
                        if not (FIXED_SPLIT_MIN_BREAK <= gap <= FIXED_SPLIT_MAX_BREAK):
                            model.Add(x[name, day, c1] + x[name, day, c2] <= 1)

    # max closing shifts per week (a "closing" shift is whichever shift(s) end at the latest hour in the catalog)
    for name in names:
        limit = int(max_closing.get(name, 0) or 0)
        if limit <= 0:
            continue
        closing_terms = [x[name, day, c] for day in DAYS for c in closing_codes if (name, day, c) in x]
        if closing_terms:
            model.Add(sum(closing_terms) <= limit)

    # minimum rest days per week (fixed FERIE/RECUPERO days don't count towards this rule)
    for row in staff_df.itertuples():
        name = row.Name
        min_rest = int(row.MinRestDays)
        if contract_type.get(name) == CONTRACT_TYPE_FIXED_SPLIT:
            min_rest = max(min_rest, FIXED_SPLIT_MIN_REST_DAYS)
        rest_terms = [1 - y[name, day] for day in DAYS if not is_off_day(name, day)]
        if rest_terms:
            model.Add(sum(rest_terms) >= min_rest)

    # With exactly 1 day off (6 working days) and a 40h target, spread those
    # hours as 3 days of 8h, 2 days of 6h, and 1 day of 4h (3*8+2*6+1*4=40).
    # Doesn't apply to the fixed 8h-split contract (that's always 2x4h blocks).
    if day_type_rule_valid:
        for row in staff_df.itertuples():
            name = row.Name
            if int(row.MinRestDays) != 1:
                continue
            if contract_type.get(name) == CONTRACT_TYPE_FIXED_SPLIT:
                continue
            if int(round(float(row.ContractHours))) != 40:
                continue

            workable_days = [day for day in DAYS if not is_off_day(name, day)]
            if len(workable_days) < len(DAYS):
                continue  # a FERIE/RECUPERO day lowers this person's target for the week (see below),
                          # which no longer matches the fixed 3*8+2*6+1*4=40 split this rule assumes

            # exactly one rest day (not just "at least one") so all 6 working
            # days are available to be distributed across the 8h/6h/4h buckets
            model.Add(sum(1 - y[name, day] for day in workable_days) == 1)

            is_8h_day = []
            six_hour_terms = []
            four_hour_terms = []
            for day in workable_days:
                four_sum = sum(x[name, day, c] for c in four_hour_codes)
                six_sum = sum(x[name, day, c] for c in six_hour_codes)
                is_8 = model.NewBoolVar(f"is8h_{name}_{day}")
                model.Add(four_sum == 2).OnlyEnforceIf(is_8)
                model.Add(four_sum <= 1).OnlyEnforceIf(is_8.Not())
                is_8h_day.append(is_8)
                six_hour_terms.append(six_sum)
                four_hour_terms.append(four_sum - 2 * is_8)  # 1 iff this day is a single 4h-shift day

            model.Add(sum(is_8h_day) == 3)
            model.Add(sum(six_hour_terms) == 2)
            model.Add(sum(four_hour_terms) == 1)

    # maximum consecutive working days (rolling window across the week)
    for row in staff_df.itertuples():
        name = row.Name
        max_consec = int(row.MaxConsecDays)
        window = max_consec + 1
        if window <= len(DAYS):
            for start in range(0, len(DAYS) - window + 1):
                model.Add(sum(y[name, DAYS[start + k]] for k in range(window)) <= max_consec)

    # "days off together" (soft): only meaningful with exactly 2 days off/week.
    # DaysOffTogether=True rewards the 2 rest days landing on adjacent calendar
    # days (e.g. a Sat+Sun weekend); False rewards them being kept apart instead.
    days_off_together_terms = []
    adjacent_day_pairs = list(zip(DAYS[:-1], DAYS[1:]))  # (Mon,Tue), (Tue,Wed), ..., (Sat,Sun)
    for row in staff_df.itertuples():
        name = row.Name
        if int(row.MinRestDays) != 2:
            continue
        wants_together = bool(getattr(row, "DaysOffTogether", True))
        for day1, day2 in adjacent_day_pairs:
            if is_off_day(name, day1) or is_off_day(name, day2):
                continue  # a FERIE/RECUPERO day isn't part of this preference
            rest1 = 1 - y[name, day1]
            rest2 = 1 - y[name, day2]
            both_rest = model.NewBoolVar(f"bothrest_{name}_{day1}_{day2}")
            model.Add(both_rest <= rest1)
            model.Add(both_rest <= rest2)
            model.Add(both_rest >= rest1 + rest2 - 1)
            if wants_together:
                days_off_together_terms.append(-WEIGHT_DAYS_OFF_TOGETHER * both_rest)  # reward adjacency
            else:
                days_off_together_terms.append(WEIGHT_DAYS_OFF_TOGETHER * both_rest)   # penalise adjacency

    # weekly contract-hours target: exact and hard when strict_weekly_hours is on.
    # Each FERIE/RECUPERO day lowers that person's target for the week by
    # HOURS_PER_FERIE_RECUPERO_DAY (a standard working day's worth of hours),
    # rather than forcing the full contract target into fewer available days.
    hours_deviation_terms = []
    for row in staff_df.itertuples():
        name = row.Name
        target = int(round(float(row.ContractHours)))
        weekly_terms = []
        for day in DAYS:
            if is_off_day(name, day):
                continue
            weekly_terms.extend(x[name, day, c] * (shift_lookup[c][1] - shift_lookup[c][0]) for c in shift_codes)
        total_hours = sum(weekly_terms) if weekly_terms else 0

        available_days = sum(1 for day in DAYS if not is_off_day(name, day))
        off_days = len(DAYS) - available_days
        max_possible = available_days * max_daily_hours
        adjusted_target = max(0, target - HOURS_PER_FERIE_RECUPERO_DAY * off_days)
        # safety net: if the adjustment still can't fit (e.g. a very short max_daily_hours),
        # fall back to the maximum achievable rather than forcing an infeasible model
        effective_target = min(adjusted_target, max_possible)

        if strict_weekly_hours:
            model.Add(total_hours == effective_target)
        else:
            model.Add(total_hours <= max_possible)
            dev = model.NewIntVar(-max_possible, max_possible, f"dev_{name}")
            model.Add(dev == total_hours - effective_target)
            abs_dev = model.NewIntVar(0, max_possible, f"absdev_{name}")
            model.AddAbsEquality(abs_dev, dev)
            hours_deviation_terms.append(abs_dev)

    # "same shift" pairs: hard link, on any day where neither has a fixed FERIE/RECUPERO entry
    for row in same_shift_df.itertuples():
        n1, n2 = row.Name1, row.Name2
        if n1 not in names or n2 not in names or n1 == n2:
            continue
        for day in DAYS:
            if is_off_day(n1, day) or is_off_day(n2, day):
                continue
            for code in shift_codes:
                model.Add(x[n1, day, code] == x[n2, day, code])

    # "complementary" pairs: soft penalty for working the exact same hour at the same time
    workhour_cache: dict[tuple[str, str, int], cp_model.IntVar] = {}

    def get_workhour_var(name: str, day: str, hour: int) -> cp_model.IntVar:
        key = (name, day, hour)
        if key in workhour_cache:
            return workhour_cache[key]
        covering = covering_by_hour[hour]
        terms = [x[name, day, c] for c in covering if (name, day, c) in x]
        var = model.NewBoolVar(f"work_{name}_{day}_{hour}")
        if terms:
            model.AddMaxEquality(var, terms)
        else:
            model.Add(var == 0)
        workhour_cache[key] = var
        return var

    complementary_terms = []
    for row in complementary_df.itertuples():
        n1, n2, weight = row.Name1, row.Name2, int(row.Weight)
        if n1 not in names or n2 not in names or n1 == n2:
            continue
        for day in DAYS:
            for hour in HOURS:
                v1 = get_workhour_var(n1, day, hour)
                v2 = get_workhour_var(n2, day, hour)
                overlap = model.NewBoolVar(f"overlap_{n1}_{n2}_{day}_{hour}")
                model.Add(overlap <= v1)
                model.Add(overlap <= v2)
                model.Add(overlap >= v1 + v2 - 1)
                complementary_terms.append(weight * overlap)

    # individual shift / day-off requests (soft)
    request_terms = []
    for row in requests_df.itertuples():
        name, day, rtype = row.Name, row.Day, row.RequestType
        code_display = getattr(row, "ShiftCode", "") or ""
        code = code_display.split(" ")[0]  # ShiftCode is stored as "CODE (start-end)"; the solver only needs CODE
        priority = getattr(row, "Priority", "Low")
        if name not in names or day not in DAYS:
            continue
        if is_off_day(name, day):
            continue  # already fixed, request is moot
        weight = PRIORITY_WEIGHTS.get(priority, 1) * REQUEST_WEIGHT_SCALE
        if rtype == "Prefer day off":
            request_terms.append(weight * y[name, day])
        elif rtype == "Prefer shift" and (name, day, code) in x:
            request_terms.append(weight * (1 - x[name, day, code]))
        elif rtype == "Avoid shift" and (name, day, code) in x:
            request_terms.append(weight * x[name, day, code])

    # hour-by-hour coverage: minimum is soft (understaffing/overstaffing tracked and
    # penalised), maximum (if a max_staff_df is given) is a hard ceiling.
    understaff_terms = []
    overstaff_terms = []
    weighted_overstaff_terms = []
    for day in DAYS:
        for hour in HOURS:
            covering = covering_by_hour[hour]
            min_needed = coverage_min[day, hour]
            present_terms = [x[name, day, c] for name in names if not is_off_day(name, day) for c in covering]
            total_present = sum(present_terms) if present_terms else 0
            under = model.NewIntVar(0, len(names), f"under_{day}_{hour}")
            over = model.NewIntVar(0, len(names), f"over_{day}_{hour}")
            model.Add(total_present + under - over == min_needed)
            understaff_terms.append(under)
            overstaff_terms.append(over)
            weighted_overstaff_terms.append(overstaff_weight_for_hour(hour) * over)

            if coverage_max is not None:
                model.Add(total_present <= coverage_max[day, hour])

    model.Minimize(
        WEIGHT_UNDERSTAFF * sum(understaff_terms)
        + sum(weighted_overstaff_terms)
        + 3 * sum(hours_deviation_terms)
        + sum(complementary_terms)
        + sum(request_terms)
        + sum(days_off_together_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_time_limit_sec
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveResult(solver.StatusName(status), False, None, -1, -1)

    rows = []
    for name in names:
        for day in DAYS:
            status_val = fixed.get((name, day), "")
            if status_val in ("FERIE", "RECUPERO"):
                rows.append({"Name": name, "Day": day, "Shifts": status_val, "Hours": 0})
                continue
            assigned = [c for c in shift_codes if solver.Value(x[name, day, c]) == 1]
            if not assigned:
                rows.append({"Name": name, "Day": day, "Shifts": "RIPOSO", "Hours": 0})
            else:
                assigned_sorted = sorted(assigned, key=lambda c: shift_lookup[c][0])
                label = "  ".join(f"{shift_lookup[c][0]}-{shift_lookup[c][1]}" for c in assigned_sorted)
                hrs = sum(shift_lookup[c][1] - shift_lookup[c][0] for c in assigned_sorted)
                rows.append({"Name": name, "Day": day, "Shifts": label, "Hours": hrs})

    schedule_df = pd.DataFrame(rows)
    return SolveResult(
        solver.StatusName(status),
        True,
        schedule_df,
        int(solver.Value(sum(understaff_terms))),
        int(solver.Value(sum(overstaff_terms))),
    )


# ============================================================================
# GRID / EXPORT HELPERS
# ============================================================================

def day_labels(week_start: date) -> dict[str, str]:
    """Map each weekday name to its 'Weekday D-M-YYYY' label for the chosen week."""
    labels = {}
    for i, day in enumerate(DAYS):
        d = week_start + timedelta(days=i)
        labels[day] = f"{day} {d.day}-{d.month}-{d.year}"
    return labels


def pivot_grid(schedule_df: pd.DataFrame, staff_order: list[str], week_start: date) -> pd.DataFrame:
    labels = day_labels(week_start)
    pivot = schedule_df.pivot(index="Name", columns="Day", values="Shifts")
    pivot = pivot.reindex(index=staff_order, columns=DAYS)
    pivot = pivot.rename(columns=labels)
    return pivot


def coverage_check_df(schedule_df: pd.DataFrame, shift_df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    """Actual staff present per hour vs. the minimum required, for a quick sanity check."""
    shift_catalog = list(shift_df.itertuples(index=False, name=None))
    rows = []
    for day in DAYS:
        day_shifts = schedule_df[schedule_df["Day"] == day]
        for hour in HOURS:
            covering_codes = shifts_covering_hour(shift_catalog, hour)
            present = 0
            for shifts_label in day_shifts["Shifts"]:
                if shifts_label in ("RIPOSO", "FERIE", "RECUPERO"):
                    continue
                for code, s, e in shift_catalog:
                    label = f"{s}-{e}"
                    if label in shifts_label.split("  ") and code in covering_codes:
                        present += 1
                        break
            required = int(coverage_df.loc[coverage_df["Day"] == day, str(hour)].values[0])
            rows.append({"Day": day, "Hour": hour, "Present": present, "Required": required,
                         "Gap": present - required})
    return pd.DataFrame(rows)


def headcount_grid(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the per-hour coverage check into a Day (rows) x Hour (columns) grid
    of how many people are working."""
    pivot = cov_df.pivot(index="Day", columns="Hour", values="Present")
    pivot = pivot.reindex(index=DAYS, columns=HOURS)
    pivot.columns = [f"{h}h" for h in pivot.columns]
    return pivot


def to_csv_bytes(grid_df: pd.DataFrame) -> bytes:
    """Export the pivoted grid (names as rows, days as columns) - same shape as
    the on-screen schedule table - rather than a long one-row-per-person-per-day format."""
    return grid_df.to_csv(index=True, index_label="Name").encode("utf-8")


# ============================================================================
# STREAMLIT UI
# ============================================================================

def _save_feedback(snapshot_key: str, new_value: pd.DataFrame, label: str) -> None:
    """Call right after a form is submitted. Toasts a confirmation, and can tell a
    genuine change apart from clicking Save/Apply with nothing new to save - this is
    the honest limit of what's detectable: st.form doesn't rerun the script while
    you're editing, only at submit, so there's no way to show a live "unsaved
    changes" state before that point (that's the trade-off for fixing the
    double-entry bug - forms can't watch for changes they haven't been told about)."""
    previous = st.session_state.get(snapshot_key)
    changed = previous is None or not new_value.equals(previous)
    if changed:
        st.toast(f"{label} saved", icon=":material/check_circle:")
    else:
        st.toast(f"No changes to {label.lower()}", icon=":material/info:")
    st.session_state[snapshot_key] = new_value.copy()


def init_session_state() -> None:
    if "staff_df" not in st.session_state:
        st.session_state.staff_df = build_default_staff_df()
    st.session_state.staff_df = ensure_staff_columns(st.session_state.staff_df)
    if "shift_df" not in st.session_state:
        st.session_state.shift_df = build_default_shift_df()
    if "fixed_df" not in st.session_state:
        st.session_state.fixed_df = build_default_fixed_df(st.session_state.staff_df["Name"].tolist())
    coverage_freshly_built = "coverage_df" not in st.session_state
    if "coverage_df" not in st.session_state:
        st.session_state.coverage_df = build_default_coverage_df()
    if "max_staff_df" not in st.session_state:
        st.session_state.max_staff_df = build_default_max_staff_df()
    if "arrivals_departures_df" not in st.session_state:
        st.session_state.arrivals_departures_df = build_default_arrivals_departures_df()
    if coverage_freshly_built and DEFAULT_ARRIVALS_DEPARTURES:
        # DEFAULT_ARRIVALS_DEPARTURES has been customised away from the flat default,
        # so derive the minimum-coverage table from it right away instead of making
        # you click "Apply" once just to get your own defaults reflected.
        st.session_state.coverage_df = derive_coverage_from_arrivals(
            st.session_state.coverage_df,
            st.session_state.arrivals_departures_df,
            max_staff_df=st.session_state.max_staff_df,
        )
    if "same_shift_df" not in st.session_state:
        st.session_state.same_shift_df = build_default_same_shift_df()
    if "complementary_df" not in st.session_state:
        st.session_state.complementary_df = build_default_complementary_df()
    if "requests_df" not in st.session_state:
        st.session_state.requests_df = build_default_requests_df()
    if "result" not in st.session_state:
        st.session_state.result = None
    if "week_start" not in st.session_state:
        today = date.today()
        days_until_next_monday = 7 - today.weekday()  # today.weekday(): Mon=0..Sun=6, always yields 1-7, never 0
        st.session_state.week_start = today + timedelta(days=days_until_next_monday)
    if "limit_max_break" not in st.session_state:
        st.session_state.limit_max_break = True


def render_staff_tab() -> None:
    st.caption("One row per team member. Split shifts (e.g. morning + evening) are allowed only when 'SplitAllowed' is checked. "
               "'Days off/week' is 1 or 2. 'Fixed 8h split' overrides SplitAllowed and forces exactly two 4-hour blocks "
               "with a 1-2h break, plus at least 2 days off/week regardless of what's set here. 'Days off together' only "
               "matters with 2 days off/week: checked prefers them consecutive (e.g. weekend), unchecked prefers them apart - "
               "it's a soft preference, not a hard rule.")

    with st.expander("Quick set: days off for everyone", icon=":material/bolt:"):
        col_a, col_b = st.columns([1, 2])
        with col_a:
            bulk_rest_days = st.selectbox("Days off/week", options=REST_DAYS_OPTIONS, key="bulk_rest_days_select")
        with col_b:
            st.write("")  # vertical alignment spacer
            if st.button(f"Apply {bulk_rest_days} day(s) off to everyone", icon=":material/check:"):
                st.session_state.staff_df["MinRestDays"] = bulk_rest_days
                st.session_state.pop("staff_editor", None)
                st.rerun()

    with st.form("staff_form", border=False):
        edited = st.data_editor(
            st.session_state.staff_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn("Name", required=True),
                "ContractHours": st.column_config.NumberColumn("Contract hrs/week", min_value=0, max_value=60, step=1),
                "MinRestDays": st.column_config.SelectboxColumn("Days off/week", options=REST_DAYS_OPTIONS, required=True),
                "MaxConsecDays": st.column_config.NumberColumn("Max consecutive work days", min_value=1, max_value=7, step=1),
                "SplitAllowed": st.column_config.CheckboxColumn("Split shift allowed"),
                "ContractType": st.column_config.SelectboxColumn("Contract type", options=CONTRACT_TYPE_OPTIONS, required=True),
                "MaxClosingShiftsPerWeek": st.column_config.NumberColumn("Max closing shifts/week (0 = unlimited)",
                                                                          min_value=0, max_value=7, step=1),
                "DaysOffTogether": st.column_config.CheckboxColumn("Days off together (vs. separate)"),
            },
            key="staff_editor",
        )
        staff_submitted = st.form_submit_button("Save staff", icon=":material/save:")

    edited = edited.dropna(subset=["Name"])
    edited = edited[edited["Name"].str.strip() != ""]
    edited["ContractHours"] = edited["ContractHours"].fillna(30)
    edited["MinRestDays"] = edited["MinRestDays"].fillna(1)
    edited["MaxConsecDays"] = edited["MaxConsecDays"].fillna(6)
    edited["SplitAllowed"] = edited["SplitAllowed"].fillna(True)
    edited["ContractType"] = edited["ContractType"].fillna(CONTRACT_TYPE_FLEXIBLE)
    edited["MaxClosingShiftsPerWeek"] = edited["MaxClosingShiftsPerWeek"].fillna(DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK)
    edited["DaysOffTogether"] = edited["DaysOffTogether"].fillna(True)
    st.session_state.staff_df = edited.reset_index(drop=True)
    if staff_submitted:
        _save_feedback("_snap_staff_df", st.session_state.staff_df, "Staff")

    names = st.session_state.staff_df["Name"].tolist()
    st.session_state.fixed_df = sync_fixed_df(st.session_state.fixed_df, names)
    st.session_state.same_shift_df = sync_pair_df(st.session_state.same_shift_df, names)
    st.session_state.complementary_df = sync_pair_df(st.session_state.complementary_df, names)
    st.session_state.requests_df = sync_requests_df(st.session_state.requests_df, names)


def render_shift_tab() -> None:
    st.caption(f"Available shift blocks, all within business hours {OPEN_HOUR}:00-{CLOSE_HOUR}:00, "
               f"max {MAX_SHIFT_LENGTH_HOURS}h each. The solver only assigns whole blocks from this catalog.")
    with st.form("shift_form", border=False):
        edited = st.data_editor(
            st.session_state.shift_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Code": st.column_config.TextColumn("Code", required=True),
                "Start": st.column_config.NumberColumn("Start hour", min_value=OPEN_HOUR, max_value=CLOSE_HOUR - 1, step=1),
                "End": st.column_config.NumberColumn("End hour", min_value=OPEN_HOUR + 1, max_value=CLOSE_HOUR, step=1),
            },
            key="shift_editor",
        )
        shift_submitted = st.form_submit_button("Save shift catalog", icon=":material/save:")

    edited = edited.dropna(subset=["Code", "Start", "End"])
    edited = edited[edited["End"] > edited["Start"]]
    too_long = edited[edited["End"] - edited["Start"] > MAX_SHIFT_LENGTH_HOURS]
    if not too_long.empty:
        st.warning(f"Removed {len(too_long)} shift(s) longer than {MAX_SHIFT_LENGTH_HOURS}h: "
                   f"{', '.join(too_long['Code'].tolist())}", icon=":material/warning:")
    edited = edited[edited["End"] - edited["Start"] <= MAX_SHIFT_LENGTH_HOURS]
    edited = edited.drop_duplicates(subset=["Code"])
    st.session_state.shift_df = edited.reset_index(drop=True)
    if shift_submitted:
        _save_feedback("_snap_shift_df", st.session_state.shift_df, "Shift catalog")


def render_fixed_tab() -> None:
    st.caption("Force a specific day off (RIPOSO), vacation (FERIE) or comp day (RECUPERO) for someone. Leave blank to let the solver decide rest days automatically.")
    column_config = {"Name": st.column_config.TextColumn("Name", disabled=True)}
    for day in DAYS:
        column_config[day] = st.column_config.SelectboxColumn(day, options=FIXED_STATUS_OPTIONS)
    with st.form("fixed_form", border=False):
        edited = st.data_editor(
            st.session_state.fixed_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
            key="fixed_editor",
        )
        fixed_submitted = st.form_submit_button("Save fixed days", icon=":material/save:")
    st.session_state.fixed_df = edited
    if fixed_submitted:
        _save_feedback("_snap_fixed_df", st.session_state.fixed_df, "Fixed days")


def render_coverage_tab() -> None:
    st.subheader("Arrivals & departures", divider=False)
    st.caption(
        "Optional: derive the minimum-coverage table below from expected guest arrivals/departures instead of "
        "setting it by hand. Each day is scaled by its own share of the week's total (day/total*100, compared "
        "against an even 1/7 share) - an average day (equal arrivals/departures every day) leaves the table "
        "unchanged; busier days get more, quieter days get less. The existing minimum/maximum tables are always "
        "leading: the derived value never goes below the minimum already set for a cell, or above its maximum - "
        "so even a day with 0 arrivals/departures keeps at least its current minimum. Departures drive the morning window "
        f"({min(ARRIVAL_DEPARTURE_MORNING_HOURS)}h-{max(ARRIVAL_DEPARTURE_MORNING_HOURS) + 1}h), arrivals drive "
        f"the evening window ({min(ARRIVAL_DEPARTURE_EVENING_HOURS)}h-{max(ARRIVAL_DEPARTURE_EVENING_HOURS) + 1}h). "
        f"The afternoon gap ({min(ARRIVAL_DEPARTURE_AFTERNOON_HOURS)}h-{max(ARRIVAL_DEPARTURE_AFTERNOON_HOURS) + 1}h) "
        "only gets extra coverage when departures exceed arrivals that day - that surplus means rooms aren't "
        "turned over back-to-back, so there's cleaning to staff for before the next check-ins."
    )
    ad_column_config = {"Metric": st.column_config.TextColumn("", disabled=True)}
    for day in DAYS:
        ad_column_config[day] = st.column_config.NumberColumn(day, min_value=0, step=1)
    with st.form("arrivals_departures_form", border=False):
        ad_edited = st.data_editor(
            st.session_state.arrivals_departures_df,
            width="stretch",
            hide_index=True,
            column_config=ad_column_config,
            key="arrivals_departures_editor",
        )
        apply_clicked = st.form_submit_button("Apply", icon=":material/sync:")
    st.session_state.arrivals_departures_df = ad_edited

    if apply_clicked:
        st.session_state.coverage_df = derive_coverage_from_arrivals(
            st.session_state.coverage_df,
            st.session_state.arrivals_departures_df,
            max_staff_df=st.session_state.max_staff_df,
        )
        st.session_state.pop("coverage_editor", None)
        st.toast("Minimum coverage updated from arrivals/departures", icon=":material/check_circle:")
        st.rerun()

    st.space("small")
    st.subheader("Minimum staff per hour", divider=False)
    st.caption("Minimum number of staff required, per day and per hour block. Defaults already differ per day of the week - edit freely.")
    column_config = {"Day": st.column_config.TextColumn("Day", disabled=True)}
    for hour in HOURS:
        column_config[str(hour)] = st.column_config.NumberColumn(f"{hour}h", min_value=0, max_value=10, step=1, width="small")
    with st.form("coverage_min_form", border=False):
        edited = st.data_editor(
            st.session_state.coverage_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
            key="coverage_editor",
        )
        min_cov_submitted = st.form_submit_button("Save minimum coverage", icon=":material/save:")
    st.session_state.coverage_df = edited
    if min_cov_submitted:
        _save_feedback("_snap_coverage_df", st.session_state.coverage_df, "Minimum coverage")

    with st.expander("Quick fill (minimum)", icon=":material/bolt:"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fill_value = st.number_input("Value", min_value=0, max_value=10, value=2, step=1, key="min_fill_value")
        with col2:
            fill_days = st.multiselect("Days", DAYS, default=DAYS, key="min_fill_days")
        with col3:
            fill_hours = st.multiselect("Hours", HOURS, default=HOURS, key="min_fill_hours")
        if st.button("Apply to selected cells", icon=":material/check:", key="min_fill_apply"):
            df = st.session_state.coverage_df.copy()
            for day in fill_days:
                for hour in fill_hours:
                    df.loc[df["Day"] == day, str(hour)] = fill_value
            st.session_state.coverage_df = df
            st.session_state.pop("coverage_editor", None)
            st.rerun()

    st.space("small")
    st.subheader("Maximum staff per hour", divider=False)
    st.caption("A hard ceiling - the solver will never schedule more people than this in a given hour block, regardless of the objective.")
    max_column_config = {"Day": st.column_config.TextColumn("Day", disabled=True)}
    for hour in HOURS:
        max_column_config[str(hour)] = st.column_config.NumberColumn(f"{hour}h", min_value=0, max_value=20, step=1, width="small")
    with st.form("coverage_max_form", border=False):
        max_edited = st.data_editor(
            st.session_state.max_staff_df,
            width="stretch",
            hide_index=True,
            column_config=max_column_config,
            key="max_staff_editor",
        )
        max_cov_submitted = st.form_submit_button("Save maximum coverage", icon=":material/save:")
    st.session_state.max_staff_df = max_edited
    if max_cov_submitted:
        _save_feedback("_snap_max_staff_df", st.session_state.max_staff_df, "Maximum coverage")

    with st.expander("Quick fill (maximum)", icon=":material/bolt:"):
        col4, col5, col6 = st.columns(3)
        with col4:
            max_fill_value = st.number_input("Value", min_value=0, max_value=20, value=DEFAULT_MAX_STAFF_PER_HOUR, step=1, key="max_fill_value")
        with col5:
            max_fill_days = st.multiselect("Days", DAYS, default=DAYS, key="max_fill_days")
        with col6:
            max_fill_hours = st.multiselect("Hours", HOURS, default=HOURS, key="max_fill_hours")
        if st.button("Apply to selected cells", icon=":material/check:", key="max_fill_apply"):
            df = st.session_state.max_staff_df.copy()
            for day in max_fill_days:
                for hour in max_fill_hours:
                    df.loc[df["Day"] == day, str(hour)] = max_fill_value
            st.session_state.max_staff_df = df
            st.session_state.pop("max_staff_editor", None)
            st.rerun()


def render_rules_tab() -> None:
    names = st.session_state.staff_df["Name"].tolist()

    st.subheader("Same shift", divider=False)
    st.caption("These pairs must always work identical shift blocks (unless one of them has a fixed FERIE/RECUPERO entry that day).")
    with st.form("same_shift_form", border=False):
        same_edited = st.data_editor(
            st.session_state.same_shift_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Name1": st.column_config.SelectboxColumn("Person 1", options=names, required=True),
                "Name2": st.column_config.SelectboxColumn("Person 2", options=names, required=True),
            },
            key="same_shift_editor",
        )
        same_shift_submitted = st.form_submit_button("Save same-shift pairs", icon=":material/save:")
    st.session_state.same_shift_df = same_edited.dropna().reset_index(drop=True)
    if same_shift_submitted:
        _save_feedback("_snap_same_shift_df", st.session_state.same_shift_df, "Same-shift pairs")

    st.space("small")
    st.subheader("Complementary staff", divider=False)
    st.caption("These pairs are softly discouraged from working the exact same hour at the same time - higher weight means a stronger preference for alternating cover.")
    with st.form("complementary_form", border=False):
        comp_edited = st.data_editor(
            st.session_state.complementary_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Name1": st.column_config.SelectboxColumn("Person 1", options=names, required=True),
                "Name2": st.column_config.SelectboxColumn("Person 2", options=names, required=True),
                "Weight": st.column_config.NumberColumn("Weight", min_value=1, max_value=100, step=1),
            },
            key="complementary_editor",
        )
        complementary_submitted = st.form_submit_button("Save complementary pairs", icon=":material/save:")
    st.session_state.complementary_df = comp_edited.dropna().reset_index(drop=True)
    if complementary_submitted:
        _save_feedback("_snap_complementary_df", st.session_state.complementary_df, "Complementary pairs")


def render_requests_tab() -> None:
    st.caption("Individual shift or day-off requests. These are soft preferences the solver tries to honour, weighted by priority - "
               "they never override coverage or rest-day rules, and a higher priority is honoured before a lower one when both can't be satisfied. "
               "A 'Prefer shift' or 'Avoid shift' request *does* overrule a RIPOSO day already set in the Fixed days tab (that's just a "
               "scheduling default), but never a FERIE or RECUPERO entry - those are real absences.")
    names = st.session_state.staff_df["Name"].tolist()
    shift_codes = [""] + [
        f"{row.Code} ({row.Start}-{row.End})" for row in st.session_state.shift_df.itertuples()
    ]
    with st.form("requests_form", border=False):
        edited = st.data_editor(
            st.session_state.requests_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Name": st.column_config.SelectboxColumn("Name", options=names, required=True),
                "Day": st.column_config.SelectboxColumn("Day", options=DAYS, required=True),
                "RequestType": st.column_config.SelectboxColumn("Request", options=REQUEST_TYPE_OPTIONS, required=True),
                "ShiftCode": st.column_config.SelectboxColumn("Shift code (if applicable)", options=shift_codes),
                "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITY_OPTIONS, required=True),
            },
            key="requests_editor",
        )
        requests_submitted = st.form_submit_button("Save requests", icon=":material/save:")
    st.session_state.requests_df = edited.dropna(subset=["Name", "Day", "RequestType", "Priority"]).reset_index(drop=True)
    if requests_submitted:
        _save_feedback("_snap_requests_df", st.session_state.requests_df, "Requests")


def render_results_tab() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        week_start = st.date_input("Week starts on (Monday)", value=st.session_state.week_start)
        st.session_state.week_start = week_start
    with col2:
        min_break = st.number_input("Min break between split shifts (hrs)", min_value=0, max_value=4,
                                     value=DEFAULT_MIN_BREAK_HOURS, step=1)
    with col3:
        max_daily = st.number_input("Max paid hours/day", min_value=4, max_value=12,
                                     value=DEFAULT_MAX_DAILY_HOURS, step=1)

    col4, col5, col6 = st.columns(3)
    with col4:
        limit_max_break = st.checkbox("Limit max break between split shifts",
                                       value=st.session_state.limit_max_break, key="limit_max_break")
    with col5:
        max_break = st.number_input("Max break (hrs)", min_value=1, max_value=8,
                                     value=DEFAULT_MAX_BREAK_HOURS_OPTION, step=1, disabled=not limit_max_break)
    with col6:
        time_limit = st.number_input("Solver time limit (sec)", min_value=5, max_value=180,
                                      value=30, step=5,
                                      help="Tight rules (e.g. 2 days off + a 40h target, which forces every "
                                           "working day to be a full split shift) need more search time.")

    if st.button("Generate schedule", type="primary", icon=":material/auto_awesome:", width="stretch"):
        if st.session_state.staff_df.empty:
            st.warning("Add at least one team member first.")
        else:
            errors, warnings = validate_inputs(
                st.session_state.staff_df,
                st.session_state.shift_df,
                st.session_state.coverage_df,
                st.session_state.max_staff_df,
                st.session_state.same_shift_df,
                st.session_state.complementary_df,
            )
            for w in warnings:
                st.warning(w, icon=":material/warning:")
            if errors:
                for e in errors:
                    st.error(e, icon=":material/error:")
            else:
                with st.spinner("Solving with CP-SAT..."):
                    result = solve_schedule(
                        st.session_state.staff_df,
                        st.session_state.shift_df,
                        st.session_state.fixed_df,
                        st.session_state.coverage_df,
                        st.session_state.same_shift_df,
                        st.session_state.complementary_df,
                        st.session_state.requests_df,
                        max_staff_df=st.session_state.max_staff_df,
                        min_break_hours=min_break,
                        max_break_hours=(max_break if limit_max_break else None),
                        max_daily_hours=max_daily,
                        solver_time_limit_sec=time_limit,
                    )
                st.session_state.result = result

    result: SolveResult | None = st.session_state.result
    if result is None:
        st.info("Fill in the Staff, Shifts, Fixed days, Coverage, Rules and Requests tabs, then generate a schedule.", icon=":material/info:")
        return

    if not result.feasible:
        if result.status_name == "UNKNOWN":
            st.warning("The solver ran out of time before proving whether a schedule exists (status: UNKNOWN). "
                       "This combination of rules is hard to search - try raising the solver time limit above, "
                       "or relax a rule (rest days, coverage caps, contract hours, same-shift/complementary rules).",
                       icon=":material/hourglass_top:")
        else:
            st.error(f"No feasible schedule found (solver status: {result.status_name}). "
                     "Try relaxing rest-day rules, coverage targets, contract hours, or the same-shift/complementary rules.",
                     icon=":material/error:")
        return

    with st.container(horizontal=True):
        st.badge(f"Solver status: {result.status_name}", icon=":material/check_circle:", color="green")
        if result.total_understaffing > 0:
            st.badge(f"Understaffed hours: {result.total_understaffing}", icon=":material/warning:", color="orange")

    staff_order = st.session_state.staff_df["Name"].tolist()
    grid = pivot_grid(result.schedule_df, staff_order, st.session_state.week_start)
    st.dataframe(grid, width="stretch")
    st.caption(":material/gpp_maybe: **Verify before using.** This schedule is generated automatically from the "
               "rules you configured - it doesn't know local labour law, verbal agreements, or anything you "
               "haven't entered. Review it before publishing or relying on it.")

    cov = coverage_check_df(result.schedule_df, st.session_state.shift_df, st.session_state.coverage_df)
    
    gaps = cov[cov["Gap"] < 0]
    if gaps.empty:
        st.success("All hours comply to minimum and maximum occupation.", icon=":material/check_circle:")
    else:
        with st.expander("Understaffed hours", icon=":material/warning:", expanded=True):
            st.warning(f"{len(gaps)} hour blocks are understaffed.", icon=":material/warning:")
            styled_gaps = gaps.style.map(lambda v: "background-color: #ffd6d6; font-weight: bold" if v < 0 else "",
                                            subset=["Gap"])
            st.dataframe(styled_gaps, width="stretch", hide_index=True)
    
    st.subheader("Staff working per hour", divider=False)
    st.caption("Rows are days, columns are hour blocks - each cell is the number of people working during that hour.")
    hc = headcount_grid(cov)
    st.dataframe(hc, width="stretch")
    
    hours_summary = result.schedule_df.groupby("Name")["Hours"].sum().reindex(staff_order)
    targets = st.session_state.staff_df.set_index("Name")["ContractHours"].reindex(staff_order)
    summary_df = pd.DataFrame({"Scheduled hours": hours_summary, "Contract target": targets})
    summary_df["Difference"] = summary_df["Scheduled hours"] - summary_df["Contract target"]
    st.dataframe(summary_df, width="stretch")

    

    csv_bytes = to_csv_bytes(grid)
    st.download_button("Download rota CSV", data=csv_bytes, file_name="rota.csv", mime="text/csv",
                        width="stretch", icon=":material/download:")

    csv_bytes_hc = to_csv_bytes(hc)
    st.download_button("Download headcount CSV", data=csv_bytes_hc, file_name="headcount.csv", mime="text/csv",
                            width="stretch", icon=":material/download:")
    
SESSION_STATE_KEYS: list[str] = [
    "staff_df", "shift_df", "fixed_df", "coverage_df", "max_staff_df",
    "arrivals_departures_df", "limit_max_break",
    "same_shift_df", "complementary_df", "requests_df", "result",
]


ABOUT_INTRO_MD = """
The Staff Rota Planner is a small web app that generates a full week's schedule automatically. You tell it who's
on the team, what the shop's rules are, and what coverage you need — it hands back a complete, valid rota in
seconds. This page explains what it does, how to use it, and (in the boxed section below) a bit about the
technology under the hood, for anyone who's curious.

## What problem does this actually solve?

Building a rota by hand means juggling a lot of rules at once, in your head, all the time:

- Everyone needs to hit their contracted hours — not more, not less.
- Everyone needs their days off, and not four working days followed by a scramble.
- Some hours of the day need more people than others; some need very few.
- A few people might have personal arrangements — always work the same shift as a colleague, never clash hours
  with another, or have a standing request for a specific day off.
- Nobody should always get stuck closing.

Any one of these is easy. All of them, at once, for eleven people, seven days a week — that's where a
spreadsheet stops helping and starts fighting you. The planner handles all of it simultaneously and tells you
honestly when a request can't be satisfied, instead of quietly producing a rota that breaks a rule you didn't
notice.

## What it guarantees, every time

- **Exact contracted hours.** If someone's contract says 40 hours, they get 40 — not "roughly 40," exactly 40.
  Each confirmed FERIE (holiday) or RECUPERO (comp day) that week lowers their target by 8 hours automatically,
  so a normal week off doesn't force an impossible schedule.
- **1 or 2 days off, your choice, per person.** With 1 day off, the six working days are automatically balanced
  into three 8-hour days, two 6-hour days, and one 4-hour day — so nobody works six long days in a row.
- **Split shifts, capped sensibly.** A day can be one block or two (e.g. a morning and an evening shift), with a
  minimum — and optionally a maximum — break in between, so nobody has an accidental 10-hour gap in the middle
  of their day.
- **Shift length capped at 6 hours.** No single block longer than that.
- **Minimum *and* maximum staffing per hour.** You set how many people you need at 9am, at 2pm, at closing — and
  also a hard ceiling, so the tool doesn't just throw extra bodies at quiet hours.
- **Fair closing duty.** You can cap how many times per week any one person closes.
- **Staff relationships.** Two people who should always share a shift, or two who should never overlap — both
  are supported directly.
- **Personal requests.** A day off, a preferred shift, or a shift to avoid — ranked by priority, honoured
  whenever it doesn't break a hard rule.

## Using it, tab by tab

The app is organised as a set of tabs, filled in roughly left to right:

1. **Staff** — one row per person: contract hours, days off (1 or 2), whether split shifts are allowed, and a
   couple of advanced options (a fixed "two 4-hour blocks" contract type, and a cap on closing shifts).
2. **Shift catalog** — the list of shift blocks the rota is built from (e.g. 9–13, 14–20). Edit this if your
   actual shift times differ.
3. **Fixed days** — mark someone's confirmed holiday, comp day, or a day off you want to lock in yourself.
   Everything else is left to the planner.
4. **Coverage** — the minimum and maximum number of people needed, hour by hour, day by day.
5. **Rules** — "these two always work together," "these two should alternate rather than overlap."
6. **Requests** — individual preferences, with a priority level.
7. **Generate & results** — press the button. You get the full week's grid, an hour-by-hour headcount table, a
   check that every coverage rule was met, and a CSV you can download.

If a rota can't be produced — say, two rules directly contradict each other — the tool says so plainly, rather
than silently giving you something wrong.
"""

ABOUT_KADER_MD = """
Under the bonnet, this tool doesn't "figure out" a schedule the way a person would, by trial and error. It
hands the whole problem — every rule, every person, every hour of the week — to a piece of software called a
**constraint solver**, which searches through the possibilities mathematically until it finds one that
satisfies everything at once (or tells you honestly that no such schedule exists).

There were, broadly, three ways we could have built this:

**1. Rule-based / hand-written logic.** Write code that fills the rota step by step, following a priority order
("first give everyone their days off, then fill morning shifts, then afternoons..."). This is the simplest to
build and the easiest to explain, but it falls apart fast: the moment two rules conflict, the code has to guess
which one to break, and that guess is baked in rather than chosen deliberately. It also tends to produce a
*valid* rota rather than a *good* one — it stops at the first thing that works, rather than the best available
option.

**2. Classic linear optimisation (e.g. a tool called PuLP).** This treats the problem as equations and
inequalities to be solved together — good for questions like "minimise total cost" or "maximise output," where
everything is a number on a scale. Rota rules, though, are mostly *logical* rather than numeric: "if this person
works the late shift, they cannot also work the early one the next day," "exactly two people, no more, no
fewer." Forcing that kind of yes/no logic into pure arithmetic works, but it's clunky, and the underlying
engines aren't built for problems shaped like this — they tend to slow down badly as the rules pile up.

**3. Constraint solving, specifically Google's OR-Tools (the "CP-SAT" engine).** This is what the planner uses.
It's built from the ground up for exactly this kind of problem — shift rosters, delivery routes, staff
schedules — where the rules are mostly logical ("this AND not that," "at most N of these") rather than purely
arithmetic. It's free, actively maintained by Google, and in practice solves rosters like ours in seconds rather
than minutes, even as we've kept adding rules (exact hours, day-type balancing, closing limits, staff pairings)
on top of each other.

In short: option 1 is fast to build but brittle; option 2 is powerful for cost-style problems but a poor fit for
scheduling logic; option 3 is purpose-built for exactly this job. That's why the tool is built on OR-Tools.

One honest caveat: a genuinely hard combination of rules (for example, "2 days off" combined with very tight
coverage limits) can take the solver longer to search, and occasionally it runs out of time before it can prove
there's *no* answer, rather than confirming there's a good one. The app tells you when that's happened and lets
you give it more time or loosen a rule — it never quietly hands you a broken rota instead.
"""

ABOUT_OUTRO_MD = """
## What's next

The tool currently exports to CSV; a formatted spreadsheet export is a natural next step if it'd be useful day
to day. Longer term, feeding in real arrival/departure numbers (rather than fixed coverage targets) could let
the rota adjust itself automatically to how busy a given week actually is.

For now: fill in the Staff and Coverage tabs, hit **Generate schedule**, and see what comes back.
"""


def render_about_tab() -> None:
    st.markdown(ABOUT_INTRO_MD)
    with st.container(border=True):
        st.markdown("### :material/engineering: Box: which engine, and why")
        st.markdown(ABOUT_KADER_MD)
    st.markdown(ABOUT_OUTRO_MD)


def render_advanced_tab() -> None:
    st.caption("A technical reference for every tunable constant in the script (all live in the CONFIGURATION "
               "BLOCK at the top of the file). Useful if you're adapting this for a different team, business "
               "hours, or set of rules.")

    st.markdown(f"""
## Business hours & calendar

- `DAYS` — the seven weekday names, fixed order. Everything else (grids, loops, constraints) iterates over this list.
- `OPEN_HOUR = {OPEN_HOUR}`, `CLOSE_HOUR = {CLOSE_HOUR}` — the business day. `HOURS` is the derived list of hour
  *blocks* (`{OPEN_HOUR}` to `{CLOSE_HOUR - 1}`), i.e. `{OPEN_HOUR}` means "the {OPEN_HOUR}:00–{OPEN_HOUR + 1}:00 block."
  Change these two and every coverage table, shift catalog bound, and closing-hour calculation follows automatically.

## Dropdown option lists

- `FIXED_STATUS_OPTIONS` — the choices in the Fixed-days grid: blank, `RIPOSO`, `FERIE`, `RECUPERO`. Add a new
  status here first if you need one; then teach the solver what it means (search for `"FERIE", "RECUPERO"` in
  the code — every place absences are excluded from hours/rest-day counting checks against this exact tuple).
- `REQUEST_TYPE_OPTIONS` — the three request kinds in the Requests tab (prefer day off / prefer shift / avoid shift).
- `PRIORITY_OPTIONS` + `PRIORITY_WEIGHTS = {PRIORITY_WEIGHTS}` — Low/Medium/High map to these raw weights, which
  then get multiplied by `REQUEST_WEIGHT_SCALE` (see the weights section below).
- `REST_DAYS_OPTIONS = {REST_DAYS_OPTIONS}` — deliberately restricted to exactly these two choices. Widen this
  list if you want to offer 0 or 3+ days off; the 3×8h/2×6h/1×4h day-mix rule only triggers for the value `1`,
  so a wider range wouldn't automatically get an equivalent rule.

## Contract types

- `CONTRACT_TYPE_FLEXIBLE` / `CONTRACT_TYPE_FIXED_SPLIT` — the two options in the Staff tab's "Contract type" column.
- `FIXED_SPLIT_SHIFT_HOURS = {FIXED_SPLIT_SHIFT_HOURS}` — how long each of the two blocks is for the fixed-split contract.
- `FIXED_SPLIT_MIN_BREAK = {FIXED_SPLIT_MIN_BREAK}`, `FIXED_SPLIT_MAX_BREAK = {FIXED_SPLIT_MAX_BREAK}` — the
  break between those two blocks must land inside this range (inclusive).
- `FIXED_SPLIT_MIN_REST_DAYS = {FIXED_SPLIT_MIN_REST_DAYS}` — the floor on rest days this contract type enforces,
  regardless of what's set in the person's own "Days off/week" column.

## Shifts & breaks

- `MAX_SHIFT_LENGTH_HOURS = {MAX_SHIFT_LENGTH_HOURS}` — hard cap on any single shift block's length. Enforced
  both on the default catalog and on any row you add in the Shift catalog tab (longer rows get dropped with a warning).
- `DEFAULT_MIN_BREAK_HOURS = {DEFAULT_MIN_BREAK_HOURS}` — minimum gap between the two halves of *any* split shift
  (not just the fixed-split contract type). Exposed as an editable field in Generate & results.
- `DEFAULT_MAX_BREAK_HOURS_OPTION = {DEFAULT_MAX_BREAK_HOURS_OPTION}` — the value pre-filled when someone switches
  on "Limit max break between split shifts" (off by default; that field itself has no cap unless switched on).
- `DEFAULT_MAX_DAILY_HOURS = {DEFAULT_MAX_DAILY_HOURS}` — hard cap on paid hours per person per day. This
  quietly does a lot of work: with 4h/6h shifts, a cap of {DEFAULT_MAX_DAILY_HOURS} is exactly what forces "two
  shifts in a day" to always mean two 4h shifts (4+6=10 and 6+6=12 both exceed it) — several other rules
  (the day-mix rule, the fixed-split contract) rely on that fact.

## Hours & absences

- `HOURS_PER_FERIE_RECUPERO_DAY = {HOURS_PER_FERIE_RECUPERO_DAY}` — each FERIE/RECUPERO day in a week subtracts
  this many hours from that person's target for the week, rather than forcing the full target into fewer days.
- `STRICT_WEEKLY_HOURS = {STRICT_WEEKLY_HOURS}` — if `True`, weekly hours must equal the (absence-adjusted)
  target *exactly* (a hard constraint). Set to `False` to make it a soft target instead — the solver will get as
  close as it can, weighted by the deviation term described below, rather than refusing to solve if it can't hit
  the number exactly.

## Closing shifts

- `DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK = {DEFAULT_MAX_CLOSING_SHIFTS_PER_WEEK}` — default value for each person's
  "Max closing shifts/week" field (`0` = unlimited). A "closing" shift is computed dynamically as whichever
  shift(s) in the *current* catalog end at the latest hour — so this keeps working correctly even if you change
  the catalog's closing time.

## Coverage defaults and overrides

The Coverage tab's starting values (both the minimum and maximum tables) aren't one flat number — they're built
from a small set of constants, then specific hours are overridden on top. This is the part of the config most
worth understanding if your business has a different daily shape than ours.

**Minimum staffing** (`DEFAULT_HOURLY_MIN_BY_DAY`, built from three pieces):

- `_BASELINE_MIN = {_BASELINE_MIN}` — the fallback minimum for any hour not otherwise specified. Quiet hours
  (open, mid-afternoon lull, last hour before close) sit at this value by default.
- `_PEAK_MIN = {_PEAK_MIN}` — the minimum during `_PEAK_HOURS`.
- `_PEAK_HOURS = {sorted(_PEAK_HOURS)}` — the hour blocks considered "busy": {min(_PEAK_HOURS)}h–{max([h for h in _PEAK_HOURS if h < 13]) + 1}h
  and {min([h for h in _PEAK_HOURS if h >= 13])}h–{max(_PEAK_HOURS) + 1}h (the two coverage windows from the
  original brief). Every hour in this set gets `_PEAK_MIN` instead of `_BASELINE_MIN`.
- `_MIN_OVERRIDES = {_MIN_OVERRIDES}` — applied *last*, after baseline/peak, so it wins over both. Right now
  this pins hour {list(_MIN_OVERRIDES.keys())[0]} down to {list(_MIN_OVERRIDES.values())[0]} even though it
  falls inside `_PEAK_HOURS` (which would otherwise give it `_PEAK_MIN = {_PEAK_MIN}`). Add more entries here for
  any other hour that needs to break the general peak/baseline pattern, on every day at once.

Build order, in plain terms: *start every hour at `_BASELINE_MIN`* → *raise the hours in `_PEAK_HOURS` to
`_PEAK_MIN`* → *apply `_MIN_OVERRIDES` on top of anything that came before.* The same value currently applies to
all seven days — the Coverage tab itself lets you diverge per day after the fact (e.g. a busier Saturday), this
constant just controls what you start from.

**Maximum staffing** (`build_default_max_staff_df`, built from two pieces):

- `DEFAULT_MAX_STAFF_PER_HOUR = {DEFAULT_MAX_STAFF_PER_HOUR}` — the ceiling for every hour by default. Deliberately
  generous so it rarely binds unless you tighten it.
- `_MAX_STAFF_OVERRIDES = {_MAX_STAFF_OVERRIDES}` — specific hours pinned to a tighter (or looser) ceiling than
  the default. Right now: a quiet cap right at opening ({list(_MAX_STAFF_OVERRIDES.keys())[0]}h), a cap matching
  the {list(_MIN_OVERRIDES.values())[0]}-person minimum above so that hour is pinned exactly
  ({list(_MAX_STAFF_OVERRIDES.keys())[1]}h), and two quiet hours near close
  ({list(_MAX_STAFF_OVERRIDES.keys())[2]}h, {list(_MAX_STAFF_OVERRIDES.keys())[3]}h).

Both tables are only *starting points* — every cell in both the minimum and maximum Coverage grids is editable
per day, per hour, in the UI. These constants just decide what's pre-filled the first time (or after "Reset all
to code defaults" in the sidebar). If a minimum ever ends up higher than the maximum for the same hour, the
solver will correctly report `INFEASIBLE` rather than silently picking one — that's a genuine contradiction, not
a bug.

## Solver

- `DEFAULT_SOLVER_TIME_LIMIT_SEC = {DEFAULT_SOLVER_TIME_LIMIT_SEC}` — the constant's own default; the actual
  field in Generate & results currently defaults to 30 seconds. Tight rule combinations (e.g. 2 days off with a
  high hours target, which forces every working day into a full split shift) need more search time — raise this
  if you see solver status `UNKNOWN` rather than `OPTIMAL`/`FEASIBLE`/`INFEASIBLE`.
""")

    st.markdown("## The objective weights, explained")
    st.markdown(f"""
The solver doesn't just find *a* valid schedule — every rule that isn't a hard constraint gets a cost, and it
finds the *cheapest* valid schedule. Understanding the relative size of these numbers is the key to understanding
why it makes the trade-offs it does.

| Constant | Value | What it costs |
|---|---|---|
| `WEIGHT_UNDERSTAFF` | **{WEIGHT_UNDERSTAFF}** | per person-hour below the Coverage tab's minimum. Deliberately huge — coverage is sacrificed only when there is truly no other way to satisfy every hard constraint. |
| `WEIGHT_OVERSTAFF` | **{WEIGHT_OVERSTAFF}** | per person-hour above the minimum, for hours *outside* the morning/afternoon windows below. |
| `WEIGHT_OVERSTAFF_MORNING` | **{WEIGHT_OVERSTAFF_MORNING}** | per extra person-hour between {min(MORNING_PRIORITY_HOURS)}h–{max(MORNING_PRIORITY_HOURS) + 1}h. Deliberately cheap. |
| `WEIGHT_OVERSTAFF_AFTERNOON` | **{WEIGHT_OVERSTAFF_AFTERNOON}** | per extra person-hour between {min(AFTERNOON_HOURS)}h–{max(AFTERNOON_HOURS) + 1}h. Deliberately expensive. |
| hours-deviation weight | **3** (hardcoded) | per hour away from someone's target, but *only* when `STRICT_WEEKLY_HOURS = False`. Irrelevant while hours are a hard constraint. |
| `REQUEST_WEIGHT_SCALE` | **{REQUEST_WEIGHT_SCALE}** | multiplies `PRIORITY_WEIGHTS` ({PRIORITY_WEIGHTS}) for each request, so Low/Medium/High become {PRIORITY_WEIGHTS['Low']*REQUEST_WEIGHT_SCALE}/{PRIORITY_WEIGHTS['Medium']*REQUEST_WEIGHT_SCALE}/{PRIORITY_WEIGHTS['High']*REQUEST_WEIGHT_SCALE}. |
| complementary-pair weight | *set per pair in the Rules tab* | penalty for two linked people working the same hour simultaneously; no global constant, it's a column in that table (default {DEFAULT_COMPLEMENTARY_PAIRS[0]['Weight']}). |
| `WEIGHT_DAYS_OFF_TOGETHER` | **{WEIGHT_DAYS_OFF_TOGETHER}** | per person with exactly 2 days off/week: rewards (or, if their "Days off together" checkbox is unchecked, penalises) their 2 rest days landing on adjacent calendar days. No effect with 1 day off. |

Why {WEIGHT_UNDERSTAFF} for understaffing but only {WEIGHT_OVERSTAFF} for overstaffing? Because with everyone on
an exact hours target, the total contracted hours in the week is usually well above the *minimum* coverage
needs — that surplus has to go somewhere, and some overstaffing is therefore normal and expected, not a
failure. Understaffing, by contrast, means a rule you explicitly set (the minimum) wasn't met, so it's punished
far more harshly. The morning/afternoon split then decides *where* that inevitable surplus lands: {WEIGHT_OVERSTAFF_MORNING}
vs {WEIGHT_OVERSTAFF_AFTERNOON} means the solver will always prefer stacking extra people into the morning window
before the afternoon one, purely because it's mathematically ten times cheaper to do so.

A practical way to retune this: raising `WEIGHT_UNDERSTAFF` further makes coverage even more untouchable at the
cost of possibly infeasible-looking results elsewhere; lowering it lets the solver trade away a little coverage
to satisfy more requests or balance hours more evenly. `REQUEST_WEIGHT_SCALE` is the single dial for "how much
do personal requests matter compared to everything else" — since {PRIORITY_WEIGHTS['High']*REQUEST_WEIGHT_SCALE}
(a High request) is still far below {WEIGHT_UNDERSTAFF}, no request can ever be granted at the cost of leaving
an hour understaffed.
""")

    with st.container(border=True):
        st.markdown("### :material/memory: Box: the algorithm, for the technically curious")
        st.markdown(f"""
This app uses **Google OR-Tools' CP-SAT solver** (`ortools.sat.python.cp_model`) — a Constraint Programming
solver with a SAT (Boolean satisfiability) core underneath. In practice that means every decision in the model
— "does person X work shift Y on day Z?" — is a Boolean variable, and every rule in this document is one or
more linear or logical constraints over those variables (`model.Add(...)`, `model.AddBoolOr(...)`,
`model.AddMaxEquality(...)`, and so on). The objective (the weights table above) is a single linear expression
the solver minimises subject to every hard constraint holding.

A few implementation details, if you're reading the code:

- **Search parallelism**: `solver.parameters.num_search_workers = 8` runs multiple search strategies in parallel
  threads and takes the best result found by any of them — this is a large part of why problems that would be
  slow single-threaded solve in seconds here.
- **Time limit**: `solver.parameters.max_time_in_seconds` is the field you set in Generate & results. CP-SAT is
  a complete solver — given enough time it will always find the optimal answer *or* prove none exists — the
  time limit just controls how long it's allowed to try before reporting its best-so-far answer instead.
- **Solver statuses**: `OPTIMAL` (best possible answer found and proven optimal), `FEASIBLE` (a valid answer
  found, but time ran out before proving it's the best one — still a fully valid schedule), `INFEASIBLE`
  (proven mathematically that no schedule satisfies every hard constraint), `UNKNOWN` (time ran out before
  finding *any* answer or proving infeasibility — try a longer time limit).
- **Why not a generic LP/MIP solver** (like the PuLP-based approach mentioned in the About tab): most of this
  model's constraints are Boolean logic ("at most one of these," "if this then not that") rather than pure
  arithmetic. CP-SAT's propagation engine reasons about that logic natively and prunes huge swathes of the
  search space before ever doing arithmetic — which is exactly what makes rota-style problems tractable at all.

Package: [`ortools`](https://developers.google.com/optimization) (Apache 2.0 licence, maintained by Google).
Docs for the CP-SAT solver specifically: the
[CP-SAT primer](https://developers.google.com/optimization/cp/cp_solver).
""")

    with st.container(border=True):
        st.markdown("### :material/rocket_launch: Box: deploying your own copy (GitHub + Streamlit Cloud)")
        st.markdown("""
This app is a single `.py` file with no server-side state beyond one browser session, so putting it online is
mostly a GitHub + Streamlit Cloud exercise, not a real deployment project.

1. **Get the file into a GitHub repo.** For now it lives alongside other scripts here:
   [rcsmit/streamlit_scripts/staff_rota_planner.py](https://github.com/rcsmit/streamlit_scripts/blob/main/staff_rota_planner.py).
   A dedicated repo of its own is a natural next step later, once/if this grows beyond "one script in a shared
   folder" - nothing about the app itself needs to change for that, it's purely a matter of moving the file and
   updating the deploy target.
2. **Adjust the constants for your own situation** directly in that file before deploying - everything in the
   sections above (business hours, shift catalog, coverage windows, weights, and so on) lives in the
   configuration block at the top, so there's no code logic to touch, just values. Beware that the information
   is public, so use abbrevations for privacy reasons etc.
3. **Add a `requirements.txt`** alongside the script listing `streamlit`, `pandas`, and `ortools` (the three
   dependencies noted at the top of the file) - Streamlit Cloud installs from this automatically.
4. **Deploy it**: go to [share.streamlit.io](https://share.streamlit.io), sign in with the GitHub account that
   owns the repo, click "New app," point it at the repo/branch and at `staff_rota_planner.py`, and deploy. Any
   push to that branch afterwards redeploys automatically.

Anyone with the app's URL can then use it directly in a browser - no local Python install needed on their end.
""")


def main() -> None:
    st.set_page_config(page_title="Staff rota planner", page_icon=":material/event_available:", layout="wide",
                        initial_sidebar_state="collapsed")
    init_session_state()

    with st.sidebar:
        st.caption("Session data (staff, shifts, coverage, etc.) persists across reruns. "
                   "If you've changed the app's defaults and still see old values, reset below.")
        if st.button("Reset all to code defaults", icon=":material/restart_alt:", width="stretch"):
            for key in SESSION_STATE_KEYS:
                st.session_state.pop(key, None)
            st.rerun()

    st.title("Staff rota planner")
    st.caption("OR-Tools CP-SAT builds a weekly rota from your team's contract hours, rest-day rules, "
               "fixed vacation/comp days, per-day coverage targets, staff relationships, and shift/day-off requests.")

    tab_staff, tab_shifts, tab_fixed, tab_coverage, tab_rules, tab_requests, tab_results, tab_about, tab_advanced = st.tabs([
    
        ":material/group: Staff",
        ":material/schedule: Shift catalog",
        ":material/event_busy: Fixed days",
        ":material/bar_chart: Arrivals, departures & coverage",
        ":material/link: Rules",
        ":material/how_to_reg: Requests",
        ":material/auto_awesome: Generate & results",
        ":material/menu_book: About",
        ":material/tune: Advanced",
    ])

    with tab_about:
        render_about_tab()
    with tab_staff:
        render_staff_tab()
    with tab_shifts:
        render_shift_tab()
    with tab_fixed:
        render_fixed_tab()
    with tab_coverage:
        render_coverage_tab()
    with tab_rules:
        render_rules_tab()
    with tab_requests:
        render_requests_tab()
    with tab_results:
        render_results_tab()
    with tab_advanced:
        render_advanced_tab()


if __name__ == "__main__":
    main()