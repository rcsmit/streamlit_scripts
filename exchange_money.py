# WHAT IS THE BEST WAY TO EXCHANGE MONEY IN THAILAND?
# Exchange cash or take it from ATM
#
# Calculation to find the Break Even Point

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly

from utils import get_data_yfinance  # kept for future use if needed


# --- INPUTS / PARAMETERS ------------------------------------------------------

rate_yfinance = st.sidebar.number_input(
    "Rate EUR-THB", 0.0, 100.0, 37.4228347
)

# Credit card
cost_creditcard_fix = st.sidebar.number_input(
    "Fixed costs credit card (EUR)", 0.0, 100.0, 4.5
)
cost_creditcard_variable = 1 + (
    st.sidebar.number_input(
        "Koersopslag creditcard (%)", 0.0, 100.0, 2.0
    )
    / 100
)
rate_cc = rate_yfinance / cost_creditcard_variable

# Debit card
cost_debitcard_fix = st.sidebar.number_input(
    "Fixed costs debit card (EUR)", 0.0, 100.0, 0.0
)
cost_debitcard_variable = 1 + (
    st.sidebar.number_input(
        "Koersopslag debitcard (%)", 0.0, 100.0, 1.4
    )
    / 100
)
rate_dc = rate_yfinance / cost_debitcard_variable

# ATM fee in Baht
cost_atm = 250

# Street exchange rate
rate_street_default = rate_yfinance * 0.98
rate_street = st.sidebar.number_input(
    "Rate street", 0.0, 100.0, rate_street_default
)

# “Bad” conversion rate (POS/ATM conversion)
rate_with_conversion = 34.76008 / 37.4228347 * rate_yfinance


fixed_amount_baht = st.sidebar.number_input(
    "Fixed amount baht", 0.0, 100_000.0, 30_000.0
)
fixed_amount_euro = st.sidebar.number_input(
    "Fixed amount euro", 0.0, 10_000.0, 800.0
)


# --- CORE CALCULATION FUNCTIONS ----------------------------------------------


def calculate_from_euro(euro: float):
    """
    Input: euro you spend
    Output: Baht you receive with different methods.
    """
    cc = (euro - cost_creditcard_fix - (cost_atm / rate_cc)) * rate_cc
    dc = (euro - cost_debitcard_fix - (cost_atm / rate_dc)) * rate_dc

    cc_with_conv = (
        euro - cost_creditcard_fix - (cost_atm / rate_with_conversion)
    ) * rate_with_conversion
    dc_with_conv = (
        euro - cost_debitcard_fix - (cost_atm / rate_with_conversion)
    ) * rate_with_conversion

    street = euro * rate_street
    return cc, dc, cc_with_conv, dc_with_conv, street


def calculate_from_baht(baht: float):
    """
    Input: Baht you want to withdraw
    Output: Euro cost for different methods.
    """
    cc = (baht + cost_atm) / rate_cc + cost_creditcard_fix
    dc = (baht + cost_atm) / rate_dc + cost_debitcard_fix

    cc_with_conv = (baht + cost_atm) / rate_with_conversion + cost_creditcard_fix
    dc_with_conv = (baht + cost_atm) / rate_with_conversion + cost_debitcard_fix

    street = baht / rate_street
    return cc, dc, cc_with_conv, dc_with_conv, street


# --- WRAPPERS FOR DISPLAY -----------------------------------------------------


def from_baht():
    st.subheader("From Baht to Euro")
    rows = []

    for baht in range(0, 35_000, 5_000):
        cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_baht(baht)
        rows.append(
            [
                baht,
                round(cc, 2),
                round(dc, 2),
                round(cc_with_conv, 2),
                round(dc_with_conv, 2),
                round(street, 2),
            ]
        )
        # if round(street, 2) == round(cc, 2):
        #     st.write(f"Street is same as cc {baht} - {cc}")
        # if round(street, 2) == round(dc, 2):
        #     st.write(f"Street is same as dc {baht} - {dc}")

    total_df_baht = pd.DataFrame(
        rows,
        columns=[
            "baht",
            "creditcard",
            "debitcard",
            "creditcard_with_conv",
            "debitcard_with_conv",
            "street",
        ],
    )
    st.write(total_df_baht)
    fig = px.line(
        total_df_baht,
        x="baht",
        y=["creditcard", "debitcard", "street"],
        title="From Baht to Euro",
    )
    st.plotly_chart(fig)


def from_euro():
    st.subheader("From Euro to Baht")
    rows = []

    for euro in range(0, 1_000, 10):
        cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_euro(euro)

        cc = max(cc, 0)
        dc = max(dc, 0)
        cc_with_conv = max(cc_with_conv, 0)
        dc_with_conv = max(dc_with_conv, 0)

        rows.append(
            [
                euro,
                round(cc, 2),
                round(dc, 2),
                round(cc_with_conv, 2),
                round(dc_with_conv, 2),
                round(street, 2),
            ]
        )
        # if int(street) == int(cc):
        #     st.write(f"Street is same as cc {euro} - {cc}")
        # if int(street) == int(dc):
        #     st.write(f"Street is same as dc {euro} - {dc}")

    total_df = pd.DataFrame(
        rows,
        columns=[
            "euro",
            "creditcard",
            "debitcard",
            "creditcard_with_conv",
            "debitcard_with_conv",
            "street",
        ],
    )
    st.write(total_df)
    fig = px.line(
        total_df,
        x="euro",
        y=[
            "creditcard",
            "debitcard",
            "creditcard_with_conv",
            "debitcard_with_conv",
            "street",
        ],
        title="From Euro to Baht",
    )
    st.plotly_chart(fig)


def how_much_euro_do_i_get_for_x_baht(baht: float):
    cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_baht(baht)
    st.write(
        f"BAHT {baht} -> EURO<br> "
        f"cc={cc:.2f}<br> "
        f"dc={dc:.2f}<br> "
        f"cc_with_conv={cc_with_conv:.2f}<br>"
        f"dc_with_conv={dc_with_conv:.2f}<br>"
        f"street={street:.2f}<br>",  unsafe_allow_html=True,
    )


def how_much_baht_do_i_get_for_x_euro(euro: float):
    cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_euro(euro)
    st.write(
        f"EURO {euro} -> BAHT<br>cc={cc:.2f}<br>dc={dc:.2f}<br>cc_with_conv={cc_with_conv:.2f}<br>dc_with_conv={dc_with_conv:.2f}<br>street={street:.2f}",  unsafe_allow_html=True
    )


# --- MAIN ---------------------------------------------------------------------


def main():
    st.subheader("Currency Exchange Calculation")
    st.info(
        """
Compare Euros to Baht and Baht to Euros for different methods (credit card,
debit card, street exchange). Find the break-even point where different methods
give the same value.

Exchange Rates:
- YFinance rate: EUR/THB rate from Yahoo Finance.
- Street rate: 98% of the YFinance rate.

Costs:
- Credit card: fixed + variable fee.
- Debit card: fixed + variable fee.
- Conversion rate: worse POS/ATM conversion.
- ATM fee: per transaction (both credit and debit).
"""
    )
    st.write(
        f"""rate_yfinance={rate_yfinance} <br> 
rate_street={rate_street}<br>
rate_dc={rate_dc}<br>
rate_cc={rate_cc}<br>
rate_with_conversion={rate_with_conversion}""",
        unsafe_allow_html=True,
    )

    from_baht()
    from_euro()
    how_much_euro_do_i_get_for_x_baht(fixed_amount_baht)
    how_much_baht_do_i_get_for_x_euro(fixed_amount_euro)


if __name__ == "__main__":
    main()

