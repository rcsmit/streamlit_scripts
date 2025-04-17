import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

def main():
    # Streamlit app title
    st.title("Compound ROI Calculator")

    # User inputs
    st.sidebar.header("Input Parameters")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"), min_value=pd.to_datetime("1900-01-01"))
    end_date = st.sidebar.date_input("End Date")
    asset_code = st.sidebar.selectbox(
        "Select Asset Code",
        options=["AAPL", "BTC-USD", "EURUSD=X", "^GSPC (S&P 500)", "^AEX (AEX Index)", "ETH-USD (Ethereum)", "^DJI (Dow Jones)", "^IXIC (NASDAQ)"],
        format_func=lambda x: x.split(" ")[0]  # Display only the asset code in the dropdown
    )

    # Fetch data from yfinance
    # if st.sidebar.button("Calculate"):
    if 1==1:
        asset_code = asset_code.split(" ")[0]
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            try:
                data = yf.download(asset_code, start=start_date, end=end_date)
                if data.empty:
                    st.error("No data found for the given asset code and date range.")
                else:
                    
                    data.columns = ['_'.join(col) for col in data.columns]
                   
                    data["Close"] = data[f"Close_{asset_code}"]
                    # Calculate ROI and compound ROI
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    roi = ((end_price - start_price) / start_price) * 100
                    num_days = (end_date - start_date).days
                    compound_roi = ((end_price / start_price) ** (1 / (num_days / 365)) - 1) * 100

                    col1,col2 = st.columns([2,1])
                    with col1:
                        # Plot data using Plotly
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
                        fig.update_layout(title=f"{asset_code} Price Chart", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(fig)
                    with col2:
                    
                        # Display results
                        st.subheader("Results")
                        st.write(f"**Start Price at {start_date}:** {start_price:.2f}")
                        st.write(f"**End Price at {end_date}:** {end_price:.2f}")
                        st.write(f"**Number of years:** {num_days / 365:.2f}")
                        st.write(f"**Compound ROI (Annualized):** {compound_roi:.2f}%")
                        st.write(f"**ROI:** {roi:.2f}%")
                    
                    

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Enter values and press Calculate")
if __name__ == "__main__":
    
    main()
