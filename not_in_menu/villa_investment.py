import streamlit as st

def calculate_roi(start,end,years):
    roi_ = (end/start)**(1/years)-1
    print(end/start)
    print (1/years )
    print (roi_)
    roi = round(roi_*100,2)
    return roi
def main():
    interest = st.sidebar.number_input("Interest  %", 0,20,5)
    inflation = st.sidebar.number_input("inflation rent  %", 0,20,3)
    investment = st.sidebar.number_input("Investment", 0,None,6_200_000)
    initial_rent = st.sidebar.number_input("Initial rent month", 0,None,62_000)

    looptijd =  st.sidebar.number_input("Time in years", 0,100,30)

    current_rent = initial_rent
    total_investment = investment
    total_rent = (initial_rent*12)


    for y in range (1,looptijd+1):
        current_rent =  current_rent* (1+(inflation/100))

        total_investment = total_investment* (1+(interest/100))
        total_rent += (current_rent*12)
        
        print (f"After {y} years: current rent/month: {int(current_rent)} -total rent:  {int(total_rent)} - total_investment {int(total_investment)}")
    print (f"Total rent - initial investment =  {int(total_rent)} - {int(investment)}  = {int(total_rent-investment)}")
    print (f"ROI {calculate_roi (investment,total_rent-investment, looptijd)}")

main()