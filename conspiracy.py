# reproduction of https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147905

import streamlit as st
import math
import plotly.express as px
import pandas as pd
from numpy import log as ln
from keys import * # Imports twitter keys so they're not on github :) 
# from keys_dummy import *

if 'TEST' in locals():
    st.write("Nope")
else:
    st.write (TEST)
d = []
df = pd.DataFrame()

N0 = st.sidebar.number_input("Population",0,None,5000)
thalf = st.sidebar.number_input("Halveringstijd populatie",0,None,10) #halveringstijd
psi  = 1-st.sidebar.number_input("Kans op lekken",0.0,None,  5 * 10**-6,format="%.8f" )
alpha, beta,te = 1* 10**4,0.085,0

for t in range (61):
    Nt_halfwaarde = N0 * 0.5**(t/thalf)
    Nt_gompertz = N0 * math.exp ( (alpha/beta)*(1-(math.exp(beta*(t+te)))))

    L1 = 1 - math.exp(-t*(1-psi**N0))
    L2 = 1 - math.exp(-t*(1-psi**Nt_halfwaarde))
    L3 = 1 - math.exp(-t*((1-psi)**Nt_gompertz))

    d.append({
            'x': t,
            'y1': L1,
            'y2':L2,
            'y3':L3,
            'Nt_halfwaarde':Nt_halfwaarde,
            'Nt_gompertz':Nt_gompertz })

df = pd.DataFrame(d)
print (df)
st.write("Kans")
y2_txt = f'N halves every {thalf} years'
fig = px.line(df,"x",["y1","y2","y3"])
newnames =  {'y1':'fixed N', 'y2': y2_txt, 'y3':"N follows Gompertz curve"}
# rename columns for legend: https://stackoverflow.com/a/64378982/4173718
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )

#fig.show()
st.plotly_chart(fig)

st.write("Populatie in de tijd")
fig2 = px.line(df,"x",["Nt_halfwaarde", "Nt_gompertz"])
#fig.show()
st.plotly_chart(fig2)

