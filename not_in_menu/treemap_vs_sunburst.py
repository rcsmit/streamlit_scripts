from vizmath import rad_treemap as rt # pip install vizmath==0.0.9
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# make a treemap as described in 
# https://towardsdatascience.com/radial-treemaps-extending-treemaps-to-circular-mappings-7b47785191da
# make also a sunburst chart with plotly with same data to compare       

def make_df():
    """Creates the df with some given data

    Returns:
        df: df
    """    
    
    data = [
        ['a1', 'b1', 'c1', 9.3],
        ['a1', 'b1', 'c2', 6.7],
        ['a1', 'b1', 'c3', 2.4],
        ['a1', 'b2', 'c1', 4.5],
        ['a1', 'b2', 'c2', 3.1],

        ['a2', 'b1', 'c1', 5.9],
        ['a2', 'b1', 'c2', 32.3],
        ['a2', 'b1', 'c3', 12.3],
        ['a2', 'b1', 'c4', 2.3],
        ['a2', 'b2', 'c1', 9.1],
        ['a2', 'b2', 'c2', 17.3],
        ['a2', 'b2', 'c3', 6.7],
        ['a2', 'b2', 'c4', 4.4],
        ['a2', 'b2', 'c5', 11.3],

        ['a3', 'b1', 'c1', 7.5],
        ['a3', 'b1', 'c2', 9.5],
        ['a3', 'b2', 'c3', 17.1],

        ['a4', 'b2', 'c1', 5.1],
        ['a4', 'b2', 'c2', 2.1],
        ['a4', 'b2', 'c3', 11.1],
        ['a4', 'b2', 'c4', 1.5]]

    data_ = [
        ['a1', 'b1', 'c1', 12.3],
        ['a1', 'b2', 'c1', 4.5],
        ['a2', 'b1', 'c2', 32.3],
        ['a1', 'b2', 'c2', 2.1],
        ['a2', 'b1', 'c1', 5.9],
        ['a3', 'b1', 'c1', 3.5],
        ['a4', 'b2', 'c1', 3.1]]
    df = pd.DataFrame(data, columns = ['a', 'b', 'c', 'value'])
    return df

def make_treemap(df):
   

     # C:\Users\rcxsm\AppData\Local\Programs\Python\Python310\lib\site-packages\vizmath

    # change line 296 in C:\Users\rcxsm\AppData\Local\Programs\Python\Python310\Lib\site-packages\vizmath\radial_treemap.py
    # st.pyplot (fig) 
    # also add 'import streamlit as st' at top

    # create a rad_treemap object
    #   > df: DataFrame with 1 or more categorical columns of data
    #     and an optional 'value' column for the areas
    #     (otherwise groups counts are used for areas)
    #   > groupers: group-by columns
    #   > value: optional value column
    #   > r1, r2: inner and outer radius positions
    #   > a1, a2: start and end angle positions
    #   > rotate_deg: overall rotation around the center
    #   > mode: container orientation method
    #   > other options: 'points', 'default_sort', 'default_sort_override',
    #     'default_sort_override_reversed', 'mode', 'no_groups', 'full'
      
    # Radial Treemap chart object
    rt_obj = rt(df=df, groupers=['a','b','c'], value='value', 
        r1=0.5, r2=1, a1=0, a2=180, rotate_deg=-90 ,mode='legend')
     # plot the Radial Treemap
    rt_obj.plot_levels(level=3, fill='w')
    rt_df = rt_obj.to_df()
    rt_df['type'] = 'chart'

    # Radial Treemap legend object
    rt_legend_obj = rt(df=df, groupers=['a','b','c'], value='value', 
        r1=1.04, r2=1.09, a1=0, a2=180, rotate_deg=-90 ,mode='legend',
        no_groups=True)

    rt_legend_df = rt_legend_obj.to_df()
    rt_legend_df['type'] = 'legend'

    # export the drawing data
    df_out = pd.concat([rt_df, rt_legend_df], axis=0)
    st.write(df_out)
    # df_out.to_csv(os.path.dirname(__file__) + '/radial_treemap.csv', 
    # encoding='utf-8', index=False)


def show_sunburst_annotated(df):
    """generate a sunburst plot in plotly

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    # https://stackoverflow.com/questions/70129355/value-annotations-around-plotly-sunburst-diagram
    
    
    from math import sin,cos,pi

    aaa = 'a'
    bbb = 'b'
    ccc = 'c'
    ddd = "value"

    fig = px.sunburst(df, path=[aaa, bbb, ccc], values=ddd, width=600, height=600, title=f"Test",)
    totals_groupby =  df.groupby([aaa, bbb, ccc]).sum()
    totals_groupby["aaa_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=aaa).transform('sum')
    totals_groupby["aaa_bbb_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=[aaa,bbb]).transform('sum')
    totals_groupby["aaa_bbb_ccc_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=[aaa,bbb,ccc]).transform('sum')
    totals_groupby = totals_groupby.sort_values(by=["aaa_sum","aaa_bbb_sum","aaa_bbb_ccc_sum"], ascending=[0,0,0])
    
    ## calculate the angle subtended by each category
    sum_ddd = getattr(df,ddd).sum()
    delta_angles = 360*totals_groupby[ddd] / sum_ddd
   
    annotations = [format(v,".0f") for v in  getattr(totals_groupby,ddd).values]
    
    ## calculate cumulative sum starting from 0, then take a rolling mean 
    ## to get the angle where the annotations should go
    angles_in_degrees = pd.concat([pd.DataFrame(data=[0]),delta_angles]).cumsum().rolling(window=2).mean().dropna().values

    def get_xy_coordinates(angles_in_degrees, r=1):
        return [r*cos(angle*pi/180) for angle in angles_in_degrees], [r*sin(angle*pi/180) for angle in angles_in_degrees]
 
    x_coordinates, y_coordinates = get_xy_coordinates(angles_in_degrees, r=1.5)
    fig.add_trace(go.Scatter(
        x=x_coordinates,
        y=y_coordinates,
        mode="text",
        text=annotations,
        hoverinfo="skip",
        textfont=dict(size=14)
    ))

    padding = 0.50
    fig.update_layout( margin=dict(l=20, r=20, t=20, b=20),
        
        xaxis=dict(
            range=[-1 - padding, 1 + padding], 
            showticklabels=False
        ), 
        yaxis=dict(
            range=[-1 - padding, 1 + padding],
            showticklabels=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=800,
        height=800,
    )
    st.plotly_chart(fig)

def main():
    df=make_df()
    st.write (df)
    make_treemap(df)
    show_sunburst_annotated(df)


if __name__ == "__main__":
    main()
