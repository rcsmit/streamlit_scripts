from vizmath import rad_treemap as rt # pip install vizmath==0.0.9
import pandas as pd

def make_df():
  # using the example data from above:
  data = [
      ['a1', 'b1', 'c1', 12.3],
      ['a1', 'b2', 'c1', 4.5],
      ['a2', 'b1', 'c2', 32.3],
      ['a1', 'b2', 'c2', 2.1],
      ['a2', 'b1', 'c1', 5.9],
      ['a3', 'b1', 'c1', 3.5],
      ['a4', 'b2', 'c1', 3.1]]
  df = pd.DataFrame(data, columns = ['a', 'b', 'c', 'value'])
  retun df



def make_treemap (df):
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
  rt_1 = rt(df=df, groupers=['a','b','c'], value='value', r1=0.5, r2=1,
    a1=0, a2=180, rotate_deg=-90, mode='alternate')

  # plot the Radial Treemap
  rt_1.plot_levels(level=3, fill='w')

def main():
  df=make_df()
  make_treemap(df)

main()
