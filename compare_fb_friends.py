import pandas as pd
url1 ="C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\friendlist2012.csv"
url2 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\facebookfriends_nov2018.csv"
#url3 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\facebookfriends_nov2020.csv"
url3 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\facebookfriends_nov_2020b.csv"
#url4 ="C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\friends_aug2021.csv"
url4 ="C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\facebookfrends_aug_2021.csv"
df1 = pd.read_csv(url1, delimiter=';')
df2 = pd.read_csv(url2, delimiter=',')
df3 = pd.read_csv(url3, delimiter=';')
df4 = pd.read_csv(url4, delimiter=",")

namen1 =  df1["naam"].drop_duplicates().sort_values().tolist()
namen2 =  df2["name"].drop_duplicates().sort_values().tolist()

namen3 = df3["name"].tolist()
namen4 = df4["name"].tolist()
url3 =  df3["profile_url"].tolist()
url4 =  df4["url"].tolist()
print (len(namen3))
print (len(namen4))

# Who defriended me
x=1
for n,u in zip(namen3,url3):
    if n  not in namen4:
        if u not in url4:
            print (f"{x}. {n} {u}")
            x +=1
