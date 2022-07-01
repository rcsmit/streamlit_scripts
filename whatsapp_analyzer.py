# https://www.analyticsvidhya.com/blog/2021/04/whatsapp-group-chat-analyzer-using-python/
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import *
# import datetime as dt
from matplotlib.ticker import MaxNLocator
# import regex
# import emoji
from seaborn import *
#from heatmap import heatmap
# from wordcloud import WordCloud , STOPWORDS , ImageColorGenerator
# from nltk import *
# from plotly import express as px
import streamlit as st
#from floweaver import *
import plotly.graph_objects as go

from collections import Counter
from  itertools import chain

def startsWithDateAndTime(s):
    # had to change the pattern
    pattern = "^([0-9]+)(-)([0-9]+)(-)([0-9][0-9])([0-9][0-9]) ([0-9]+):([0-9][0-9]) -"
    result = re.match(pattern, s)
    if result:
        return True
    return False


def FindAuthor(s):
    ### Regex pattern to extract username of Author.

    pattern ='.*(?=:)'
    result = re.match(pattern, s)

    if result:
        return True
    return False




def getDataPoint(line):
    ### Extracting Date, Time, Author and message from the chat file.
    splitLine = line.split(' - ')
    dateTime = splitLine[0]

    date, time = dateTime.split(' ')
    message =  ' '.join(splitLine[1:])
    if FindAuthor(message):
        splitMessage = message.split(': ')
        author = splitMessage[0]
        message = ' '.join(splitMessage[1:])
    else:
        author = "x"
    return date, time, author, message

def get_data_from_url():
    parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe
    ### Uploading exported chat file

    #conversationPath = 'C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\WhatsApp-chat met Go ABCDE Joost.txt' # chat file
    #conversationPath = 'C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\WhatsApp-chat met Go ABCDE Joost - kopie.txt'
    #conversationPath = 'C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\WhatsApp-chat met NYE supercrew.txt'
    conversationPath = r"C:\Users\rcxsm\Documents\python_scripts\in\WhatsApp-chat met Go ABCDE Joost.txt"
    #conversationPath = 'C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\WhatsApp-chat met Acro Lovers Phangan.txt'
    with open(conversationPath, encoding="utf-8") as fp:

        ### Skipping first line of the file because contains information related to something about end-to-end encryption
        fp.readline()

        messageBuffer= []
        date, time, author= None, None, None
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            if startsWithDateAndTime(line):
                if len(messageBuffer) > 0:
                    parsedData.append([date, time, author, ' '.join(messageBuffer)])
                messageBuffer.clear()
                date, time, author, message = getDataPoint(line)
                messageBuffer.append(message)
            else:
                messageBuffer.append(line)
    df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message']) # Initialising a pandas Dataframe.
    df = df[df['Author'] != 'x'].reset_index()

    return df



def get_data_from_file():
    parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe
    ### Uploading exported chat file
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        messageBuffer= []
        date, time, author= None, None, None

        for line in uploaded_file:
            if not line:
                    break
            line = line.decode("utf-8", "ignore").strip()
            if startsWithDateAndTime(line):

                if len(messageBuffer) > 0:
                    parsedData.append([date, time, author, ' '.join(messageBuffer)])
                messageBuffer.clear()
                date, time, author, message = getDataPoint(line)
                messageBuffer.append(message)
            else:
                messageBuffer.append(line)
        df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message']) # Initialising a pandas Dataframe.
        df = df[df['Author'] != 'x'].reset_index()
        return df
    else:
        st.warning("You need to upload a csv or excel file. Files are not stored anywhere after the processing of this script")
        st.stop()
def manipulate_df(df):
    ### changing datatype of "Date" column.

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y") # %H:%M:%S")
    df["Month_Year"] =  pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
    df["Day_Month_Year"] =  pd.to_datetime(df['Date']).dt.strftime('%d-%m-%Y')

    ### Droping Nan values from dataset
    # df = df.dropna()
    # df = df.reset_index(drop=True)

    ### Checking no. of authors of group
    st.write (f"Number of authors {df['Author'].nunique()}")
    ### Checking authors of group
    st.write(f"Authors : {df['Author'].unique()}")
    ### Adding one more column of "Day" for better analysis, here we use datetime lirary which help us to do this task easily.
    weeks = {
    0 : 'Monday',
    1 : 'Tuesday',
    2 : 'Wednesday',
    3 : 'Thrusday',
    4 : 'Friday',
    5 : 'Saturday',
    6 : 'Sunday'
    }
    df['Day'] = df['Date'].dt.weekday.map(weeks)
    ### Rearranging the columns for better understanding
    df = df[['Date','Day','Time','Author','Message', "Month_Year", "Day_Month_Year"]]
    ### Changing the datatype of column "Day".
    #df['Day'] = df['Day'].astype('category')
    df.loc[:, 'Day'] = df.loc[:, 'Day'].astype('category')
    ### Looking newborn dataset.

    ### Counting number of letters in each message
    df['Letters'] = df['Message'].apply(lambda s : len(s))
    ### Counting number of word's in each message
    df['Words'] = df['Message'].apply(lambda s : len(s.split(' ')))
    ### Function to count number of links in dataset, it will add extra column and store information in it.
    URLPATTERN = r'(https?:\/\/)'
    df['Url_Count'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
    links = np.sum(df.Url_Count)
    ### Function to count number of media in chat.
    MEDIAPATTERN = r'<Media weggelaten>'
    df['Media_Count'] = df.Message.apply(lambda x : re.findall(MEDIAPATTERN, x)).str.len()
    media = np.sum(df.Media_Count)

    return df, links, media

def generate_teltabel(df, auteurs):


    teltabel=[] #teltabel 2 is voor een tabel zonder reacties op zichzelf
    for i in range(len(auteurs)):
        list_y=[]
        for j in range(len(auteurs)):
            list_y.append(0)
        teltabel.append(list_y)

    for i in range(1,len(df)):
        for x in range(len(auteurs)):

            if df.loc[i-1,'Author'] == auteurs[x]:
                df.loc[i,"is_reaction_to"] = auteurs[x]
                for y in range(len(auteurs)):
                    if df.loc[i,'Author'] ==  auteurs[y]:
                        teltabel[x][y] +=1
    return teltabel

def generate_flows(df):
    """Generate a new df with the number of reactions of X to Y
    """
    flows = (
        df.groupby(["is_reaction_to","Author" ])
        .agg({"Count": "sum"})
        .dropna()
        .reset_index()
        )
    flows = (
    flows.rename(
        columns={
            "is_reaction_to": "source",
            "Author": "target",

                    }
                )
            )
    st.write(flows)
    return flows
def plot_sankey(values1,values2, auteurs):
    label = [*auteurs, *auteurs]

    source, source2, target, target2 =[],[],[],[]
    for i in range(len(auteurs)):
        for j in range(len(auteurs)):
            source.append(i)
            target.append(j+len(auteurs))

            if j!=(len(auteurs)-1):
                source2.append(i)
            if i!=j:
                target2.append(j+len(auteurs))

    fig1 = go.Figure(data=[go.Sankey(
        node = dict(
        label = label,
        color = ['#a6cee3', '#cea6e3', '#cee3a6', '#fdbf6f','#bffd6f', '#bf6ffd', '#fb9a99', '#fb999a','#9a99fb']

        ),
        link = dict(
        source = source,
        target = target,
        value = values1,
        color = ['#a6cee3', '#cea6e3', '#cee3a6', '#fdbf6f','#bffd6f', '#bf6ffd', '#fb9a99', '#fb999a','#9a99fb']


        # generating from the generated df doesn't work ---------
        # source= flows["source"],
        # target = flows["target"],
        # value = flows["Count"],
    ))])

    fig1.update_layout(title_text="Who writes who?", font_size=10)
    st.plotly_chart(fig1)

    fig2 = go.Figure(data=[go.Sankey(
        node = dict(
        label = label,
        color = ['#a6cee3', '#cea6e3', '#cee3a6', '#fdbf6f','#bffd6f', '#bf6ffd', '#fb9a99', '#fb999a','#9a99fb']
        ),
        link = dict(
        source = source2,
        target = target2,
        value = values2,
        color = ['#a6cee3', '#cea6e3', '#cee3a6', '#fdbf6f','#bffd6f', '#bf6ffd', '#fb9a99', '#fb999a','#9a99fb']


        # source= flows["source"],
        # target = flows["target"],
        # value = flows["Count"],
    ))])

    fig2.update_layout(title_text="Who writes who? (excl. message to themself)", font_size=10)
    st.plotly_chart(fig2)
def who_reacts_who(df):
    #df = df[df['Author'] == None]
    df["is_reaction_to"] =None


    #auteurs= ["René", "Renee", "Joost"]
    auteurs  = df.Author.unique()
    teltabel = generate_teltabel(df, auteurs)

    # 1 is with replies to themselves, 2 is without
    values1 = [item for sublist in teltabel for item in sublist]
    values2=[]
    for x in range(len(auteurs)):
        for y in range(len(auteurs)):
                if auteurs[x] !=  auteurs[y]:
                    values2.append(teltabel[x][y])


    df["Count"] = 1
    df["Type"] = "x"

    st.write("WIE REAGEERT OP WIE")
    #st.write(teltabel)

    flows = generate_flows(df)
    plot_sankey(values1, values2, auteurs)


def main():
    df = get_data_from_file()
    #df = get_data_from_url()
    df, links, media = manipulate_df(df)
    st.subheader("Show Totals")
    show_totals(df)
    st.subheader("Show messages author")

    show_messages_author(df)
    st.subheader("Most commom words")

    most_common_words(df)
    st.subheader("Who reacts who")
    who_reacts_who(df)
    #wordcloud(df)
    st.subheader("Rest")
    rest(df)

    st.write("Based on the work of https://www.analyticsvidhya.com/blog/2021/04/whatsapp-group-chat-analyzer-using-python/")


def most_common_words(df):
    
    # https://towardsdatascience.com/build-your-own-whatsapp-chat-analyzer-9590acca9014
    null_authors_df = df[df['Author'].isnull()]
    media_messages_df = df[df['Message'] == '<Media weggelaten>']
    messages_df = df.drop(null_authors_df.index) # Drops all rows of the data frame containing messages from null authors
    messages_df = messages_df.drop(media_messages_df.index) # Drops all rows of the data frame containing media messages
   
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    count_words_by_author(messages_df, None)
    authors = messages_df["Author"].drop_duplicates().to_list()
    for author in authors:
        count_words_by_author(messages_df, author)
    
    #count the letters
   


    df_letter_count = pd.Series(Counter(chain(*messages_df.Message))).sort_values( ascending=False)
    st.write("Lettercount")
    st.write (df_letter_count)
    discrete_columns = [['Date', 'Time', 'Author', 'Message']]
    #messages_df[discrete_columns].describe()

    continuous_columns = [['Letter_Count', 'Word_Count']]
    #messages_df[continuous_columns].describe()
    st.write(f"Number of letters: {messages_df['Letter_Count'].sum()} - Number of words {messages_df['Word_Count'].sum()}")

    fig = plt.figure()

    word_count_value_counts = messages_df['Word_Count'].value_counts()
    top_40_word_count_value_counts = word_count_value_counts.head(40)
    top_40_word_count_value_counts.plot.bar()
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('most common number of words in a message')
    st.pyplot(fig)

    # doesnt work
    # st.subheader("Number of letters by author")
    # total_letter_count_grouped_by_author = messages_df[['Author', 'Letter_Count']].groupby('Author').sum()
    # sorted_total_letter_count_grouped_by_author = total_letter_count_grouped_by_author.sort_values('Letter_Count', ascending=False)
    # top_10_sorted_total_letter_count_grouped_by_author = sorted_total_letter_count_grouped_by_author.head(10)
    # st.write(top_10_sorted_total_letter_count_grouped_by_author)
    # fig2 = plt.figure()
    
    # top_10_sorted_total_letter_count_grouped_by_author.plot.bar()
    # plt.xlabel('Number of Letters')
    # plt.ylabel('Authors')
    # plt.title ("Number of letters by author")
    # st.pyplot(fig2)

def count_words_by_author(messages_df, author):
    if author != None:
        messages_df = messages_df[messages_df["Author"] == author]
    else:
        author = "All"
    df_words = pd.DataFrame()
   
    for i in range(len(messages_df)):
        wordlist = []
        a = messages_df.iloc[i,4]
        b = a.split(' ')
        for c in b:
            wordlist.append(c)
        df_ =  pd.DataFrame([ {
                                    "words": wordlist
                                    }]
                            )           

        df_words = pd.concat([df_words, df_],axis = 0) 
    
    df_word_count = pd.Series(Counter(chain(*df_words.words))).sort_values( ascending=False)

    st.write(f"word count by {author}")
    st.write (df_word_count)

def show_totals(df):
    total_messages = df.shape[0]
    media_messages = df[df['Message'] == '<Media weggelaten>'].shape[0]
    links = np.sum(df.Url_Count)
    st.write('Group Chatting Stats : ')
    st.write('Total Number of Messages : {}'.format(total_messages))
    st.write('Total Number of Media Messages : {}'.format(media_messages))
    st.write('Total Number of Links : {}'.format(links))

def show_messages_author(df):
    l = df.Author.unique()

    for i in range(len(l)):
        req_df = df[df["Author"] == l[i]]  ## Filtering out messages of particular user
        if l[i] :
            ### req_df will contain messages of only one particular user
            st.write(f'--> Stats of {l[i]} <-- ')
            ### shape will st.write number of rows which indirectly means the number of messages
            st.write('Total Message Sent : ', req_df.shape[0])
            ### Word_Count contains of total words in one message. Sum of all words/ Total Messages will yield words per message
            words_per_message = (np.sum(req_df['Words']))/req_df.shape[0]
            total_words = np.sum(req_df['Words'])
            w_p_m = ("%.1f" % round(words_per_message, 1))
            st.write(f'Average Words per Message : {w_p_m}')
            st.write(f'Total Words  : {total_words}')
            ### media conists of media messages
            media = sum(req_df["Media_Count"])
            st.write('Total Media Message Sent : ', media)
            ### links consist of total links
            links = sum(req_df["Url_Count"])
            st.write('Total Links Sent : ', links)
            st.write()
            st.write('----------------------------------------------------------')

def wordcloud(df):

    # WORDCLOUD GIVES AN ERROR, LOOKS LIKE SCRIPT ISNT PLAYED

    ### Word Cloud of mostly used word in our Group
    text = " ".join(review for review in df.Message)
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)
    ### Display the generated image:
    fig = plt.figure( figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)


def rest_not_in_use(df):

    ### Creates a list of unique Authors
    l = df.Author.unique()
    for i in range(len(l)):
        ### Filtering out messages of particular user
        req_df = df[df["Author"] == l[i]]
        ### req_df will contain messages of only one particular user
        st.write(l[i],'  ->  ',req_df.shape[0])

    l = df.Day.unique()
    for i in range(len(l)):
        ### Filtering out messages of particular user
        req_df = df[df["Day"] == l[i]]
        ### req_df will contain messages of only one particular user
        st.write(l[i],'  ->  ',req_df.shape[0])

def barplot(what, x_label,y_label, title, y_to_integer):
    fig = plt.figure()
    what.plot.bar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_to_integer:
        what.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    st.pyplot(fig)

def rest(df):
    mostly_active_author(df)

    ### Mostly Active day in the Group
    active_day = df['Day'].sort_index(ascending=False).value_counts()
    a_d = active_day.head(10)
    barplot(a_d, "Day", "No. of messages", 'Mostly active day of Week in the Group',False)



    ## Top-10 Media Contributor of Group
    mm = df[df['Message'] == '<Media weggelaten>']
    mm1 = mm['Author'].value_counts()
    top10 = mm1.head(10)
    barplot(top10, 'Author', "No. of media", 'Top-10 media contributor of Group',False)


    max_words = df[['Author','Words']].groupby('Author').sum() #.reset_index()
    m_w = max_words.sort_values('Words',ascending=False).head(10)
    st.write('Analysis of members who has used max. no. of words in his/her messages')
    st.write (m_w)
    # barplot(m_w, 'Author', "No. of words", 'Analysis of members who has used max. no. of words in his/her messages',False)


    ### Member who has shared max numbers of link in Group
    # fig = plt.figure()
    max_links = df[['Author','Url_Count']].groupby('Author').sum()
    m_l = max_links.sort_values('Url_Count',ascending=False).head(10)
    st.write('Analysis of members who has shared max no. of links in Group')
    st.write(m_l)
    # barplot(m_l, 'Author', "No. of links", 'Analysis of members who has shared max no. of links in Group',False)



    ### Time whenever our group is highly active
    fig = plt.figure()
    t = df['Time'].sort_index(ascending=False).value_counts().head(20)
    tx = t.plot.bar()
    tx.yaxis.set_major_locator(MaxNLocator(integer=True))  #Converting y axis data to integer
    plt.xlabel('Time')
    plt.ylabel('No. of messages')
    plt.title('Analysis of time when Group was highly active.')
    st.pyplot(fig)

    lst = []
    for i in df['Time'] :
        out_time = datetime.strftime(datetime.strptime(i,"%H:%M"),"%H:%M")
        lst.append(out_time)
    df['24H_Time'] = lst
    df['Hours'] = df['24H_Time'].apply(lambda x : x.split(':')[0])

    ### Most suitable hour of day, whenever there will more chances of getting responce from group members.
    fig = plt.figure()
    std_time = df['Hours'].sort_index(ascending=False).value_counts().head(15)
    s_T = std_time.plot.bar()
    s_T.yaxis.set_major_locator(MaxNLocator(integer=True))  #Converting y axis data to integer
    plt.xlabel('Hours (24-Hour)')
    plt.ylabel('No. of messages')
    plt.title('Most suitable hour of day.')
    st.pyplot(fig)

    # active_m = mostly_active
    # for i in range(len(active_m)) :
    #     # Filtering out messages of particular user
    #     m_chat = df[df["Author"] == active_m[i]]
    #     st.write(f'--- Author :  {active_m[i]} --- ')
    #     # Word Cloud of mostly used word in our Group
    #     msg = ' '.join(x for x in m_chat.Message)
    #     wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(msg)
    #     fig = plt.figure(figsize=(10,5))
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis("off")
    #     st.pyplot(fig)
    #     st.write('____________________________________________________________________________________n')

    ### Date on which our Group was highly active.
    fig = plt.figure()
    df['Day_Month_Year'].value_counts().head(15).plot.bar()

    plt.xlabel('Date')
    plt.ylabel('No. of messages')
    plt.title('Analysis of Date on which Group was highly active')
    st.pyplot(fig)

    # z = df['Date'].value_counts()
    # z1 = z.to_dict() #converts todictionary
    # df['Msg_count'] = df['Date'].map(z1)
    # ### Timeseries plot
    # fig = px.line(x=df['Date'],y=df['Msg_count'])
    # fig.update_layout(title='Analysis of number of messages using TimeSeries plot.',
    #                 xaxis_title='Month',
    #                 yaxis_title='No. of Messages')
    # fig.update_xaxes(nticks=20)
    # fig.show()

    df['Year'] = df['Date'].dt.year
    df['Mon'] = df['Date'].dt.month
    months = {
        1 : 'Jan',
        2 : 'Feb',
        3 : 'Mar',
        4 : 'Apr',
        5 : 'May',
        6 : 'Jun',
        7 : 'Jul',
        8 : 'Aug',
        9 : 'Sep',
        10 : 'Oct',
        11 : 'Nov',
        12 : 'Dec'
    }
    df['Month'] = df['Mon'].map(months)
    df.drop('Mon',axis=1,inplace=True)

    ### Mostly Active month
    fig = plt.figure()
    active_month = df['Month_Year'].value_counts()
    a_m = active_month
    a_m.plot.bar()
    plt.xlabel('Month')
    plt.ylabel('No. of messages')
    plt.title('Analysis of mostly active month.',fontdict={'fontsize': 20,
            'fontweight': 8})
    st.pyplot(fig)

    # z = df['Month_Year'].value_counts()
    # z1 = z.to_dict() #converts to dictinary
    # df['Msg_count_monthly'] = df['Month_Year'].map(z1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.bar(df['Month_Year'], df['Msg_count_monthly'])
    # ax.set_xticks(df['Month_Year'])
    # ax.set_xticklabels(df['Month_Year'], rotation=45)
    # #sns.lineplot(data=df,x='Month_Year',y='Msg_count_monthly',markers=True,marker='o')
    # plt.xlabel('Month')
    # plt.ylabel('No. of messages')
    # plt.title('Analysis of mostly active month using line plot.')
    # st.pyplot(fig)

    ### Total message per year
    ### As we analyse that the group was created in mid 2019, thats why number of messages in 2019 is less.
    fig = plt.figure()
    active_month = df['Year'].value_counts()
    a_m = active_month
    a_m.plot.bar()
    plt.xlabel('Year')
    plt.ylabel('No. of messages')
    plt.title('Analysis of mostly active year.')
    st.pyplot(fig)

def mostly_active_author(df):
    mostly_active = df['Author'].value_counts()
    m_a = mostly_active.head(10)
    barplot(m_a, "Authors", "No. of messages", "Mostly active member of Group", False)


    # df2 = df.groupby(['Hours', 'Day'], as_index=False)["Message"].count()
    # df2 = df2.dropna()
    # df2.reset_index(drop = True,inplace = True)
    # ### Analysing on which time group is mostly active based on hours and day.
    # analysis_2_df = df.groupby(['Hours', 'Day'], as_index=False)["Message"].count()
    # ### Droping null values
    # analysis_2_df.dropna(inplace=True)
    # analysis_2_df.sort_values(by=['Message'],ascending=False)
    # day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thrusday', 'Friday', 'Saturday', 'Sunday']
    # fig = plt.figure(figsize=(15,8))
    # heatmap(
    #     x=analysis_2_df['Hours'],
    #     y=analysis_2_df['Day'],
    #     size_scale = 500,
    #     size = analysis_2_df['Message'],
    #     y_order = day_of_week[::-1],
    #     color = analysis_2_df['Message'],
    #     palette = sns.cubehelix_palette(18)
    # )
    # st.pyplot(fig)


main()