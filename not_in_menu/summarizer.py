from keys import * # secret file with the key
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from urllib import parse
import streamlit as st

openai.api_key = OPENAI_API_KEY

def transcribe_really(video_url):
    """Transcribe a video

    Args:
        video_url (str): url of the video

    Returns:
        str: the transcription of the video
    """    
    video_id = get_video_id(video_url)
    
    list_languages = [ "en","nl", "de", "fr", "it", "sv", "da", "af", "sq", "am", "ar", "hy", "az", "eu", "be", "bn", "my",
                "bs", "bg", "ca", "ceb", "zh-Hant", "zh-Hans",
                "co", "eo", "et", "fil", "fi", "fr", "fy", "gl", "ka", "el", "gu", "ht", "ha", "haw", "iw",
                "hi", "hmn", "hu", "ga", "ig", "is", "id", "ja", "jv", "yi", "kn", "kk", "km", "rw", "ky", "ku", "ko",
                "hr", "lo", "la", "lv", "lt", "lb", "mk", "mg", "ml", "ms", "mt", "mi", "mr", "mn", "ne", "no", "ny",
                "or", "ug", "uk", "uz", "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sm", "gd", "sr", "sn", "sd", "si", "sl",
                "sk", "su", "so", "es", "sw", "tg", "ta", "tt", "te", "th", "cs", "tk", "tr", "ur", "vi", "cy", "xh", "yo", "zu", "st"]
    try:
        transcript_fetched = YouTubeTranscriptApi.get_transcript(video_id ,  languages=list_languages)
    except Exception as e:
        st.warning("ERROR with transcribing the video")
        st.warning (e)
        st.stop()

    total_text = ""
    for t in transcript_fetched:
        total_text += (t["text"])+ " "

    return total_text

def get_video_id(url):
    """get the string after "v=" in the URL

    Args:
        url (str): the url

    Returns:
        str: the video ID
    """    
    try:
        url_parsed = parse.urlparse(url)
        video_id_ = parse.parse_qs(url_parsed.query)
        video_id__ = video_id_['v']
        for vx in video_id__:
            video_id = vx
        return video_id
    except:
        st.warning("Enter an (valid) URL to transcribe")
        st.stop()

def splitter(input_text, cut_off_limit,max_tokens,average_token_length, split_string):
    """Split the text

    Args:
    	input_text (str): the input text
        cut_off_limit (int): Split the text after ... tokens
        max_tokens (int): max number of tokens to generate 
        average_token_length (int): avg chars for each token
       split_string (str) : where to split  in parts
    Returns:
        list : list with the various  parts
    """   
    
    parts = []
    bufferline=""
    for i,line in enumerate(input_text.split(split_string)):
        # print (f"{i} - {line}")
        if len( bufferline+line)<((cut_off_limit*average_token_length)-max_tokens*average_token_length):
            bufferline = bufferline+ " "+ line
        else:
            parts.append(bufferline)
            bufferline=line
    parts.append(bufferline)
    return parts

def action(input_text, cut_off_limit,max_tokens,average_token_length,model, temperature, split_string,commando):
    """Split and summariaze the text
    Args:
        input_text (str): the input text
        cut_off_limit (int): Split the text after ... tokens
        max_tokens (int): max number of tokens to generate 
        average_token_length (int): avg chars for each token
        model (int): Which AI Model
        temperature (float): temperature. Lower = less random, higher = more random
        split_string (string) : where to split in parts
        commando:
    """  
    splitted_text_ =""
    total_summary = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if model == "text-davinci-003":
        price = 0.0200 / 1000
    else:
        price = 0.00200 / 1000 

    splitted_text = splitter(input_text, cut_off_limit,max_tokens,average_token_length, split_string)
    
    # with st.expander("Transcription / splitted text"):
    #     for i,s in enumerate(splitted_text):
    #         splitted_text_ +=  f"\n\n-------------------------------------- {i} ({len(s)} chars)---------------------------------\n\n" + s
    #     st.info(splitted_text_) 
    with st.expander("Splitted summary"):    
        for i,s in enumerate(splitted_text):
            summary, completion_tokens,prompt_tokens = summarize(s, max_tokens,model, temperature,commando)  
            st.info(f"\n\n------------------------------------summary -- {i} ({len(summary)} chars) {prompt_tokens=} {completion_tokens=}---------------------------------\n\n" +summary)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens +=completion_tokens
            total_summary += summary +"\n\n"

    st.info(total_summary)
    with st.expander("Extra info"):
        st.write(f"CHARS ORIGINAL TEXT : {len(input_text)}")
        st.write(f"NUMBER OF PARTS : {len(splitted_text)}")
        st.write(f"total_prompt_tokens : {round(total_prompt_tokens/1000,3)} K")
        st.write(f"total_completion_tokens : {round(total_completion_tokens)} ")
        st.write(f"Total price: $ {round(total_completion_tokens* price,4)} ")
def summarize(prompt, max_tokens,model, temperature,commando):
    """Summarize with OPEN AI

    Args:
        prompt (str): _description_
        max_tokens (int): max number of tokens to generate 
        model (int): Which AI Model
        temperature (float): temperature. Lower = less random, higher = more random

    Returns:
        str: the generated summary
    """    
    augmented_prompt = f"{commando}\n{prompt}"
    summary = openai.Completion.create(
            model=model,
            prompt=augmented_prompt,
            temperature=temperature,
            max_tokens=int(max_tokens),
            )["choices"][0]["text"]
    output = openai.Completion.create(
            model=model,
            prompt=augmented_prompt,
            temperature=temperature,
            max_tokens=int(max_tokens),
            )
    completion_tokens=output["usage"]["completion_tokens"]
    prompt_tokens =output["usage"]["prompt_tokens"]
    return summary, completion_tokens,prompt_tokens

def main():    
        model = st.sidebar.selectbox("Model", ["text-curie-001", "text-davinci-003"], index = 0)
        if model == "text-davinci-003":
            max_tokens = 4000
            cut_off_default = 3000
        else:
            max_tokens = 2048
            cut_off_default = 1500
        temperature=st.sidebar.slider("Temperature", min_value = 0.0, max_value =1.0,value = 0.5,step= 0.1, help="Lower = less random, higher = more random")
        cut_off_limit =  st.sidebar.number_input("Splitsen na .. tokens",0,4000, cut_off_default)
        max_tokens_answer = st.sidebar.number_input("Max tokens answer", 0,4000,max_tokens-cut_off_default)
        
        if max_tokens_answer + cut_off_limit > max_tokens:
            st.error(f"Total (max tokens answer + cut_of_limit) has to be lower than {max_tokens}")
            st.stop()

        average_token_length = 4
        menu_choice = st.sidebar.radio("",["YOUTUBE URL", "TEXT"], index=0)
        st.title("Text Summarizer")
        
        if menu_choice=="YOUTUBE URL":
            input_url = st.sidebar.text_input(label="Enter URL:", value="")
            input_text = transcribe_really(input_url)
            split_string = " "
        elif menu_choice=="TEXT":
            split_string = st.sidebar.text_input(label="Split string:", value=" ")
            input_text = st.text_area(label="Enter full text:", value="", height=250)
        commando = st.sidebar.text_input(label="Commando:", value="Summarize:")
    
        action(input_text, cut_off_limit,max_tokens_answer,average_token_length,model, temperature, split_string,commando)
     
if __name__ == "__main__":
    main()



# Number of words 13619 / number of characters 71583 / number of tokens 17895 (max. 4097) = 


# text-davinci, 11 requests -> 6,781 prompt + 1,181 completion = 7,962 tokens $0,16

# in playground 14425 tokens = 5 characters per token
# Curie : $0.0020  / 1K tokens
# Davinci Most powerful $0.0200  / 1K tokens

# engine	    OpenAi offers 4 engines: text-davinci-002 is the most capable GPT-3 model.
#               (See https://beta.openai.com/docs/engines/gpt-3 for details)  text-ada-001 is the fastest and cheapest
# prompt        A text string with the question.  For summarization tasks, the document text is embedded in this string. 
#               Note that this string is limited in length.
# temperature	Used to tell the model how much risk to task in computing the answer.  Higher values mean the model will take more risks.
# max_tokens	The maximum number of tokens to include in the answer.  16 is the default.
# top_p     	“Nucleus sampling” – Limits the number of results to consider. 
#               1.0 here means that 100% of the results are considered, so the temperature 
#               setting is the only one that limits the return values.
# frequency_penalty	Limits the likelihood the summary response will repeat itself.  0.0 is the default.
# presence_penalty	Controls the number of new tokens that will appear in the response.  0.0 is the default.