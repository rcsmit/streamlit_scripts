from youtube_transcript_api import YouTubeTranscriptApi
from urllib import parse
import streamlit as st

def transcribe_video(video_id, translate, language_from, translate_to, list_languages):
    if translate == True:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language_from])
        translated_transcript = transcript.translate(translate_to)
        transcript_fetched = (translated_transcript.fetch())
    else:
        transcript_fetched = YouTubeTranscriptApi.get_transcript(video_id,  languages=list_languages)

    total_text = ""
    for t in transcript_fetched:
        total_text += (t["text"])+ " "
    st.write(total_text)
    st.subheader("*To copy the text, roll over the mouse the box below and click the copy to clipboard icon at the right*")
    st.code (total_text)

    st.sidebar.markdown(
        '<br><br><a href="http://bark.phon.ioc.ee/punctuator" target="_blank">Go to Punctuator to punctuate the text</a>',
        unsafe_allow_html=True)


def do_punctuate(total_text):
    # VERY VERY SLOW !!!!!
    p = Punctuator('C:\\Users\\rcxsm\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\punctuator\\Demo-Europarl-EN.pcl')
    punctuated_text = p.punctuate(total_text)
    print (punctuated_text)
    pc.copy(punctuated_text)
    print ("Text copied")

def get_video_id(url):
    if url != "":
        st.info (f"Transcribing {url }")
        url_parsed = parse.urlparse(url)
        video_id_ = parse.parse_qs(url_parsed.query)
        video_id__ = video_id_['v']
        for vx in video_id__:
            video_id = vx
        return video_id
    else:
        st.warning("Enter an URL to transcribe")

def main():
    st.sidebar.title ("Video transcriber")
    languages = ["nl", "en", "de", "fr", "it", "sv", "da", "af", "sq", "am", "ar", "hy", "az", "eu", "be", "bn", "my",
                "bs", "bg", "ca", "ceb", "zh-Hant", "zh-Hans",
                "co", "eo", "et", "fil", "fi", "fr", "fy", "gl", "ka", "el", "gu", "ht", "ha", "haw", "iw",
                "hi", "hmn", "hu", "ga", "ig", "is", "id", "ja", "jv", "yi", "kn", "kk", "km", "rw", "ky", "ku", "ko",
                "hr", "lo", "la", "lv", "lt", "lb", "mk", "mg", "ml", "ms", "mt", "mi", "mr", "mn", "ne", "no", "ny",
                "or", "ug", "uk", "uz", "ps", "fa", "pl", "pt", "pa", "ro", "ru", "sm", "gd", "sr", "sn", "sd", "si", "sl",
                "sk", "su", "so", "es", "sw", "tg", "ta", "tt", "te", "th", "cs", "tk", "tr", "ur", "vi", "cy", "xh", "yo", "zu", "st"]

    languages_full = ["Afrikaans", "Albanees", "Amhaars", "Arabisch", "Armeens", "Azerbeidzjaans", "Baskisch", "Belarussisch",
                "Bengaals", "Birmaans", "Bosnisch", "Bulgaars", "Catalaans", "Cebuano", "Chinees (Traditioneel)", "Chinees (Vereenvoudigd)",
                "Corsicaans", "Deens", "Duits", "Engels", "Esperanto", "Estisch", "Filipijns", "Fins", "Frans", "Fries", "Galicisch",
                "Georgisch", "Grieks", "Gujarati", "HaÃ¯tiaans Creools", "Hausa", "HawaÃ¯aans", "Hebreeuws", "Hindi", "Hmong",
                "Hongaars", "Iers", "Igbo", "IJslands", "Indonesisch", "Italiaans", "Japans", "Javaans", "Jiddisch", "Kannada",
                "Kazachs", "Khmer", "Kinyarwanda", "Kirgizisch", "Koerdisch", "Koreaans", "Kroatisch", "Laotiaans", "Latijn",
                "Lets", "Litouws", "Luxemburgs", "Macedonisch", "Malagassisch", "Malayalam", "Maleis", "Maltees", "Maori",
                "Marathi", "Mongools", "Nederlands", "Nepalees", "Noors", "Nyanja", "Odia", "Oeigoers", "OekraÃ¯ens",
                "Oezbeeks", "Pasjtoe", "Perzisch", "Pools", "Portugees", "Punjabi", "Roemeens", "Russisch", "Samoaans",
                "Schots-Gaelisch", "Servisch", "Shona", "Sindhi", "Singalees", "Sloveens", "Slowaaks", "Soendanees",
                "Somalisch", "Spaans", "Swahili", "Tadzjieks", "Tamil", "Tataars", "Telugu", "Thai", "Tsjechisch",
                "Turkmeens", "Turks", "Urdu", "Vietnamees", "Welsh", "Xhosa", "Yoruba", "Zoeloe", "Zuid-Sotho", "Zweeds"]
    url = st.sidebar.text_input("Video to transcribe")
    translate = st.sidebar.selectbox("Translate", [True, False], index=1)
    if translate:
        language_from =  st.sidebar.selectbox("Translate from", languages, index=1)
        translate_to =  st.sidebar.selectbox("Translate to", languages, index=0)
    else:
        language_from, translate_to = None, None

    video_id = get_video_id(url)

    if video_id != None:
        transcribe_video(video_id, translate, language_from, translate_to, languages)

if __name__ == "__main__":
    main()


