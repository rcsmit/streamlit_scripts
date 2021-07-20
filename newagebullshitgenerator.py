#  * New Age Bullshit Generator
#  * © 2014-15 Seb Pearce (sebpearce.com)
#  * Licensed under the MIT License.
#  * Converted into Python by Rene Smit, Nov. 2020

import random
import re
import streamlit as st

def cleanSentence(sentence):
    # replace 'a [vowel]' with 'an [vowel]' - works
    # Seb Pearce added a \W before the [Aa] because one time he got
    # 'Dogman is the antithesis of knowledge' :)
    sentence =  re.sub(r'(^|\W)([Aa]) ([aeiou])', r'\1\2n \3', sentence)

    # remove spaces before commas/periods/semicolons - works
    sentence =  re.sub(r'\s([?.!"](?:\s|$))', r'\1', sentence)

    # take care of prefixes (delete the space after the hyphen) - works
    sentence =  re.sub(r'- ', r'-', sentence)

    # add space after question marks if they're mid-sentence - works
    sentence =  re.sub(r'\?(\w)', r'? \1', sentence)

    # insert a space between sentences (after periods and question marks) works
    sentence =  re.sub(r'(?<=[.,])(?=[^\s])', r' ', sentence)

    # insertSpaceBeforePunctuation { WHY THIS ONE ????
    # sentence =  re.sub(r'(?<=[^\s])(?=[.,])', r' ', sentence)

    # capitalize letters after .!? - https://stackoverflow.com/a/41662260/4173718
    punc_filter = re.compile('([.!?]\s*)')
    split_with_punctuation = punc_filter.split(sentence)
    sentence = ''.join([i.capitalize() for i in split_with_punctuation])

    return sentence


def willekeurig(category):
    """ Chooses random word from the dictionary from the category category"""
    temporary = myDict[category]
    z =  random.choice(temporary)
    #print (z)
    return z

def searchText():
    tsextx = random.choice(sentencePatternsCopy)
    sentencePatternsCopy.remove(tsextx)
    return (tsextx)


def generatebullshit():
    output = ""

    for x in range(0, 13):
            textx = searchText()
            #print (textx)
            for w in wordtypes:
                while w in textx:
                    # replace only one time and do it again, otherwise a code will be replaced by the
                    # same word
                    y = willekeurig(w)
                    textx = textx.replace(w,y,1)
            output = output + textx + " "
            #print (output)
            if (x+1) % 3 == 0:
                # New paragraph after three lines
                output = output + "\n\n"
                st.write (" ")

    st.write (cleanSentence(output))


def main_():
    global sentencePatternsCopy

    #making a copy of the patternlist, in case of a refresh-function
    sentencePatternsCopy = sentencePatterns.copy()

    generatebullshit()



#  * Vocabulary words to be used in sentence patterns (patterns.js)

myDict = {
    'nCosmos': ['cosmos', 'quantum soup', 'infinite', 'universe', 'galaxy', 'multiverse',
        'grid', 'quantum matrix', 'totality', 'quantum cycle', 'nexus', 'planet', 'solar system',
         'world', 'stratosphere', 'dreamscape', 'biosphere', 'dreamtime'],
    'nPerson': ['being', 'child', 'traveller', 'entity', 'lifeform', 'wanderer', 'visitor',
    'prophet', 'seeker', 'Indigo Child'],
        'nP_ersonPlural': ['beings', 'travellers', 'entities', 'lifeforms', 'dreamweavers',
        'adventurers', 'pilgrims', 'warriors', 'messengers', 'dreamers', 'storytellers', 'seekers',
        'spiritual brothers and sisters', 'mystics', 'starseeds'],
    'nMass': ['consciousness', 'nature', 'beauty', 'knowledge', 'truth', 'life', 'healing', 'potential',
        'freedom', 'purpose', 'coherence', 'choice', 'passion', 'understanding', 'balance', 'growth',
       'inspiration', 'conscious living', 'energy', 'health', 'spacetime', 'learning', 'being', 'wisdom',
        'stardust', 'sharing', 'science', 'curiosity', 'hope', 'wonder', 'faith', 'fulfillment', 'peace',
        'rebirth', 'self-actualization', 'presence', 'power', 'will', 'flow', 'potentiality', 'chi',
        'intuition', 'synchronicity', 'wellbeing', 'joy', 'love', 'karma', 'life-force', 'awareness',
        'guidance', 'transformation', 'grace', 'divinity', 'non-locality', 'inseparability',
        'interconnectedness', 'transcendence', 'empathy', 'insight', 'rejuvenation', 'ecstasy',
        'aspiration', 'complexity', 'serenity', 'intention', 'gratitude', 'starfire', 'manna'],
    'nM_assBad': ['turbulence', 'pain', 'suffering', 'stagnation', 'desire', 'bondage', 'greed',
        'selfishness', 'ego', 'dogma', 'illusion', 'delusion', 'yearning', 'discontinuity', 'materialism'],
    'nOurPlural': ['souls', 'lives', 'dreams', 'hopes', 'bodies', 'hearts', 'brains', 'third eyes',
        'essences', 'chakras', 'auras'], 'nPath': ['circuit', 'mission', 'journey', 'path', 'quest',
        'vision quest', 'story', 'myth'], 'nOf': ['quantum leap', 'evolution', 'spark', 'lightning bolt',
        'reintegration', 'vector', 'rebirth', 'revolution', 'wellspring', 'fount', 'source', 'fusion',
        'canopy', 'flow', 'network', 'current', 'transmission', 'oasis', 'quantum shift', 'paradigm shift',
        'metamorphosis', 'harmonizing', 'reimagining', 'rekindling', 'unifying', 'osmosis', 'vision', 'uprising',
        'explosion'],
    'in_g': ['flowering', 'unfolding', 'blossoming', 'awakening', 'deepening', 'refining', 'maturing',
        'evolving', 'summoning', 'unveiling', 'redefining', 'condensing', 'ennobling', 'invocation'],
'a_d_j': ['enlightened', 'zero-point', 'quantum', 'high-frequency', 'Vedic', 'non-dual', 'conscious',
    'sentient', 'sacred', 'infinite', 'primordial', 'ancient', 'powerful', 'spiritual', 'higher',
    'advanced', 'internal', 'sublime', 'technological', 'dynamic', 'life-affirming', 'sensual',
    'unrestricted', 'ever-present', 'endless', 'ethereal', 'astral', 'cosmic', 'spatial',
    'transformative', 'unified', 'non-local', 'mystical', 'divine', 'self-aware', 'magical',
    'amazing', 'interstellar', 'unlimited', 'authentic', 'angelic', 'karmic', 'psychic',
    'pranic', 'consciousness-expanding', 'perennial', 'heroic', 'archetypal', 'mythic',
    'intergalactic', 'holistic', 'joyous', 'eternal'], 'Ad_jB_ig': ['epic', 'unimaginable',
    'colossal', 'unfathomable', 'magnificent', 'enormous', 'jaw-dropping', 'ecstatic', 'powerful',
    'untold', 'astonishing', 'incredible', 'breathtaking', 'staggering'], 'adjWith': ['aglow with',
    'buzzing with', 'beaming with', 'full of', 'overflowing with', 'radiating', 'bursting with',
    'electrified with'], 'adjPrefix': ['ultra-', 'supra-', 'hyper-', 'pseudo-'],
'vtMass': ['inspire', 'integrate', 'ignite', 'discover', 'rediscover', 'foster', 'release', 'manifest',
     'harmonize', 'engender', 'bring forth', 'bring about', 'create', 'spark', 'reveal', 'generate', 'leverage'],
'vtPerson': ['enlighten', 'inspire', 'empower', 'unify', 'strengthen', 'recreate', 'fulfill', 'change',
    'develop', 'heal', 'awaken', 'synergize', 'ground', 'bless', 'beckon'],
    'viPerson': ['exist', 'believe', 'grow', 'live', 'dream', 'reflect', 'heal',
    'vibrate', 'self-actualize'], 'vtDestroy': ['destroy', 'eliminate', 'shatter',
    'disrupt', 'sabotage', 'exterminate', 'obliterate', 'eradicate', 'extinguish',
    'erase', 'confront'],
'nTheXOf': ['richness', 'truth', 'growth', 'nature', 'healing', 'knowledge', 'birth', 'deeper meaning'],
'ppPerson': ['awakened', 're-energized', 'recreated', 'reborn', 'guided', 'aligned'],
'ppThingPrep': ['enveloped in', 'transformed into', 'nurtured by', 'opened by', 'immersed in',
    'engulfed in', 'baptized in'], 'fixedAdvP': ['through non-local interactions', 'inherent in nature',
    'at the quantum level', 'at the speed of light', 'of unfathomable proportions', 'on a cosmic scale',
    'devoid of self', 'of the creative act'],
'fixedAdvPPlace': ['in this dimension', 'outside time', 'within the Godhead', 'at home in the cosmos'],
'fixedNP': ['expanding wave functions', 'superpositions of possibilities', 'electromagnetic forces',
    'electromagnetic resonance', 'molecular structures', 'atomic ionization', 'electrical impulses',
    'a resonance cascade', 'bio-electricity', 'ultrasonic energy', 'sonar energy', 'vibrations',
    'frequencies', 'four-dimensional superstructures', 'ultra-sentient particles', 'sub-atomic particles',
    'chaos-driven reactions', 'supercharged electrons', 'supercharged waveforms', 'pulses', 'transmissions',
    'morphogenetic fields', 'bio-feedback', 'meridians', 'morphic resonance', 'psionic wave oscillations'],
'nSubject': ['alternative medicine', 'astrology', 'tarot', 'crystal healing', 'the akashic record',
    'feng shui', 'acupuncture', 'homeopathy', 'aromatherapy', 'ayurvedic medicine', 'faith healing',
    'prayer', 'astral projection', 'Kabala', 'reiki', 'naturopathy', 'numerology', 'affirmations',
    'the Law of Attraction', 'tantra', 'breathwork'],
'vOpenUp': ['open up', 'give us access to', 'enable us to access', 'remove the barriers to',
    'clear a path toward', 'let us access', 'tap into', 'align us with', 'amplify our connection to',
    'become our stepping-stone to', 'be a gateway to'],
'vTraverse': ['traverse', 'walk', 'follow', 'engage with', 'go along', 'roam', 'navigate', 'wander', 'embark on'],
'nameOfGod': ['Gaia', 'Shiva', 'Parvati', 'the Goddess', 'Shakti'],
'nBenefits': ['improved focus', 'improved concentration', 'extreme performance',
    'enlightenment', 'cellular regeneration', 'an enhanced sexual drive', 'improved hydration',
    'psychic rejuvenation', 'a healthier relationship with the Self'],
'ad_jProduct': ['alkaline', 'quantum', 'holographic', 'zero-point energy', '“living”',
    'metaholistic', 'ultraviolet', 'ozonized', 'ion-charged', 'hexagonal-cell', 'organic'],
'nProduct': ['water', 'healing crystals', 'Tibetan singing bowls', 'malachite runes', 'meditation bracelets',
    'healing wands', 'rose quartz', 'karma bracelets', 'henna tattoos', 'hemp garments', 'hemp seeds',
    'tofu', 'massage oil', 'herbal incense', 'cheesecloth tunics']}


#  * A list of sentence patterns to be parsed by main.js

wordtypes = ["a_d_j","Ad_jB_ig","adjPrefix","adjWith","fixedAdvP","fixedAdvPPlace","fixedNP","in_g","nCosmos",
    "nMass","nM_assBad", "nameOfGod",
    "nOf","nOurPlural","nPath","nPerson","nP_ersonPlural","nSubject","nTheXOf","ppPerson","ppThingPrep","vOpenUp",
    "vTraverse","viPerson","vtDestroy","vtMass","vtPerson", "ad_jProduct", "nProduct"]

sentencePatterns = [
    "nMass is the driver of nMass.",
    "nMass is the nTheXOf of nMass, and of us.",
    "You and I are nP_ersonPlural of the nCosmos.",
    "We exist as fixedNP.",


    "We viPerson, we viPerson, we are reborn.",
    "Nothing is impossible.",
    "This life is nothing short of a in_g nOf of a_d_j nMass.",
    "Consciousness consists of fixedNP of quantum energy. “Quantum” means a in_g of the a_d_j.",
    "The goal of fixedNP is to plant the seeds of nMass rather than nM_assBad.",
    "nMass is a constant.",
    "By in_g, we viPerson.",
    "The nCosmos is adjWith fixedNP.",

    "To vTraverse the nPath is to become one with it.",
    "Today, science tells us that the essence of nature is nMass.",
    "nMass requires exploration.",

     "We can no longer afford to live with nM_assBad.",
    "Without nMass, one cannot viPerson.",
    "Only a nPerson of the nCosmos may vtMass this nOf of nMass.",
    "You must take a stand against nM_assBad.",
    "Yes, it is possible to vtDestroy the things that can vtDestroy us, but not without nMass on our side.",
    "nM_assBad is the antithesis of nMass.",
    "You may be ruled by nM_assBad without realizing it. Do not let it vtDestroy the nTheXOf of your nPath.",
    "The complexity of the present time seems to demand a in_g of our nOurPlural if we are going to survive.",
    "nM_assBad is born in the gap where nMass has been excluded.",
    "Where there is nM_assBad, nMass cannot thrive.",

    "Soon there will be a in_g of nMass the likes of which the nCosmos has never seen.",
    "It is time to take nMass to the next level.",
    "Imagine a in_g of what could be.",
    "Eons from now, we nP_ersonPlural will viPerson like never before as we are ppPerson by the nCosmos.",
    "It is a sign of things to come.",
    "The future will be a a_d_j in_g of nMass.",
    "This nPath never ends.",
    "We must learn how to lead a_d_j lives in the face of nM_assBad.",
    "We must vtPerson ourselves and vtPerson others.",
    "The nOf of nMass is now happening worldwide.",
    "We are being called to explore the nCosmos itself as an interface between nMass and nMass.",
    "It is in in_g that we are ppPerson.",
    "The nCosmos is approaching a tipping point.",
    "nameOfGod will vOpenUp a_d_j nMass.",

    "Although you may not realize it, you are a_d_j.",
    "nPerson, look within and vtPerson yourself.",
    "Have you found your nPath?",
    "How should you navigate this a_d_j nCosmos?",
    "It can be difficult to know where to begin.",
    "If you have never experienced this nOf fixedAdvP, it can be difficult to viPerson.",
    "The nCosmos is calling to you via fixedNP. Can you hear it?",

    "Throughout history, humans have been interacting with the nCosmos via fixedNP.",
    "Reality has always been adjWith nP_ersonPlural whose nOurPlural are ppThingPrep nMass.",
    "Our conversations with other nP_ersonPlural have led to a in_g of adjPrefix a_d_j consciousness.",
    "Humankind has nothing to lose.",
    "We are in the midst of a a_d_j in_g of nMass that will vOpenUp the nCosmos itself.",
    "Who are we? Where on the great nPath will we be ppPerson?",
    "We are at a crossroads of nMass and nM_assBad."

     'Through nSubject, our nOurPlural are ppThingPrep nMass.',
     'nSubject may be the solution to what’s holding you back from a Ad_jB_ig nOf of nMass.',
     'You will soon be ppPerson by a power deep within yourself — a power that is a_d_j, a_d_j.',
     'As you viPerson, you will enter into infinite nMass that transcends understanding.',
     'This is the vision behind our 100% ad_jProduct, ad_jProduct nProduct.',
     'With our ad_jProduct nProduct, nBenefits is only the beginning.'
]

main:()

    st.sidebar.title('New Age Bullshit generator')
    if st.sidebar.button('GENERATE'):
        main_()

    tekst = (
        '<hr>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\'>@rcsmit</a>) <br>'

        'Sourcecode : <a href=\"https://github.com/rcsmit/newagebullshitgenerator/edit/main/newagebullshitgenerator.py\">github.com/rcsmit</a><br><br>'
        '© 2014-15 Seb Pearce (sebpearce.com)<br>'
        'Licensed under the MIT License.')

    st.sidebar.markdown(tekst, unsafe_allow_html=True)

if __name__ == "__main__":

    main()


