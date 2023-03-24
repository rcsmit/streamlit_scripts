import openai

from keys import * # secret file with the prices
import random

# Set up OpenAI API credentials
openai.api_key = OPENAI_API_KEY
def read_tarot_cards():
    cards = ["The Fool",
        "The Magician",
        "The High Priestess",
        "The Empress",
        "The Emperor",
        "The Hierophant",
        "The Lovers",
        "The Chariot",
        "Strength",
        "The Hermit",
        "Wheel of Fortune",
        "Justice",
        "The Hanged Man",
        "Death",
        "Temperance",
        "The Devil",
        "The Tower",
        "The Star",
        "The Moon",
        "The Sun",
        "Judgment",
        "The World",
        "Ace of Wands",
        "Two of Wands",
        "Three of Wands",
        "Four of Wands",
        "Five of Wands",
        "Six of Wands",
        "Seven of Wands",
        "Eight of Wands",
        "Nine of Wands",
        "Ten of Wands",
        "Page of Wands",
        "Knight of Wands",
        "Queen of Wands",
        "King of Wands",
        "Ace of Cups",
        "Two of Cups",
        "Three of Cups",
        "Four of Cups",
        "Five of Cups",
        "Six of Cups",
        "Seven of Cups",
        "Eight of Cups",
        "Nine of Cups",
        "Ten of Cups",
        "Page of Cups",
        "Knight of Cups",
        "Queen of Cups",
        "King of Cups",
        "Ace of Swords",
        "Two of Swords",
        "Three of Swords",
        "Four of Swords",
        "Five of Swords",
        "Six of Swords",
        "Seven of Swords",
        "Eight of Swords",
        "Nine of Swords",
        "Ten of Swords",
        "Page of Swords",
        "Knight of Swords",
        "Queen of Swords",
        "King of Swords",
        "Ace of Pentacles",
        "Two of Pentacles",
        "Three of Pentacles",
        "Four of Pentacles",
        "Five of Pentacles",
        "Six of Pentacles",
        "Seven of Pentacles",
        "Eight of Pentacles",
        "Nine of Pentacles",
        "Ten of Pentacles",
        "Page of Pentacles",
        "Knight of Pentacles",
        "Queen of Pentacles",
        "King of Pentacles"]

    number_of_cards = 3
    # Initialize an empty prompt string
    prompt = "I have a Tarot reading. My cards are: "

    # Select random cards and add them to the prompt string
    for i in range(number_of_cards):
        # Generate a random number between 1 and the length of the cards list
        random_number = random.randint(0, len(cards)-1)
        
        # Select a card based on the random number generated
        card = cards[random_number]
        
        # Append the card to the prompt string
        if i == number_of_cards - 1:
            prompt += " and " + card
        elif i == 0:
            prompt += " " + card
        else:
            prompt += ", " + card
    prompt += ". Please provide a very short reading."
    # Print the prompt string
    print(prompt)


    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ])
    
    # print (completion)
    if completion.choices[0].message.content!=None:
        print (completion.choices[0].message.content)
    

st.header ("Read Tarot Cards")
if st.button("Read"):
     read_tarot_cards()