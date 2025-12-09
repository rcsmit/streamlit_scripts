import random
import streamlit as st

def deal_into_columns(stack):
    cols = [[], [], []]  # A, B, C
    for i, card in enumerate(stack):
        cols[i % 3].append(card)
    return cols

def encode_card(card, chosen_card):
    if card == chosen_card:
        return "X"
    if card <= 9:
        return str(card)
    # 10 -> 'a', 11 -> 'b', ...
    return chr(ord('a') + (card - 10))

def print_grid(cols, chosen_card, title):
    st.code(title)
    st.code("   A  B  C")
    for row in range(7):
        line = []
        for col in range(3):
            card = cols[col][row]
            line.append(encode_card(card, chosen_card))
        st.code(f"{row+1}: " + "  ".join(line))
    st.code(" ")

def make_new_stack(cols, chosen_col_index):
    # gekozen kolom in het midden
    if chosen_col_index == 0:      # A
        order = [1, 0, 2]          # B A C
    elif chosen_col_index == 1:    # B
        order = [0, 1, 2]          # A B C
    else:                          # C
        order = [0, 2, 1]          # A C B
    new_stack = []
    for idx in order:
        new_stack.extend(cols[idx])
    return new_stack

def find_chosen_column(cols, chosen_card):
    for i, col in enumerate(cols):
        if chosen_card in col:
            return i
    raise ValueError("Card not found in any column")

def simulate_trick(chosen_position=None):
    stack = list(range(1, 22))

    if chosen_position is None:
        chosen_position = random.randint(1, 21)
    chosen_card = stack[chosen_position - 1]

    st.code(f"Gekozen kaart zit op positie {chosen_position} in de stapel.\n")
    columns=st.columns(4)
    # 3x kolom kiezen
    for round_nr in range(1, 4):
        with columns[round_nr-1]:
            cols = deal_into_columns(stack)
            print_grid(cols, chosen_card, f"Ronde {round_nr}")

            chosen_col_index = find_chosen_column(cols, chosen_card)
            col_name = ["A", "B", "C"][chosen_col_index]
            st.code(f"Vriend wijst kolom {col_name} aan.\n")

            stack = make_new_stack(cols, chosen_col_index)

    # Eindresultaat
    with columns[3]:
        cols = deal_into_columns(stack)
        print_grid(cols, chosen_card, "Eindpositie")
        final_pos = stack.index(chosen_card) + 1
        st.code(f"Eindpositie in de stapel: {final_pos} (middelste kaart van 21)")

import math

def p1(p):
    r = (p - 1) // 3 + 1       # rijnummer 1..7
    return 7 + r               # positie na 1e keer

def next_p(pn):
        return 8 + (pn - 1) // 3   # algemene stap: p_{n+1}

def simulatie_formula():
    st.code(f"{'p':>2} {'p1':>3} {'p2':>3} {'p3':>3}")
    st.code("-" * 14)

    for p in range(1, 22):
        p_1 = p1(p)
        p_2 = next_p(p_1)
        p_3 = next_p(p_2)
        st.code(f"{p:>2} {p_1:>3} {p_2:>3} {p_3:>3}") 


if __name__ == "__main__":

    simulatie_formula()
    # kies zelf een positie 1..21 of laat None voor random
    for i in range (1,22):
        
        simulate_trick(chosen_position=i)
        st.code("-----------------------------")

