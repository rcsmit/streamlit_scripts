import streamlit as st

def main_chatgpt():
    NOTE_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    NOTE_TO_INDEX = {n: i for i, n in enumerate(NOTE_ORDER)}

    # Standard tuning pitch classes for strings E A D G B e
    OPEN_PITCH = {
        'E': NOTE_TO_INDEX['E'],  # low E
        'A': NOTE_TO_INDEX['A'],
        'D': NOTE_TO_INDEX['D'],
        'G': NOTE_TO_INDEX['G'],
        'B': NOTE_TO_INDEX['B'],
        'e': NOTE_TO_INDEX['E'],  # high e
    }

    def fret_to_note(open_pc: int, fret: int) -> str:
        return NOTE_ORDER[(open_pc + fret) % 12]

    def convert_tab_block(lines):
        out = []
        for line in lines:
            if not line.strip():
                out.append(line)
                continue

            # find first non space char as the string name
            idx = 0
            while idx < len(line) and line[idx] == ' ':
                idx += 1
            if idx >= len(line):
                out.append(line)
                continue

            string_name = line[idx]
            if string_name not in OPEN_PITCH:
                out.append(line)
                continue

            buf = list(line)
            i = idx + 1
            L = len(buf)

            while i < L:
                if buf[i].isdigit():
                    j = i
                    while j < L and buf[j].isdigit():
                        j += 1
                    fret_txt = ''.join(buf[i:j])
                    try:
                        fret = int(fret_txt)
                    except ValueError:
                        fret = None

                    if fret is not None:
                        note = fret_to_note(OPEN_PITCH[string_name], fret)
                        span = j - i
                        need = len(note)

                        # write note over the same grid width
                        for k, ch in enumerate(note):
                            pos = i + k
                            if pos < L:
                                buf[pos] = ch
                        end_fill = i + max(span, need)
                        for pos in range(i + need, min(end_fill, L)):
                            buf[pos] = '-'

                    i = j
                else:
                    i += 1

            out.append(''.join(buf))
        return out

    def convert_tab(text: str) -> str:
        lines = text.splitlines()
        return '\n'.join(convert_tab_block(lines))

    st.set_page_config(page_title="TAB to Notes", page_icon="🎸")

    st.title("🎸 TAB to Notes")
    st.write("Paste your TAB or upload a txt file. The app keeps spacing and layout.")

    tab_text = st.text_area(
        "Paste TAB",
        height=300,
        placeholder="e------------------------------------------------------\nB---------9------------9------------7------------7-----\nG------9-----9------9-----9------8-----8------8-----8--\nD---9------------9------------9------------9-----------\nA------------------------------------------------------\nE------------------------------------------------------",
    )

    uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])
    if uploaded is not None:
        try:
            file_text = uploaded.read().decode("utf-8")
        except Exception:
            file_text = uploaded.read().decode("latin-1")
        tab_text = file_text

    col1, col2 = st.columns(2)
    with col1:
        run = st.button("Convert")
    with col2:
        clear = st.button("Clear")

    if clear:
        st.experimental_rerun()

    if run and tab_text.strip():
        out_text = convert_tab(tab_text)
        st.subheader("Notes")
        st.code(out_text, language="text")
        st.download_button(
            "Download as notes.txt",
            data=out_text.encode("utf-8"),
            file_name="notes.txt",
            mime="text/plain",
        )
    elif run:
        st.warning("Please paste TAB or upload a file.")


def main_grok():
    import streamlit as st
    import re

    def parse_tab_to_notes(tab):
        # Define note mappings for each string (standard tuning: EADGBE)
        string_notes = {
            'e': ['E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5'],
            'B': ['B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'],
            'G': ['G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4'],
            'D': ['D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4', 'C#4', 'D4'],
            'A': ['A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3'],
            'E': ['E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3']
        }
        
        # Initialize output
        notes_output = []
        current_chord = None
        
        # Split input into lines
        lines = tab.strip().split('\n')
        
        # Process chord names (if any) and tab lines
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check if line is a chord line (e.g., "E E B B" or "F#m F#m C#m C#m")
            if re.match(r'^[A-G](?:#|b)?(?:m)?(?:\s+[A-G](?:#|b)?(?:m)?)*$', line):
                current_chord = line
                notes_output.append(f"\nChord(s): {current_chord}")
                continue
                
            # Process tablature lines
            if line.startswith(('e|', 'B|', 'G|', 'D|', 'A|', 'E|')):
                string = line[0]
                frets = line[2:].replace('-', ' ').split()
                notes = []
                for fret in frets:
                    try:
                        fret_num = int(fret)
                        if 0 <= fret_num < len(string_notes[string]):
                            notes.append(string_notes[string][fret_num])
                        else:
                            notes.append('?')  # Unknown fret
                    except ValueError:
                        notes.append('-')  # No note played
                notes_output.append(f"{string} string: {' '.join(notes)}")
        
        return '\n'.join(notes_output)

    # Streamlit app
    st.title("Guitar Tab to Notes Converter")

    # Default tablature input
    default_tab = """(Riff) E E B B
    e|------------------------------------------------------
    B|---------9------------9------------7------------7-----
    G|------9-----9------9-----9------8-----8------8-----8--
    D|---9------------9------------9------------9-----------
    A|------------------------------------------------------
    E|------------------------------------------------------
         F#m              F#m              C#m           C#m
    e|----------------------------------------------------------------
    B|-----------10---------------10--------------9-------------9-----
    G|-------11------11-------11------11-------9-----9-------9-----9--
    D|---11---------------11---------------11------------11-----------
    A|----------------------------------------------------------------
    E|----------------------------------------------------------------"""

    # Input textarea for tablature
    tab_input = st.text_area("Enter Guitar Tablature:", value=default_tab, height=300)

    # Convert button
    if st.button("Convert to Notes"):
        if tab_input.strip():
            notes = parse_tab_to_notes(tab_input)
            st.subheader("Converted Notes:")
            st.text(notes)
        else:
            st.error("Please enter some tablature to convert.")

def main_chatgpt2():
    import streamlit as st

    # ---------- Pitch helpers ----------
    NOTE_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    NOTE_TO_INDEX = {n: i for i, n in enumerate(NOTE_ORDER)}
    INDEX_TO_NOTE = {i: n for i, n in enumerate(NOTE_ORDER)}
    OPEN_MIDI = {'E': 40, 'A': 45, 'D': 50, 'G': 55, 'B': 59, 'e': 64}  # E2 A2 D3 G3 B3 E4

    def midi_to_note_name(midi: int) -> str:
        return INDEX_TO_NOTE[midi % 12]  # uses sharps

    def fret_to_midi(string_name: str, fret: int) -> int:
        return OPEN_MIDI[string_name] + fret

    # ---------- TAB → timed events ----------
    def tab_to_timed_events(tab_text: str, columns_per_quarter: int = 4):
        lines = tab_text.splitlines()
        string_lines = []
        for line in lines:
            idx = 0
            while idx < len(line) and line[idx] == ' ':
                idx += 1
            if idx < len(line) and line[idx:idx+1] in OPEN_MIDI:
                string_lines.append((line[idx], idx, line))

        if not string_lines:
            return [], []

        starts = {}
        for s, idx0, line in string_lines:
            i = idx0 + 1
            L = len(line)
            while i < L:
                if line[i].isdigit():
                    j = i
                    while j < L and line[j].isdigit():
                        j += 1
                    fret = int(line[i:j])
                    starts.setdefault(i, []).append((s, fret))
                    i = j
                else:
                    i += 1

        if not starts:
            return [], []

        cols = sorted(starts.keys())
        events = []
        for c in cols:
            midis = [fret_to_midi(s, fr) for s, fr in starts[c]]
            events.append({'time_col': c, 'midi': sorted(midis)})

        durations_cols = []
        for k, ev in enumerate(events):
            if k < len(events) - 1:
                dcols = max(1, events[k+1]['time_col'] - ev['time_col'])
            else:
                dcols = columns_per_quarter
            durations_cols.append(dcols)

        return events, durations_cols

    # ---------- Notes-in-layout ----------
    def convert_tab_to_notes_layout(tab_text: str) -> str:
        lines = tab_text.splitlines()
        out = []
        for line in lines:
            idx = 0
            while idx < len(line) and line[idx] == ' ':
                idx += 1
            if idx >= len(line):
                out.append(line)
                continue
            string_name = line[idx:idx+1]
            if string_name not in OPEN_MIDI:
                out.append(line)
                continue

            open_pc = NOTE_TO_INDEX[midi_to_note_name(OPEN_MIDI[string_name])]
            buf = list(line)
            i = idx + 1
            L = len(buf)
            while i < L:
                if buf[i].isdigit():
                    j = i
                    while j < L and buf[j].isdigit():
                        j += 1
                    fret_txt = ''.join(buf[i:j])
                    try:
                        fret = int(fret_txt)
                    except ValueError:
                        fret = None
                    if fret is not None:
                        note = INDEX_TO_NOTE[(open_pc + fret) % 12]
                        span = j - i
                        need = len(note)
                        for k, ch in enumerate(note):
                            pos = i + k
                            if pos < L:
                                buf[pos] = ch
                        end_fill = i + max(span, need)
                        for pos in range(i + need, min(end_fill, L)):
                            buf[pos] = ' '
                    i = j
                else:
                    i += 1
            out.append(''.join(buf))
        return '\n'.join(out)

    # ---------- ASCII staff (with wrapping + accidentals) ----------
    # Rows include lines and spaces around E4..F5
    STAFF_PITCHES = [
        ('A5', 81),  # extra above
        ('G5', 79),
        ('F5', 77),  # top line
        ('E5', 76),
        ('D5', 74),  # line
        ('C5', 72),
        ('B4', 71),  # line
        ('A4', 69),
        ('G4', 67),  # line
        ('F4', 65),
        ('E4', 64),  # bottom line
        ('D4', 62),  # extra below
    ]
    PITCH_TO_ROW = {midi: r for r, (_, midi) in enumerate(STAFF_PITCHES)}

    def nearest_row_for_midi(midi: int) -> int:
        return min(PITCH_TO_ROW.keys(), key=lambda m: abs(m - midi))

    def build_staff_grid(events, width_cols, col_scale=2):
        """Build one wide grid; we will wrap it later."""
        W = width_cols * col_scale + 8
        H = len(STAFF_PITCHES)
        grid = [[' ' for _ in range(W)] for _ in range(H)]

        # Draw staff lines E4 G4 B4 D5 F5
        for midi in [64, 67, 71, 74, 77]:
            r = PITCH_TO_ROW[midi]
            for c in range(3, W):
                grid[r][c] = '-'

        # G clef marker
        g_row = PITCH_TO_ROW[67]
        grid[g_row][0] = 'G'

        # Notes with accidentals
        for ev in events:
            c = 5 + ev['time_col'] * col_scale
            for midi in ev['midi']:
                target_midi = nearest_row_for_midi(midi)
                r = PITCH_TO_ROW[target_midi]
                name = midi_to_note_name(midi)
                if '#' in name and c > 0:
                    grid[r][c-1] = '#'
                if 'b' in name and c > 0:
                    grid[r][c-1] = 'b'
                grid[r][c] = 'O'

        return grid  # 2D char array

    def wrap_grid_to_blocks(grid, wrap_width):
        """Slice the wide grid into vertical blocks of wrap_width and stack them."""
        H = len(grid)
        W = len(grid[0]) if H else 0
        blocks = []
        for start in range(0, W, wrap_width):
            end = min(start + wrap_width, W)
            block_lines = [''.join(row[start:end]).rstrip() for row in grid]
            blocks.append('\n'.join(block_lines))
        return '\n\n'.join(blocks)

    def render_ascii_staff_wrapped(events, width_cols, col_scale=2, wrap_cols=80):
        grid = build_staff_grid(events, width_cols, col_scale)
        return wrap_grid_to_blocks(grid, wrap_cols)

    # ---------- UI ----------
    st.set_page_config(page_title="TAB → ASCII Staff (wrapped)", page_icon="🎼")
    st.title("🎼 TAB → ASCII Staff")
    st.write("Paste TAB. I draw a treble staff with **O** noteheads, **# b** accidentals, and wrap into multiple 5-line systems.")

    tab_text = st.text_area(
        "TAB input",
        height=260,
        placeholder="e------------------------------------------------------\nB---------9------------9------------7------------7-----\nG------9-----9------9-----9------8-----8------8-----8--\nD---9------------9------------9------------9-----------\nA------------------------------------------------------\nE------------------------------------------------------",
    )

    columns_per_quarter = st.number_input("Columns per quarter", 1, 16, 4, 1)
    col_scale = st.number_input("Horizontal scale", 1, 6, 2, 1)
    wrap_cols = st.number_input("Wrap width (characters per system)", 40, 200, 80, 5)

    colA, colB = st.columns(2)
    with colA:
        run = st.button("Render")
    with colB:
        clear = st.button("Clear")

    if clear:
        st.experimental_rerun()

    if run and tab_text.strip():
        notes_layout = convert_tab_to_notes_layout(tab_text)
        st.subheader("Notes in original layout")
        st.code(notes_layout, language="text")

        events, _ = tab_to_timed_events(tab_text, columns_per_quarter=int(columns_per_quarter))
        if not events:
            st.warning("No fret numbers found.")
        else:
            width_cols = max(ev['time_col'] for ev in events) + max(1, int(columns_per_quarter))
            ascii_staff = render_ascii_staff_wrapped(
                events,
                width_cols=width_cols,
                col_scale=int(col_scale),
                wrap_cols=int(wrap_cols)
            )
            st.subheader("ASCII treble staff (wrapped into multiple systems)")
            st.code(ascii_staff, language="text")
            st.download_button(
                "Download ascii_staff.txt",
                ascii_staff.encode("utf-8"),
                file_name="ascii_staff.txt",
                mime="text/plain",
            )
    elif run:
        st.warning("Please paste TAB.")

def main():
    main_chatgpt()
    
if __name__ == "__main__":
    main()