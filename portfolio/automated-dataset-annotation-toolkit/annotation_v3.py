import gradio as gr
import pandas as pd
import os
import json
from urduhack.normalization import normalize
from indicnlp.tokenize import indic_tokenize

# Configuration
DATA_FILE = "structured_ghazals.xlsx"
ANNOTATED_FILE = "structured_ghazals_annotated.xlsx"
CHECKPOINT_FILE = "checkpoint.json"

# Required columns (added Meter_Feedback)
REQUIRED_COLUMNS = [
    "Poet", "Meter", "Meter_Feedback", "Qafia", "Radif", "Matla", "Maqta", "Theme",
    "Emotion", "Takhallus", "Style", "Metaphor", "Simile", "Personification",
    "Hyperbole", "Alliteration", "Imagery", "Symbolism", "Poetry",
    "Ghazal_Title", "Sher_Number"
]

# Predefined options (19 main Bahar types)
METER_OPTIONS = [
    "Unknown",
    "Bahr-e-Hazaj",
    "Bahr-e-Rajaz",
    "Bahr-e-Ramal",
    "Bahr-e-Kamil",
    "Bahr-e-Wafir",
    "Bahr-e-Mutaqarib",
    "Bahr-e-Mutadarik",
    "Bahr-e-Muzare",
    "Bahr-e-Mujtas",
    "Bahr-e-Munsarah",
    "Bahr-e-Muqtazib",
    "Bahr-e-Saree",
    "Bahr-e-Khafif",
    "Bahr-e-Qarib",
    "Bahr-e-Jadid",
    "Bahr-e-Mushakil",
    "Bahr-e-Taweel",
    "Bahr-e-Madid",
    "Bahr-e-Baseet"
]

THEME_OPTIONS = ["Unknown", "Love", "Loss", "Mysticism", "Nature"]
STYLE_OPTIONS = ["Unknown", "Classical", "Contemporary"]
BINARY_OPTIONS = ["Unknown", "Yes", "No"]

# Accurate Bahar patterns (based on Urdu prosody)
BAHR_PATTERNS = {
    "Bahr-e-Hazaj": [1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2],  # mafa'īlun x3
    "Bahr-e-Rajaz": [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],  # mustaf'ilun x3
    "Bahr-e-Ramal": [2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2],  # fā'ilātun x3
    "Bahr-e-Kamil": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # mutafā'ilun x3
    "Bahr-e-Wafir": [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],  # mufā'ilatun x3
    "Bahr-e-Mutaqarib": [2, 1, 2, 2, 1, 2, 2, 1, 2],      # fa'ūlun x3
    "Bahr-e-Mutadarik": [2, 1, 2, 2, 1, 2, 2, 1, 2],      # fā'ilun x3
    "Bahr-e-Muzare": [2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2],# mafa'īlun fa'ūlun x2
    "Bahr-e-Mujtas": [2, 1, 2, 1, 2, 2, 1, 2],            # mustaf'ilun fā'ilātun
    "Bahr-e-Munsarah": [2, 1, 2, 1, 2, 2, 2, 1],          # mustaf'ilun maf'ūlātu
    "Bahr-e-Muqtazib": [2, 2, 2, 1, 2, 1, 2, 1],          # maf'ūlātu mustaf'ilun
    "Bahr-e-Saree": [2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1], # mustaf'ilun mustaf'ilun maf'ūlātu
    "Bahr-e-Khafif": [2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2],# fā'ilātun mustaf'ilun fā'ilātun
    "Bahr-e-Qarib": [2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2], # mafa'īlun mafa'īlun fā'ilātun
    "Bahr-e-Jadid": [2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1], # fā'ilātun fā'ilātun mustaf'ilun
    "Bahr-e-Mushakil": [2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2],# fā'ilātun mafa'īlun mafa'īlun
    "Bahr-e-Taweel": [2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2], # fa'ūlun mafa'īlun x2
    "Bahr-e-Madid": [2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2],    # fā'ilātun fā'ilun fā'ilātun
    "Bahr-e-Baseet": [2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2]   # mustaf'ilun fā'ilun mustaf'ilun
}

# Initialize DataFrame
def initialize_dataframe():
    if os.path.exists(ANNOTATED_FILE):
        df = pd.read_excel(ANNOTATED_FILE)
    else:
        df = pd.read_excel(DATA_FILE)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "Poetry":
                df["Poetry"] = df.get("Misra_1", "").astype(str) + "\n" + df.get("Misra_2", "").astype(str)
            else:
                df[col] = "Unknown"
    return df

# Checkpoint handling
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f).get('current_index', 0)
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'current_index': index}, f)

# Load data and checkpoint
df = initialize_dataframe()
current_index = load_checkpoint()

def get_row_values(index):
    if index >= len(df):
        return None
    row = df.iloc[index]
    vals = {}
    for col in REQUIRED_COLUMNS:
        val = row.get(col, "Unknown")
        vals[col] = "Unknown" if pd.isna(val) or val == "" else str(val)
    return vals

def load_current_row(index):
    if index >= len(df):
        return (
            "All annotations complete!",
            *["Unknown"] * (len(REQUIRED_COLUMNS)-1),
            f"Progress: {len(df)}/{len(df)}",
            index
        )
    vals = get_row_values(index)
    poetry_lines = vals['Poetry'].split("\n")
    couplet_md = (
        f"Poet: {vals['Poet']}  \n"
        f"Ghazal Title: {vals['Ghazal_Title']}  \n"
        f"Sher Number: {vals['Sher_Number']}  \n"
        f"---  \n"
        f"{poetry_lines[0]}  \n "
        f"{poetry_lines[1]}  \n "
    )
    return (
        couplet_md,
        vals['Poet'], vals['Meter'], vals['Meter_Feedback'], vals['Qafia'], vals['Radif'],
        vals['Matla'], vals['Maqta'], vals['Theme'], vals['Emotion'],
        vals['Takhallus'], vals['Style'], vals['Metaphor'], vals['Simile'],
        vals['Personification'], vals['Hyperbole'], vals['Alliteration'],
        vals['Imagery'], vals['Symbolism'],
        f"Progress: {index+1}/{len(df)}",
        index
    )

def normalize_urdu_text(text):
    """Normalize Urdu text using urduhack."""
    return normalize(text.strip())

def split_into_syllables(word):
    """Split Urdu word into syllables."""
    chars = list(indic_tokenize.trivial_tokenize(word, lang='ur'))
    syllables = []
    current_syllable = ""
    vowels = set('اآئیےیؤوَُِ')  # Urdu vowels and diacritics
    for i, char in enumerate(chars):
        current_syllable += char
        if char in vowels or (i + 1 < len(chars) and chars[i + 1] in vowels):
            syllables.append(current_syllable)
            current_syllable = ""
    if current_syllable:
        syllables.append(current_syllable)
    return syllables

def assign_vazn(syllable):
    """Assign weight to a syllable: 1 (short), 2 (long)."""
    short_vowels = set('َُِ')  # Kasra, Zabar, Pesh
    if any(syllable.endswith(v) for v in short_vowels):
        return 1
    return 2

def get_syllable_weights(line):
    """Get syllable weights for a line of poetry."""
    normalized = normalize_urdu_text(line.strip())
    words = normalized.split()
    weights = []
    for word in words:
        syllables = split_into_syllables(word)
        for syllable in syllables:
            weights.append(assign_vazn(syllable))
    return weights

def matches_pattern(line_weights, pattern):
    """Check if line_weights match the repeated pattern."""
    pattern_len = len(pattern)
    if pattern_len == 0:
        return False
    repetitions = len(line_weights) // pattern_len
    if repetitions * pattern_len != len(line_weights):
        return False
    full_pattern = pattern * repetitions
    return line_weights == full_pattern

def identify_beher(poetry_text):
    """Identify the Bahar of a couplet."""
    lines = poetry_text.strip().split("\n")
    if len(lines) != 2:
        return "Unknown", "Error: Couplet must have exactly two lines"
    line1_weights = get_syllable_weights(lines[0])
    line2_weights = get_syllable_weights(lines[1])
    
    for bahr_name, pattern in BAHR_PATTERNS.items():
        if matches_pattern(line1_weights, pattern) and matches_pattern(line2_weights, pattern):
            return bahr_name, f"Matched {bahr_name} with pattern {pattern}"
    return "Unknown", f"No match found. Line 1 weights: {line1_weights}, Line 2 weights: {line2_weights}"

def suggest_meter(curr_index):
    """Suggest meter for the current couplet and update DataFrame."""
    global df
    poetry_text = df.at[curr_index, 'Poetry']
    suggested_meter, message = identify_beher(poetry_text)
    df.at[curr_index, 'Meter'] = suggested_meter
    df.at[curr_index, 'Meter_Feedback'] = message
    return suggested_meter, message

def annotate_row(poet, meter, meter_feedback, qafia, radif, matla, maqta, theme, emotion,
                takhallus, style, metaphor, simile, personification,
                hyperbole, alliteration, imagery, symbolism, curr_index):
    global df, current_index
    fields = [
        'Poet', 'Meter', 'Meter_Feedback', 'Qafia', 'Radif', 'Matla', 'Maqta', 'Theme', 'Emotion',
        'Takhallus', 'Style', 'Metaphor', 'Simile', 'Personification',
        'Hyperbole', 'Alliteration', 'Imagery', 'Symbolism'
    ]
    values = [
        poet, meter, meter_feedback, qafia, radif, matla, maqta, theme, emotion,
        takhallus, style, metaphor, simile, personification,
        hyperbole, alliteration, imagery, symbolism
    ]
    for field, val in zip(fields, values):
        df.at[curr_index, field] = val
    df.to_excel(ANNOTATED_FILE, index=False)
    current_index = curr_index + 1
    save_checkpoint(current_index)
    return load_current_row(current_index)

def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(),
                   css="""
        @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap');
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #000;
            z-index: 1000;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            color: #fff;
        }
        .couplet-card {
            font-family: 'Noto Nastaliq Urdu', serif;
            font-size: 20px;
            line-height: 1.8;
            direction: rtl;
            text-align: right;
            color: #fff;
        }
        .gr-main-container {
            margin-top: 140px;
        }
    """) as demo:
        gr.HTML("<link href='https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap' rel='stylesheet'>")
        gr.Markdown("# 🌸 Urdu Ghazal Annotation Tool", elem_classes="couplet-card")

        curr_state = gr.State(current_index)

        with gr.Row(elem_classes="fixed-header"):
            couplet_display = gr.Markdown("", elem_id="couplet_display", elem_classes="couplet-card")
            progress = gr.Markdown("", elem_id="progress")

        with gr.Group(elem_classes="gr-main-container"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Core Metadata", open=True):
                        poet = gr.Textbox(label="Poet")
                        style = gr.Dropdown(STYLE_OPTIONS, label="Style")
                        theme = gr.Dropdown(THEME_OPTIONS, label="Theme")
                        emotion = gr.Textbox(label="Dominant Emotion")
                    with gr.Accordion("Structural Elements", open=True):
                        meter = gr.Dropdown(METER_OPTIONS, label="Meter (بحر)", allow_custom_value=True)
                        meter_feedback = gr.Textbox(label="Meter Feedback", interactive=True)
                        qafia = gr.Textbox(label="Qafia (قافیہ)")
                        radif = gr.Textbox(label="Radif (ردیف)")
                        matla = gr.Radio(BINARY_OPTIONS, label="Matla")
                        maqta = gr.Radio(BINARY_OPTIONS, label="Maqta")
                        takhallus = gr.Textbox(label="Takhallus")
                    with gr.Accordion("Poetic Devices", open=True):
                        metaphor = gr.Textbox(label="Metaphor")
                        simile = gr.Textbox(label="Simile")
                        personification = gr.Textbox(label="Personification")
                        hyperbole = gr.Textbox(label="Hyperbole")
                        alliteration = gr.Textbox(label="Alliteration")
                        imagery = gr.Textbox(label="Imagery")
                        symbolism = gr.Textbox(label="Symbolism")
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Quick Reference Guide")
                    gr.Markdown("""
- Qafia: Repeating rhyme scheme before Radif  
- Radif: Repeated word/phrase at end of couplet  
- Matla: Opening couplet with both lines rhyming  
- Maqta: Final couplet containing poet's pen name  
- Takhallus: Poet's pen name used in Maqta  
""")

            with gr.Row():
                suggest_btn = gr.Button("Suggest Meter")
                submit_btn = gr.Button("💾 Save & Next", variant="primary")

            inputs = [
                poet, meter, meter_feedback, qafia, radif, matla, maqta,
                theme, emotion, takhallus, style,
                metaphor, simile, personification,
                hyperbole, alliteration, imagery, symbolism,
                curr_state
            ]
            outputs = [
                couplet_display,
                poet, meter, meter_feedback, qafia, radif, matla, maqta,
                theme, emotion, takhallus, style,
                metaphor, simile, personification,
                hyperbole, alliteration, imagery, symbolism,
                progress,
                curr_state
            ]

            submit_btn.click(
                annotate_row,
                inputs=inputs,
                outputs=outputs
            )
            suggest_btn.click(
                suggest_meter,
                inputs=[curr_state],
                outputs=[meter, meter_feedback]
            )
            demo.load(
                load_current_row,
                inputs=[curr_state],
                outputs=outputs
            )

    return demo

if __name__ == "__main__":
    build_interface().launch(share=True)