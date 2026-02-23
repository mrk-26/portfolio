import gradio as gr
import pandas as pd
import os
import json

# Configuration
DATA_FILE = "structured_ghazals.xlsx"
ANNOTATED_FILE = "structured_ghazals_annotated.xlsx"
CHECKPOINT_FILE = "checkpoint.json"

# Required columns
REQUIRED_COLUMNS = [
    "Poet", "Meter", "Qafia", "Radif", "Matla", "Maqta", "Theme",
    "Emotion", "Takhallus", "Style", "Metaphor", "Simile", "Personification",
    "Hyperbole", "Alliteration", "Imagery", "Symbolism", "Poetry",
    "Ghazal_Title", "Sher_Number"
]

# Predefined options
METER_OPTIONS = [
    "Unknown",
    "Bahr-e-Tawil (بحرِ طویل)",
    "Bahr-e-Hazaj (بحرِ ہزج)",
    "Bahr-e-Kamil (بحرِ کامل)",
    "Bahr-e-Ramal (بحرِ رمل)",
    "Bahr-e-Khafeef (بحرِ خفیف)",
    "Bahr-e-Mutadarik (بحرِ متدارک)",
    "Bahr-e-Muzare (بحرِ مضارع)"
]
THEME_OPTIONS = ["Unknown", "Love", "Loss", "Mysticism", "Nature"]
STYLE_OPTIONS = ["Unknown", "Classical", "Contemporary"]
BINARY_OPTIONS = ["Unknown", "Yes", "No"]

# Initialize DataFrame
def initialize_dataframe():
    if os.path.exists(ANNOTATED_FILE):
        df = pd.read_excel(ANNOTATED_FILE)
    else:
        df = pd.read_excel(DATA_FILE)
    # Ensure all required columns exist
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
        "---  \n"
        f"{poetry_lines[0]}  \n "
        f"{poetry_lines[1]}  \n "
    )
    return (
        couplet_md,
        vals['Poet'], vals['Meter'], vals['Qafia'], vals['Radif'],
        vals['Matla'], vals['Maqta'], vals['Theme'], vals['Emotion'],
        vals['Takhallus'], vals['Style'], vals['Metaphor'], vals['Simile'],
        vals['Personification'], vals['Hyperbole'], vals['Alliteration'],
        vals['Imagery'], vals['Symbolism'],
        f"Progress: {index+1}/{len(df)}",
        index
    )

# Annotation function
def annotate_row(poet, meter, qafia, radif, matla, maqta, theme, emotion,
                takhallus, style, metaphor, simile, personification,
                hyperbole, alliteration, imagery, symbolism, curr_index):
    global df, current_index
    fields = [
        'Poet','Meter','Qafia','Radif','Matla','Maqta','Theme','Emotion',
        'Takhallus','Style','Metaphor','Simile','Personification',
        'Hyperbole','Alliteration','Imagery','Symbolism'
    ]
    values = [
        poet, meter, qafia, radif, matla, maqta, theme, emotion,
        takhallus, style, metaphor, simile, personification,
        hyperbole, alliteration, imagery, symbolism
    ]
    for field, val in zip(fields, values):
        df.at[curr_index, field] = val
    df.to_excel(ANNOTATED_FILE, index=False)
    current_index = curr_index + 1
    save_checkpoint(current_index)
    return load_current_row(current_index)

# Build Gradio interface
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

        # Fixed header: couplet + progress
        with gr.Row(elem_classes="fixed-header"):
            couplet_display = gr.Markdown("", elem_id="couplet_display", elem_classes="couplet-card")
            progress = gr.Markdown("", elem_id="progress")

        # Main content sits below the fixed header
        with gr.Group(elem_classes="gr-main-container"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Core Metadata", open=True):
                        poet = gr.Textbox(label="Poet")
                        style = gr.Dropdown(STYLE_OPTIONS, label="Style")
                        theme = gr.Dropdown(THEME_OPTIONS, label="Theme")
                        emotion = gr.Textbox(label="Dominant Emotion")
                    with gr.Accordion("Structural Elements", open=True):
                        meter = gr.Dropdown(METER_OPTIONS, label="Meter (بحر)")
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

            # Submit button
            with gr.Row():
                submit_btn = gr.Button("💾 Save & Next", variant="primary")

            inputs = [
                poet, meter, qafia, radif, matla, maqta,
                theme, emotion, takhallus, style,
                metaphor, simile, personification,
                hyperbole, alliteration, imagery, symbolism,
                curr_state
            ]
            outputs = [
                couplet_display,
                poet, meter, qafia, radif, matla, maqta,
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

            demo.load(
                load_current_row,
                inputs=[curr_state],
                outputs=outputs
            )

    return demo

if __name__ == "__main__":
    build_interface().launch(share=True)
