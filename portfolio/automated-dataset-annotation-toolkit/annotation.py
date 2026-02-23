import gradio as gr
import pandas as pd

# Load the dataset
DATA_FILE = "structured_ghazals.xlsx"
ANNOTATED_FILE = "structured_ghazals_annotated.xlsx"
df = pd.read_excel(DATA_FILE, sheet_name="Sheet1")

# List of new columns required for annotation
new_columns = [
    "Poetry", "Meter", "Qafia", "Radif", "Matla", "Maqta", "Theme",
    "Emotion", "Takhallus", "Style", "Metaphor", "Simile", "Personification",
    "Hyperbole", "Alliteration", "Imagery", "Symbolism"
]
for col in new_columns:
    if col not in df.columns:
        df[col] = ""

# Create a combined Poetry column from Misra_1 and Misra_2
df["Poetry"] = df["Misra_1"].astype(str) + "\n" + df["Misra_2"].astype(str)

# Predefined options for dropdown fields (all include "Unknown" as a default option)
meter_options = [
    "Unknown",
    "Bahr-e-Tawil (بحرِ طویل)",
    "Bahr-e-Hazaj (بحرِ ہزج)",
    "Bahr-e-Kamil (بحرِ کامل)",
    "Bahr-e-Ramal (بحرِ رمل)",
    "Bahr-e-Khafeef (بحرِ خفیف)",
    "Bahr-e-Mutadarik (بحرِ متدارک)",
    "Bahr-e-Muzare (بحرِ مضارع)"
]
theme_options = ["Unknown", "Love", "Loss", "Mysticism", "Nature"]
style_options = ["Unknown", "Classical", "Contemporary"]
binary_options = ["Unknown", "Yes", "No"]

# Global variable to keep track of the current row index
current_index = 0

def load_current_row(index):
    """Return a formatted string containing couplet details for the current row."""
    if index < len(df):
        row = df.iloc[index]
        text = (
            f"Poet: {row['Poet']}\n"
            f"Ghazal Title: {row['Ghazal_Title']}\n"
            f"Sher Number: {row['Sher_Number']}\n"
            "Couplet:\n"
            f"{row['Misra_1']}\n{row['Misra_2']}"
        )
        return text
    else:
        return "All rows annotated."

def annotate_row(
    poet, meter, qafia, radif, matla, maqta, theme, emotion, takhallus, style,
    metaphor, simile, personification, hyperbole, alliteration, imagery, symbolism,
    curr_index
):
    global df
    # Save the annotations for the current row
    df.at[curr_index, "Poet"] = poet
    df.at[curr_index, "Meter"] = meter
    df.at[curr_index, "Qafia"] = qafia
    df.at[curr_index, "Radif"] = radif
    df.at[curr_index, "Matla"] = matla
    df.at[curr_index, "Maqta"] = maqta
    df.at[curr_index, "Theme"] = theme
    df.at[curr_index, "Emotion"] = emotion
    df.at[curr_index, "Takhallus"] = takhallus
    df.at[curr_index, "Style"] = style
    df.at[curr_index, "Metaphor"] = metaphor
    df.at[curr_index, "Simile"] = simile
    df.at[curr_index, "Personification"] = personification
    df.at[curr_index, "Hyperbole"] = hyperbole
    df.at[curr_index, "Alliteration"] = alliteration
    df.at[curr_index, "Imagery"] = imagery
    df.at[curr_index, "Symbolism"] = symbolism
    
    # Move to the next row
    new_index = curr_index + 1
    if new_index >= len(df):
        # Save file if annotation is complete
        df.to_excel(ANNOTATED_FILE, index=False)
        return "Annotation complete! Data saved.", new_index
    else:
        next_row = load_current_row(new_index)
        return next_row, new_index

# Build the Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# Urdu Ghazal Annotation Tool")
    
    # State component to hold the current row index
    curr_index_state = gr.State(0)
    
    # Display for current couplet details
    couplet_display = gr.Textbox(label="Current Couplet", interactive=False, lines=6, value=load_current_row(0))
    
    # Annotation inputs
    poet_input = gr.Textbox(label="Poet", value="Unknown")
    meter_input = gr.Dropdown(label="Meter (بحر)", choices=meter_options, value="Unknown")
    qafia_input = gr.Textbox(label="Qafia (قافیہ)", value="Unknown")
    radif_input = gr.Textbox(label="Radif (ردیف)", value="Unknown")
    matla_input = gr.Dropdown(label="Matla", choices=binary_options, value="Unknown")
    maqta_input = gr.Dropdown(label="Maqta", choices=binary_options, value="Unknown")
    theme_input = gr.Dropdown(label="Theme", choices=theme_options, value="Unknown")
    emotion_input = gr.Textbox(label="Emotion", value="Unknown")
    takhallus_input = gr.Textbox(label="Takhallus", value="Unknown")
    style_input = gr.Dropdown(label="Style", choices=style_options, value="Unknown")
    
    # Poetic Devices inputs (each gets its own field)
    metaphor_input = gr.Textbox(label="Metaphor", value="Unknown")
    simile_input = gr.Textbox(label="Simile", value="Unknown")
    personification_input = gr.Textbox(label="Personification", value="Unknown")
    hyperbole_input = gr.Textbox(label="Hyperbole", value="Unknown")
    alliteration_input = gr.Textbox(label="Alliteration", value="Unknown")
    imagery_input = gr.Textbox(label="Imagery", value="Unknown")
    symbolism_input = gr.Textbox(label="Symbolism", value="Unknown")
    
    submit_button = gr.Button("Submit Annotation")
    
    # When the submit button is clicked, update the dataset and display the next row
    submit_button.click(
        annotate_row,
        inputs=[
            poet_input, meter_input, qafia_input, radif_input, 
            matla_input, maqta_input, theme_input, emotion_input, 
            takhallus_input, style_input,
            metaphor_input, simile_input, personification_input, 
            hyperbole_input, alliteration_input, imagery_input, symbolism_input,
            curr_index_state
        ],
        outputs=[couplet_display, curr_index_state]
    )
    
demo.launch()
