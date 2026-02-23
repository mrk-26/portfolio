import gradio as gr
import pandas as pd
import os
import json
import requests
import time
from urduhack.normalization import normalize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_FILE = "structured_ghazals.xlsx"
ANNOTATED_FILE = "structured_ghazals_annotated.xlsx"
CHECKPOINT_FILE = "checkpoint.json"
GROQ_API_KEY = "gsk_REDACTED"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Rate limiting configuration (based on deepseek-r1-distill-llama-70b limits)
MAX_REQUESTS_PER_MINUTE = 25
REQUEST_INTERVAL = 2  # 2 seconds between requests
last_request_time = 0

# Required columns
REQUIRED_COLUMNS = [
    "Poet", "Meter", "Meter_Feedback", "Qafia", "Radif", "Matla", "Maqta", "Theme",
    "Emotion", "Takhallus", "Style", "Metaphor", "Simile", "Personification",
    "Hyperbole", "Alliteration", "Imagery", "Symbolism", "Poetry",
    "Ghazal_Title", "Sher_Number"
]

# Accurate Bahar patterns (based on Urdu prosody)
BAHR_PATTERNS = {
    "Bahr-e-Hazaj": [1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2],
    "Bahr-e-Rajaz": [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    "Bahr-e-Ramal": [2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2],
    "Bahr-e-Kamil": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "Bahr-e-Wafir": [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
    "Bahr-e-Mutaqarib": [2, 1, 2, 2, 1, 2, 2, 1, 2],
    "Bahr-e-Mutadarik": [2, 1, 2, 2, 1, 2, 2, 1, 2],
    "Bahr-e-Muzare": [2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2],
    "Bahr-e-Mujtas": [2, 1, 2, 1, 2, 2, 1, 2],
    "Bahr-e-Munsarah": [2, 1, 2, 1, 2, 2, 2, 1],
    "Bahr-e-Muqtazib": [2, 2, 2, 1, 2, 1, 2, 1],
    "Bahr-e-Saree": [2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1],
    "Bahr-e-Khafif": [2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2],
    "Bahr-e-Qarib": [2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2],
    "Bahr-e-Jadid": [2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1],
    "Bahr-e-Mushakil": [2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2],
    "Bahr-e-Taweel": [2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2],
    "Bahr-e-Madid": [2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2],
    "Bahr-e-Baseet": [2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2]
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

# Rate-limited API request function
def rate_limited_request(payload):
    global last_request_time
    current_time = time.time()
    elapsed = current_time - last_request_time
    
    # Enforce rate limiting
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        last_request_time = time.time()
        return response
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return rate_limited_request(payload)
        else:
            raise e

# LLM-based full annotation
def auto_annotate_couplet(poetry_text, poet_name):
    """Automatically annotate all aspects using Groq API"""
    prompt = f"""
You are an expert in Urdu prosody (Ilm-e-Arooz) and poetry analysis (Ilm-e-Bayan). 
Analyze this couplet by {poet_name}:

{poetry_text}

Provide COMPLETE annotations in JSON format with these keys:

1. Meter (بحر):
   - Identify the metrical pattern using standard Urdu prosody
   - Rules for determination:
     a. Break each line into syllables using vowel nuclei (ا, آ, ئ, ے, ی, و, ؤ, ُ, َ, ِ)
     b. Assign weights: 
        - Short (1): Syllables ending with short vowels (َ, ِ, ُ) 
        - Long (2): Syllables ending with long vowels (ا, آ, و, ی, ے) or consonants
     c. Compare weight sequences to standard patterns:
        {json.dumps(BAHR_PATTERNS, indent=4, ensure_ascii=False)}
     d. Allow for valid variations (zihaf/mahzoof)
     e. Both lines MUST share the same pattern
   - Use standard names: "Bahr-e-Hazaj", "Bahr-e-Ramal", etc.
   - If uncertain, provide closest match with explanation

2. Meter_Feedback:
   - Explain analysis: syllable weights for both lines
   - Note any variations from pure form
   - Mention matching pattern name and structure

3. Qafia (قافیہ):
   - Identify the rhyming pattern BEFORE Radif
   - Extract common ending sound across both lines
   - Example: In "dard-e-dil kā dāwa kyā hai / ishq kī intehā kyā hai", Qafia = "ā"

4. Radif (ردیف):
   - Identify repeated word/phrase AFTER Qafia
   - Must be identical in both lines
   - Example: In above, Radif = "kyā hai"

5. Matla:
   - "Yes" if BOTH:
     a. First couplet of ghazal
     b. Both lines end with same Qafia+Radif
   - Otherwise "No"

6. Maqta:
   - "Yes" if BOTH:
     a. Last couplet of ghazal
     b. Contains poet's Takhallus (pen name)
   - Otherwise "No"

7. Theme:
   - Select primary theme from: 
     ["Love", "Loss", "Mysticism", "Nature", "Philosophical", 
     "Patriotic", "Social Commentary", "Spiritual", "Nostalgia"]
   - Add brief explanation if needed

8. Emotion:
   - Identify dominant emotion: 
     ["Longing", "Joy", "Sorrow", "Despair", "Hope", 
     "Nostalgia", "Devotion", "Wonder", "Resignation"]

9. Takhallus:
   - Extract poet's pen name if present in couplet
   - Leave blank if absent

10. Style:
    - "Classical" for traditional diction/imagery
    - "Contemporary" for modern language/themes

11. Poetic Devices (provide actual phrases):
    - Metaphor: Implied comparisons ("chiragh-e-dil" = heart's lamp)
    - Simile: Explicit comparisons using "like" or "as" ("aankhen sharab ki jaise")
    - Personification: Human traits to non-human ("sham-e-gham muskura rahi hai")
    - Hyperbole: Exaggeration ("hazaar aahon se")
    - Alliteration: Sound repetition ("dil dhadakne laga")
    - Imagery: Sensory descriptions ("khushboo ki tarah")
    - Symbolism: Objects representing ideas ("saba = messenger")
 
Output ONLY JSON. Example:
{{
  "Meter": "Bahr-e-Hazaj",
  "Meter_Feedback": "Line1 weights: [1,2,2,2,1,2,2,2] matches mafa'ilun pattern (3 repetitions). Minor variation in position 5 (2→1 acceptable as zihaf).",
  "Qafia": "ān",
  "Radif": "hai",
  "Matla": "No",
  "Maqta": "Yes",
  "Theme": "Love",
  "Emotion": "Longing",
  "Takhallus": "Ghalib",
  "Style": "Classical",
  "Metaphor": ["dil ko marnay wala = beloved"],
  "Simile": [],
  "Personification": ["sham-e-gham muskura rahi hai = evening of grief smiles"],
  "Hyperbole": ["hazaar aahon se = with a thousand sighs"],
  "Alliteration": ["dil dhadakne laga"],
  "Imagery": ["virane mein akeli chiragh = lone lamp in desolation"],
  "Symbolism": ["chiragh = hope"]
}}
strictly follow the give output example structure don't generate anything in roman urdu strictly follow urdu text 
"""
    
    payload = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2500
    }
    
    try:
        response = rate_limited_request(payload)
        result = response.json()
        return json.loads(result['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Annotation error: {str(e)}")
        return {
            "Meter": "Unknown",
            "Meter_Feedback": f"Analysis failed: {str(e)}",
            "Qafia": "Unknown",
            "Radif": "Unknown",
            "Matla": "Unknown",
            "Maqta": "Unknown",
            "Theme": "Unknown",
            "Emotion": "Unknown",
            "Takhallus": "",
            "Style": "Unknown",
            "Metaphor": [],
            "Simile": [],
            "Personification": [],
            "Hyperbole": [],
            "Alliteration": [],
            "Imagery": [],
            "Symbolism": []
        }

def auto_annotate(curr_index):
    """Automatically annotate current couplet using LLM"""
    global df
    row = df.iloc[curr_index]
    poetry_text = row['Poetry']
    poet_name = row.get('Poet', 'Unknown Poet')
    
    annotation = auto_annotate_couplet(poetry_text, poet_name)
    
    # Update DataFrame with annotation results
    for key, value in annotation.items():
        if key in df.columns:
            if isinstance(value, list):
                df.at[curr_index, key] = ", ".join(value)
            else:
                df.at[curr_index, key] = value
    
    # Save after each annotation
    df.to_excel(ANNOTATED_FILE, index=False)
    return load_current_row(curr_index)

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
        .api-status {
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 10px;
            text-align: center;
        }
        .api-connected {
            background-color: #4CAF50;
            color: white;
        }
        .api-disconnected {
            background-color: #F44336;
            color: white;
        }
        .auto-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
            color: white !important;
            font-weight: bold !important;
        }
        .rate-info {
            background-color: #2196F3;
            color: white;
            padding: 8px;
            border-radius: 4px;
            margin: 10px 0;
            text-align: center;
        }
    """) as demo:
        gr.HTML("<link href='https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap' rel='stylesheet'>")
        gr.Markdown("# 🌸 Urdu Ghazal Annotation Tool", elem_classes="couplet-card")

        # API status and rate info
        api_status = "connected" if GROQ_API_KEY else "disconnected"
        status_class = "api-connected" if GROQ_API_KEY else "api-disconnected"
        gr.Markdown(f"""
        <div class='api-status {status_class}'>
            Groq API Status: {api_status.upper()} | Model: deepseek-r1-distill-llama-70b
        </div>
        <div class='rate-info'>
            Rate Limit: {MAX_REQUESTS_PER_MINUTE} requests/minute | Current Delay: {REQUEST_INTERVAL:.1f}s/request
        </div>
        """, elem_id="api_status")

        curr_state = gr.State(current_index)

        with gr.Row(elem_classes="fixed-header"):
            couplet_display = gr.Markdown("", elem_id="couplet_display", elem_classes="couplet-card")
            progress = gr.Markdown("", elem_id="progress")

        with gr.Group(elem_classes="gr-main-container"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Core Metadata", open=True):
                        poet = gr.Textbox(label="Poet")
                        style = gr.Dropdown(["Unknown", "Classical", "Contemporary"], label="Style")
                        theme = gr.Dropdown(["Unknown", "Love", "Loss", "Mysticism", "Nature", 
                                            "Philosophical", "Patriotic", "Social Commentary", 
                                            "Spiritual", "Nostalgia"], label="Theme")
                        emotion = gr.Dropdown(["Unknown", "Longing", "Joy", "Sorrow", "Despair", 
                                             "Hope", "Nostalgia", "Devotion", "Wonder", 
                                             "Resignation"], label="Dominant Emotion")
                    with gr.Accordion("Structural Elements", open=True):
                        meter = gr.Textbox(label="Meter (بحر)")
                        meter_feedback = gr.Textbox(label="Meter Feedback", lines=2)
                        qafia = gr.Textbox(label="Qafia (قافیہ)")
                        radif = gr.Textbox(label="Radif (ردیف)")
                        matla = gr.Textbox(label="Matla")
                        maqta = gr.Textbox(label="Maqta")
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
                    gr.Markdown("### 🤖 Auto-Annotation Guide")
                    gr.Markdown("""
- Full automation using Groq's deepseek-r1-distill-llama-70b model
- Meter detection: Identifies 19+ Urdu bahar patterns
- Structural analysis: Qafia, Radif, Matla, Maqta
- Poetic devices: Metaphor, Simile, Personification, etc.
- Thematic analysis: Emotion, Theme, Style

Click "Auto-Annotate" to analyze current couplet
""")
                    gr.Markdown("### ⚠️ Rate Limit Info")
                    gr.Markdown(f"""
- Max requests: {MAX_REQUESTS_PER_MINUTE}/minute
- Enforced delay: {REQUEST_INTERVAL:.1f} seconds between requests
- Current queue position: {current_index + 1} of {len(df)}
""")

            with gr.Row():
                auto_btn = gr.Button("🚀 Auto-Annotate", variant="primary", elem_classes="auto-btn")
                submit_btn = gr.Button("💾 Save & Next")
                next_btn = gr.Button("⏭️ Next Without Saving")

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
            
            next_btn.click(
                lambda idx: load_current_row(idx + 1),
                inputs=[curr_state],
                outputs=outputs
            )
            
            auto_btn.click(
                auto_annotate,
                inputs=[curr_state],
                outputs=outputs
            )
            
            demo.load(
                load_current_row,
                inputs=[curr_state],
                outputs=outputs
            )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        auth=("admin", "password") if os.getenv("ENABLE_AUTH") else None
    )