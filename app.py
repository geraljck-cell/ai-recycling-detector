import io
import json
import socket
import ollama
import gradio as gr
from PIL import Image

# ==========================================
# AI RECYCLING DETECTOR - ADVANCED EDITION
# Uses local vision AI. No API key required.
# ==========================================

MODEL_NAME = "qwen2.5vl:3b"


# ==========================================
# RECYCLING CATEGORIES
# ==========================================

RECYCLING_INFO = {
    "plastic": {
        "category": "Plastic recycling",
        "advice": (
            "Empty and rinse the container. Recycle it only if "
            "your local programme accepts that type of plastic."
        )
    },
    "glass": {
        "category": "Glass bottle and jar recycling",
        "advice": (
            "Empty and rinse bottles and jars. Mirrors, ceramics "
            "and porcelain do not belong in ordinary glass recycling."
        )
    },
    "metal": {
        "category": "Metal recycling",
        "advice": (
            "Empty and clean accepted metal cans and packaging. "
            "Use a specialist service for hazardous metal items."
        )
    },
    "paper": {
        "category": "Paper recycling",
        "advice": (
            "Recycle only clean, dry paper. Greasy, wet or heavily "
            "food-soiled paper generally belongs in general waste."
        )
    },
    "cardboard": {
        "category": "Cardboard recycling",
        "advice": (
            "Flatten clean, dry cardboard. Heavily food-soiled or "
            "greasy cardboard generally belongs in general waste."
        )
    },
    "rubber": {
        "category": "Special rubber collection",
        "advice": (
            "Rubber tyres and most rubber products do not belong "
            "in ordinary household recycling. Check specialist "
            "collection options."
        )
    },
    "wood": {
        "category": "Wood reuse or specialist collection",
        "advice": (
            "Reuse the wood if possible. Painted or treated wood "
            "may need a separate disposal service."
        )
    },
    "fabric": {
        "category": "Textile collection",
        "advice": (
            "Keep textiles clean and dry. Use a textile collection "
            "service that accepts the item."
        )
    },
    "leather": {
        "category": "Reuse or specialist collection",
        "advice": (
            "Reuse the item if possible or check specialist "
            "collection and local disposal options."
        )
    },
    "ceramic": {
        "category": "Special ceramic disposal",
        "advice": (
            "Ceramics and porcelain do not belong in glass recycling. "
            "Reuse them if possible or check local disposal rules."
        )
    },
    "electronic": {
        "category": "Electronic waste (e-waste)",
        "advice": (
            "Use an approved electronic-waste collection point. "
            "Batteries may require separate handling."
        )
    },
    "mixed": {
        "category": "Mixed-material collection",
        "advice": (
            "Separate materials where possible. Recycle individual "
            "parts only if your local programme accepts them."
        )
    }
}


# ==========================================
# AI INSTRUCTIONS
# ==========================================

AI_INSTRUCTIONS = """
You are an advanced visual material identification and recycling AI.

Carefully examine the provided photograph and return ONLY valid JSON.

Your most important job is to identify the MAIN PHYSICAL OBJECT,
not the food, dirt, stains, or other things attached to it.

CRITICAL RULES:

1. A plastic container with sauce on it is PLASTIC, not food waste.
2. A metal can with food scraps stuck to it is METAL, not food waste.
3. A cardboard box with food stains is CARDBOARD, not food waste.
4. Food stains, crumbs, stuck-on food, grease, mud, dust and dirt
   are contamination when attached to another identifiable object.
5. Only classify an item as food waste when the main subject itself
   is actual discarded food or food peels, rather than another object.
6. If food scraps are attached to a container, identify the container.
7. If the main object is a rock, stone, leaves, soil or garden waste,
   do not classify it as recyclable packaging.
8. If the object is electronic waste or a battery, recommend the
   appropriate specialist collection route.
9. Never invent a material when the photograph is unclear.
10. Do not claim that an item is contaminated if there is no visible
    evidence. Do not claim that hidden germs or chemicals can be seen.
11. Treat your confidence score as an estimate, not a probability.
12. Only report contamination when visible evidence supports it.
13. Use simple, specific language.

Choose one material:
plastic, glass, metal, paper, cardboard, rubber, wood, fabric,
leather, ceramic, electronic, mixed, food waste, leaves, stone,
soil, or unknown.

Return this exact JSON structure:

{
  "item": "name of the main object",
  "material": "one material from the list",
  "confidence": 0,
  "contaminated": false,
  "contamination_types": [],
  "condition_note": "",
  "is_food_waste": false,
  "reason": "short explanation of identification"
}

The confidence must be an integer from 0 to 100.

The contaminated field must be true only if there is visible evidence
of food residue, stuck-on food, dirt, mud, oil, grease, dust or grime.

contamination_types should contain only visible contamination types,
such as:
"food stains", "stuck-on food", "dirt", "mud", "oil or grease",
or "dust and grime".

If the object is clean, return contaminated=false,
an empty contamination_types list, and an empty condition_note.

For actual food waste, use material="food waste" and
is_food_waste=true.

For food stains or scraps stuck to another object, use the object's
material and is_food_waste=false.

Return JSON only. No Markdown or additional text.
"""


# ==========================================
# CALL THE LOCAL VISION AI
# ==========================================

def analyse_with_ai(image):
    if image is None:
        return "Please upload an image first."

    try:
        # Convert the uploaded picture into bytes for Ollama.
        image = image.convert("RGB")

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="JPEG", quality=95)
        image_bytes = image_buffer.getvalue()

        # Run vision AI on the computer.
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": AI_INSTRUCTIONS
                },
                {
                    "role": "user",
                    "content": (
                        "Identify the main object and its material. "
                        "Separately inspect visible contamination. "
                        "Follow the JSON instructions exactly."
                    ),
                    "images": [image_bytes]
                }
            ],
            format="json",
            options={
                "temperature": 0,
                "num_ctx": 4096
            }
        )

        answer = response["message"]["content"]
        data = json.loads(answer)

        return create_recycling_report(data)

    except Exception as error:
        return f"""
# Analysis failed

The AI could not analyse this image.

**Possible solutions:**
- Make sure Ollama is running.
- Check that the model is installed.
- Upload a clear image.
- Check your internet connection if you are downloading
  the model for the first time.

**Technical details:** `{str(error)[:500]}`
"""


# ==========================================
# CREATE THE RECYCLING REPORT
# ==========================================

def create_recycling_report(data):

    item = str(data.get("item", "Unknown item")).strip()
    material = str(data.get("material", "unknown")).lower().strip()

    try:
        confidence = int(data.get("confidence", 0))
        confidence = max(0, min(100, confidence))
    except (ValueError, TypeError):
        confidence = 0

    contaminated = data.get("contaminated") is True
    is_food_waste = data.get("is_food_waste") is True

    contamination_types = data.get("contamination_types", [])

    if not isinstance(contamination_types, list):
        contamination_types = []

    condition_note = str(data.get("condition_note", "")).strip()
    reason = str(data.get("reason", "")).strip()

    # Normalise common alternative material names.
    aliases = {
        "aluminium": "metal",
        "aluminum": "metal",
        "steel": "metal",
        "tin": "metal",
        "textile": "fabric",
        "clothing": "fabric",
        "porcelain": "ceramic",
        "stone": "stone",
        "rocks": "stone",
        "garden waste": "leaves",
        "electronic waste": "electronic",
        "e-waste": "electronic",
        "food": "food waste",
        "food scraps": "food waste",
        "food peels": "food waste"
    }

    material = aliases.get(material, material)

    # --------------------------------------
    # ACTUAL FOOD WASTE ONLY
    # --------------------------------------

    if is_food_waste or material == "food waste":

        return """
# ♻️ Recycling Analysis

## Item identified
**Actual food waste**

## Can it go in the usual recycling bin?
**No.**

## Disposal category
**Food waste / organic waste**

## What should you do?
Use a suitable food-waste collection or composting option
if available.

Food stains or food scraps attached to a container are NOT
automatically classified as food waste. They are contamination
on the underlying object.
"""

    # --------------------------------------
    # NON-RECYCLABLE NATURAL MATERIALS
    # --------------------------------------

    if material in ["leaves", "soil", "stone"]:

        if material == "leaves":
            category = "Garden or organic waste"
            advice = (
                "Compost the leaves or use a suitable garden-waste "
                "collection service if available."
            )
        elif material == "soil":
            category = "Soil or specialist disposal"
            advice = (
                "Check local soil disposal or reuse options."
            )
        else:
            category = "Stone or construction waste"
            advice = (
                "Reuse the stone where possible or check local "
                "construction-waste disposal guidance."
            )

        return f"""
# ♻️ Recycling Analysis

## Item identified
**{item}**

## Material
**{material.capitalize()}**

## Household recycling
**Not accepted as ordinary recyclable packaging.**

## Disposal category
**{category}**

## What should you do?
{advice}
"""

    # --------------------------------------
    # UNKNOWN MATERIAL
    # --------------------------------------

    if material == "unknown" or material not in RECYCLING_INFO:

        return f"""
# ♻️ Recycling Analysis

## Item identified
**{item}**

## Material
**Could not be identified reliably**

## Recycling decision
The AI does not have enough information to recommend
a reliable recycling category.

## What should you do?
Upload a clearer picture showing the whole object,
preferably with a plain background and good lighting.

**AI explanation:** {reason}
"""

    # --------------------------------------
    # MATERIAL AND CATEGORY
    # --------------------------------------

    info = RECYCLING_INFO[material]

    material_section = f"""
# ♻️ Recycling Analysis

## Main object
**{item}**

## Material identified
**{material.capitalize()}**

## AI visual matching score
**{confidence}%**

*This is an AI estimate, not a guaranteed probability.*

## Recycling category
**{info["category"]}**
"""

    # --------------------------------------
    # CONTAMINATION SECTION ONLY IF NEEDED
    # --------------------------------------

    contamination_section = ""

    if contaminated:

        clean_instructions = (
            "Remove the visible residue and clean the item if "
            "appropriate. Let it dry before recycling."
        )

        if material in ["paper", "cardboard"]:

            clean_instructions = (
                "Remove loose food scraps. Clean paper or cardboard "
                "must be dry. Greasy, wet or heavily food-soiled "
                "paper and cardboard generally belong in general waste."
            )

        contamination_names = ", ".join(
            str(value) for value in contamination_types
            if str(value).strip()
        )

        if not contamination_names:
            contamination_names = "Possible visible residue"

        contamination_section = f"""
## ⚠️ Visible contamination
**Possible contamination detected**

**What was detected:** {contamination_names}

{condition_note}

## How to handle the contamination
{clean_instructions}

**Important:** The contamination does not change the material
into food waste. The object remains identified as {material}.
"""

    # --------------------------------------
    # RECYCLING DECISION
    # --------------------------------------

    if contaminated and material in ["paper", "cardboard"]:
        decision = (
            "Do not recycle it as paper/cardboard if it is greasy, "
            "wet or heavily food-soiled."
        )
    elif contaminated:
        decision = (
            "Clean it first. Recycle only if the item is accepted "
            "by the local collection programme."
        )
    else:
        decision = (
            "Potentially recyclable through the appropriate "
            "collection route. Check local acceptance rules."
        )

    return f"""
{material_section}

## Recycling decision
**{decision}**

## What should you do?
{info["advice"]}

{contamination_section}

---

*This is an AI estimate based on a photograph. The AI cannot
test for germs or hidden chemicals. Recycling rules vary by location.*
"""


# ==========================================
# FIND A FREE PORT
# ==========================================

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ==========================================
# ADVANCED GRADIO INTERFACE
# ==========================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(120deg, #064e3b, #15803d);
    color: white;
}

#main-title h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

#main-title p {
    font-size: 16px;
}

#analyse-button {
    min-height: 55px;
    font-size: 18px;
    font-weight: bold;
}

#result-panel {
    border-radius: 15px;
}
"""


with gr.Blocks(
    title="AI Recycling Detector",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald"
    ),
    css=CSS
) as app:

    gr.HTML("""
    <div id="main-title">
        <h1>♻️ AI Recycling Detector</h1>
        <p>
            Advanced vision AI for material identification,
            contamination checking and smarter recycling.
        </p>
    </div>
    """)

    gr.Markdown("""
    Upload a photo and press **Analyse Object**.

    The AI identifies the main material first. Food stains,
    dirt and stuck-on scraps are treated as contamination,
    not automatically as food waste.
    """)

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="Upload your object photo",
                height=380
            )

            analyse_button = gr.Button(
                "✨ Analyse Object",
                variant="primary",
                elem_id="analyse-button"
            )

            gr.Markdown("""
            **For better results**

            - Show the whole object.
            - Use clear lighting.
            - Avoid blurry pictures.
            - Try to show one main object.
            """)

        with gr.Column(scale=1):

            result_output = gr.Markdown(
                value="## Your analysis will appear here.",
                elem_id="result-panel"
            )

    analyse_button.click(
        fn=analyse_with_ai,
        inputs=[image_input],
        outputs=[result_output],
        concurrency_limit=1
    )

    gr.Markdown("""
    ---
    **Privacy:** With local Ollama, images are sent to the local
    model on your computer for analysis.

    **Accuracy:** AI results are estimates. Always follow local
    recycling rules.
    """)


# ==========================================
# START THE APPLICATION
# ==========================================

if __name__ == "__main__":

    port = find_free_port()

    print("\nStarting Advanced AI Recycling Detector...")
    print("AI model:", MODEL_NAME)
    print("Keep Ollama and this terminal running.")

    app.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True
    )
