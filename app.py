   
import os
import gradio as gr
from transformers import pipeline

# ==========================================
# AI RECYCLING DETECTOR
# No API key and no Ollama required
# ==========================================

MODEL_NAME = "openai/clip-vit-base-patch32"
ai_model = None

MATERIALS = {
    "plastic": "Plastic",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "cardboard": "Cardboard",
    "rubber": "Rubber",
    "wood": "Wood",
    "fabric": "Fabric or textile",
    "leather": "Leather",
    "ceramic": "Ceramic or porcelain",
    "electronic": "Electronic waste",
    "mixed": "Mixed materials",
    "food": "Actual food waste",
    "leaves": "Leaves or garden waste",
    "stone": "Stone or rocks",
    "soil": "Soil or dirt",
    "unknown": "Unknown material",
}

OBJECT_LABELS = {
    "plastic": "a plastic bottle, container, or plastic packaging",
    "glass": "a glass bottle or glass jar",
    "metal": "a metal can or metal packaging",
    "paper": "a sheet of paper or paper packaging",
    "cardboard": "a cardboard box or cardboard packaging",
    "rubber": "a rubber object or rubber tyre",
    "wood": "a wooden object",
    "fabric": "clothing or a fabric textile",
    "leather": "a leather object or leather bag",
    "ceramic": "a ceramic plate, cup, or porcelain object",
    "electronic": "an electronic device or battery",
    "mixed": "an object made of several different materials",
    "food": "discarded food or food peels",
    "leaves": "leaves or garden waste",
    "stone": "a stone or rock",
    "soil": "soil or loose dirt by itself",
    "unknown": "an object whose material is difficult to identify",
}

CONTAMINATION_LABELS = {
    "clean": "a clean object with no visible stains or dirt",
    "food stains": "an object with visible food stains or sauce",
    "stuck-on food": "an object with visible food scraps stuck to it",
    "oil or grease": "an object visibly covered in oil or grease",
    "mud or dirt": "an object visibly covered in mud or dirt",
    "dust and grime": "an object visibly covered in dust or grime",
}

RECYCLING_INFO = {
    "plastic": (
        "Plastic recycling",
        "Empty and rinse it. Check whether your local programme "
        "accepts this type of plastic."
    ),
    "glass": (
        "Glass bottles and jars",
        "Recycle accepted bottles and jars. Mirrors, drinking "
        "glasses, and ceramics may not belong in glass recycling."
    ),
    "metal": (
        "Metal recycling",
        "Empty and clean accepted metal cans and packaging."
    ),
    "paper": (
        "Paper recycling",
        "Recycle clean, dry paper. Greasy or heavily food-soiled "
        "paper generally belongs in general waste."
    ),
    "cardboard": (
        "Cardboard recycling",
        "Flatten clean, dry cardboard. Greasy or heavily "
        "food-soiled cardboard generally belongs in general waste."
    ),
    "rubber": (
        "Specialist rubber collection",
        "Check a specialist collection service. Rubber tyres "
        "usually do not belong in ordinary recycling bins."
    ),
    "wood": (
        "Wood reuse or specialist collection",
        "Reuse the wood or check local wood collection rules."
    ),
    "fabric": (
        "Textile collection",
        "Keep textiles clean and dry and use a service that "
        "accepts the item."
    ),
    "leather": (
        "Reuse or specialist collection",
        "Reuse the item if possible or check specialist options."
    ),
    "ceramic": (
        "Ceramic disposal",
        "Reuse it if possible. Do not put ceramics in ordinary "
        "glass bottle recycling."
    ),
    "electronic": (
        "Electronic waste (e-waste)",
        "Use an approved e-waste collection point. Batteries "
        "may need separate handling."
    ),
    "mixed": (
        "Mixed-material collection",
        "Separate the parts where possible and check local rules."
    ),
}

NON_RECYCLABLE_NATURAL = {
    "food": (
        "Food waste / organic waste",
        "Use a suitable food-waste collection or composting "
        "option if available."
    ),
    "leaves": (
        "Garden or organic waste",
        "Compost them or use an appropriate garden-waste service."
    ),
    "stone": (
        "Stone or construction waste",
        "Reuse the stone or check local construction-waste rules."
    ),
    "soil": (
        "Soil or specialist disposal",
        "Check local soil reuse and disposal options."
    ),
}


def get_model():
    global ai_model

    if ai_model is None:
        ai_model = pipeline(
            "zero-shot-image-classification",
            model=MODEL_NAME,
            device=-1
        )

    return ai_model


def analyse_image(image):
    if image is None:
        return "## Please upload an image first."

    try:
        model = get_model()
        image = image.convert("RGB")

        # Identify the main object/material.
        material_results = model(
            image,
            candidate_labels=list(OBJECT_LABELS.values()),
            hypothesis_template="This is {}.",
        )

        best_material = material_results[0]
        selected_description = best_material["label"]
        confidence = round(best_material["score"] * 100)

        material = next(
            (
                key for key, description in OBJECT_LABELS.items()
                if description == selected_description
            ),
            "unknown"
        )

        # Do not treat food attached to an object as the object.
        if material == "unknown":
            return """
## ♻️ Recycling Analysis

**Material:** Could not be identified reliably.

Please upload a clearer photo showing the whole object.
"""

        # Check for visible contamination separately.
        contamination_results = model(
            image,
            candidate_labels=list(CONTAMINATION_LABELS.values()),
            hypothesis_template="This is {}.",
        )

        best_contamination = contamination_results[0]
        contamination_description = best_contamination["label"]

        contamination_type = next(
            (
                key for key, description in CONTAMINATION_LABELS.items()
                if description == contamination_description
            ),
            "clean"
        )

        # CLIP scores are relative matching scores, not probabilities.
        # Use a conservative threshold to reduce false positives.
        contamination_score = best_contamination["score"]
        contaminated = (
            contamination_type != "clean"
            and contamination_score >= 0.30
        )

        # Natural materials and actual food are not ordinary
        # recyclable packaging.
        if material in NON_RECYCLABLE_NATURAL:
            category, advice = NON_RECYCLABLE_NATURAL[material]

            return f"""
# ♻️ Recycling Analysis

## Main object
**{MATERIALS[material]}**

## AI visual matching score
**{confidence}%**

## Recycling decision
**Not recyclable as ordinary household packaging.**

## Disposal category
**{category}**

## Advice
{advice}

---
*AI visual matching scores are estimates, not probabilities.*
"""

        # Unknown categories should never receive a false
        # recommendation to recycle.
        if material not in RECYCLING_INFO:
            return """
# ♻️ Recycling Analysis

**Material:** Unknown

**Decision:** Cannot determine safely.

Try a clearer image with good lighting.
"""

        category, advice = RECYCLING_INFO[material]

        if contaminated:
            decision = "NOT RECYCLABLE WHILE CONTAMINATED"
            contamination_report = (
                f"**Possible visible contamination:** "
                f"{contamination_type.capitalize()}.\n\n"
                "Clean the item if appropriate, let it dry, and "
                "analyse it again. This image check cannot confirm "
                "that all contamination has been removed."
            )
        else:
            decision = (
                "Potentially recyclable, subject to local rules"
            )
            contamination_report = (
                "No clear contamination was identified by the AI. "
                "Hidden dirt, germs, or chemicals cannot be detected "
                "reliably from a photograph."
            )

        return f"""
# ♻️ AI Recycling Analysis

## Main object
**{selected_description.capitalize()}**

## Material identified
**{MATERIALS[material]}**

## AI visual matching score
**{confidence}%**

*This score is not a calibrated probability.*

## Contamination check
{contamination_report}

## Recycling decision
**{decision}**

## Recycling category
**{category}**

## What should you do?
{advice}

---
*This is an AI estimate. Recycling acceptance varies by location.*
"""

    except Exception as error:
        return f"""
# Analysis failed

The AI could not analyse this picture.

Check that the model finished downloading and that the
server has enough memory.

**Technical details:** {str(error)[:400]}
"""


# ==========================================
# WEB APP
# ==========================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(120deg, #064e3b, #15803d);
    color: white;
}

#analyse-button {
    min-height: 55px;
    font-size: 18px;
    font-weight: bold;
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
        <p>Identify materials, check visible contamination,
        and find a suitable recycling category.</p>
    </div>
    """)

    gr.Markdown("""
    Upload one object photo and press **Analyse Object**.

    Food stains and dirt are checked separately from the main
    object's material. Contaminated items are marked not recyclable
    until cleaned and checked again.
    """)

    with gr.Row():
        with gr.Column():
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

        with gr.Column():
            result_output = gr.Markdown(
                "## Your analysis will appear here."
            )

    analyse_button.click(
        fn=analyse_image,
        inputs=image_input,
        outputs=result_output,
        concurrency_limit=1
    )

    gr.Markdown("""
    ---
    **Privacy:** This app uses an open-source vision model.
    Uploaded images are processed by the server hosting the app.

    **Accuracy:** Visual AI can make mistakes. Always check
    local recycling rules.
    """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
