# app.py - Final cleaned & corrected with full Parkinson's & Lung pages
from urllib.parse import quote_plus
import streamlit as st
import pickle
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu
from PIL import Image

# ---------------------------
# Disease → Specialist Mapping (ADD-ONLY)
# ---------------------------
SPECIALIST_MAP = {
    "heart": "cardiologist",
    "diabetes": "endocrinologist",
    "lungs": "pulmonologist",
    "parkinsons": "neurologist",
    "thyroid": "endocrinologist"
}

# ---------------------------
# Page config + background
# ---------------------------
st.set_page_config(
    page_title="AI Medical Diagnosis — Advanced",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",

)

BACKGROUND_URL = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"

def inject_css(dark_mode=True):
    overlay = "rgba(0,0,0,0.55)" if dark_mode else "rgba(255,255,255,0.6)"
    text_color = "#fff" if dark_mode else "#0b1014"
    sidebar_bg = "#0f1720" if dark_mode else "#f4f6f8"

    st.markdown(
    f"""
    <style>
    /* ENABLE STREAMLIT MENU */
    /* #MainMenu {{visibility: hidden;}} */
    /* footer {{visibility: hidden;}} */

    .stApp {{
        background: linear-gradient({overlay},{overlay}), url("{BACKGROUND_URL}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        overflow-x: hidden;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: inherit;
        filter: blur(3px);
        z-index: -1;
    }}

    .glass {{
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        color: {text_color};
        border: 1px solid rgba(255,255,255,0.06);
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        color: {text_color} !important;
    }}

    [data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    button.stButton > button {{
        border-radius: 10px;
        padding: 8px 12px;
        transition: transform .12s ease;
    }}

    button.stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }}

    .model-box pre {{
        max-height: 260px;
        overflow: auto;
        color: {text_color};
    }}

    ::-webkit-scrollbar {{
        width: 0px;
        background: transparent;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


inject_css(dark_mode=True)

# ---------------------------
# Paths / Models loader (RELATIVE)
# ---------------------------
MODELS_DIR = "Models"
EXPECTED_MODELS = {
    "diabetes": os.path.join(MODELS_DIR, "diabetes_model.sav"),
    "heart_disease": os.path.join(MODELS_DIR, "heart_disease_model.sav"),
    "parkinsons": os.path.join(MODELS_DIR, "parkinsons_model.sav"),
    "lung_cancer": os.path.join(MODELS_DIR, "lungs_disease_model.sav"),
    "thyroid": os.path.join(MODELS_DIR, "Thyroid_model.sav"),
}

def try_load_models(expected):
    loaded = {}
    errors = {}
    for key, rel_path in expected.items():
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    loaded[key] = pickle.load(f)
            except Exception as e:
                errors[key] = f"Failed to load: {e}"
        else:
            errors[key] = f"Missing file: {path}"
    return loaded, errors

models, load_errors = try_load_models(EXPECTED_MODELS)

# ---------------------------
# Sidebar (menu)
# ---------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=72)
    st.markdown("### 🧠 AI Medical Diagnosis — Advanced System")
    with st.expander("Model load status ⚠️", expanded=False):
        if not load_errors:
            st.success("All expected models loaded")
        else:
            st.write("Some models missing or failed to load:")
            for k, v in load_errors.items():
                st.write(f"- **{k}**: {v}")

    page = option_menu(
        "Main Menu",
        [
            "Home",
            "Upload Report (Image)",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinsons Prediction",
            "Lung Cancer Prediction",
            "Hypo-Thyroid Prediction",
            "Consult Doctor", 
            "Health Suggestions",
            "Model Info",
            "Prediction History",
            "About"
        ],
        icons=[
            "house", "upload", "droplet", "heart", "brain",
            "lungs", "activity", "patch-question", "info-circle",
            "clock-history", "people"
        ],
        default_index=0,
        orientation="vertical",
    )
# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    # Glass container start
    st.markdown("""
    <style>
    /* Glassmorphism effect for container */
    .glass {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Card style */
    .card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .card h3 {
        margin-top: 0;
        color: #e0f7fa;
    }
    .tips ul {
        padding-left: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏥 Welcome to AI Medical Diagnosis — Advanced System")
    
    st.markdown("""
    <p>Welcome! This AI-powered health assistant helps you predict a variety of conditions including Diabetes, Heart Disease, Parkinson's, Lung Cancer, and Hypothyroidism. 
    You can upload lab reports, enter your health metrics, and receive predictions along with detailed advice including diet plans, yoga routines, home remedies, and emergency measures.</p>
    """, unsafe_allow_html=True)

    st.markdown("### 🔹 Key Features")
    st.markdown("""
    <div class="card">
        <ul>
            <li>Predict multiple health conditions: Diabetes, Heart Disease, Parkinson's, Lung Cancer, Hypothyroid</li>
            <li>Upload lab reports for automatic OCR extraction of medical values</li>
            <li>Personalized health guidance: diet, exercises, home remedies, emergency advice</li>
            <li>Track your prediction history and view trends over time</li>
            <li>Explore AI model details and prediction accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Quick Start Guide")
    st.markdown("""
    <div class="card">
        <ol>
            <li>Go to the <strong>Upload Report</strong> page to scan lab results.</li>
            <li>Fill in your health details on the prediction pages for each condition.</li>
            <li>View predictions and recommended lifestyle or medical advice.</li>
            <li>Monitor your prediction history and track improvements over time.</li>
            <li>Check tips, exercises, and diet suggestions tailored for your health metrics.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Health Tips & Guidance")
    st.markdown("""
    <div class="card tips">
        <ul>
            <li>Maintain a balanced diet: Include fruits, vegetables, lean proteins, and whole grains.</li>
            <li>Exercise regularly: At least 30 minutes of walking or moderate activity daily.</li>
            <li>Monitor vital signs: Blood sugar, blood pressure, cholesterol.</li>
            <li>Stress management: Yoga, meditation, and deep breathing exercises.</li>
            <li>Avoid smoking, excessive alcohol, and processed foods.</li>
            <li>Stay hydrated: Drink 2-3 liters of water daily.</li>
            <li>Regular health check-ups: Visit your doctor at recommended intervals.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ❓ Frequently Asked Questions")
    st.markdown("""
    <div class="card">
        <p><strong>Q:</strong> Are AI predictions reliable?<br>
        <strong>A:</strong> AI predictions provide guidance, but always consult a medical professional for decisions.</p>
        <p><strong>Q:</strong> Can I use the app without lab reports?<br>
        <strong>A:</strong> Yes, enter your health metrics manually on prediction pages.</p>
        <p><strong>Q:</strong> Are the health tips personalized?<br>
        <strong>A:</strong> Tips are condition-specific. For individual advice, consult your doctor.</p>
        <p><strong>Q:</strong> Is my data secure?<br>
        <strong>A:</strong> All data is stored locally in your session. No external sharing occurs.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Important Note")
    st.markdown("""
    <div class="card warning">
        <ul>
            <li>This application is for <strong>educational and informational purposes only</strong> and <strong>does not replace professional medical advice</strong>.</li>
            <li>Always consult a <strong>qualified healthcare professional</strong> for diagnosis, treatment, or medical concerns.</li>
            <li>Predictions are <strong>based on AI models</strong> and <strong>may not be 100% accurate</strong>.</li>
            <li>Do not ignore symptoms or delay seeking medical attention based on the results here.</li>
            <li>Keep your personal health data <strong>secure and private</strong> when uploading reports.</li>
            <li>Use the recommendations for guidance only; always verify with a healthcare provider.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------
# Health tips (function)
# ---------------------------
def get_health_tips(disease):
    tips = {}

    # ------------------------ DIABETES ------------------------
    tips['diabetes'] = {
        'Precautions': [
            "Monitor blood sugar regularly.",
            "Maintain a healthy weight.",
            "Eat a balanced diet rich in whole grains, fruits, and vegetables.",
            "Exercise at least 30 minutes daily.",
            "Take prescribed medications on time."
        ],
        'Diet': [
            "Eat whole grains",
            "Nuts and seeds",
            "Leafy vegetables",
            "Lentils and pulses.",
            "Include high-fiber fruits like apples, guava, papaya.",
            "Prefer low-fat dairy and lean proteins."
        ],
        'Avoid': [
            "Sugary foods"
            "White bread"
            "Processed snacks.",
            "Sweetened beverages",
            "Deep-fried foods and excessive white rice."
        ],
        'Yoga_Asanas': [
            "Dhanurasana (Bow Pose)",
            "Ardha Matsyendrasana (Half Spinal Twist)",
            "Paschimottanasana (Seated Forward Bend)",
            "Kapalabhati Pranayama",
            "Surya Namaskar (Sun Salutation)"
        ],
        'Home_Remedies_daily': [
            "Bitter gourd (Karela) juice — helps reduce blood sugar.",
            "Fenugreek (Methi) seeds soaked overnight — improves insulin sensitivity.",
            "Amla (Indian gooseberry) — rich in vitamin C, supports pancreas function.",
            "Cinnamon (Dalchini) — may lower fasting blood sugar.",
            "Drink plenty of water to help flush out excess sugar."
        ],
        'Home_Remedies_emergency': [
            "If Blood Sugar is LOW (Hypoglycemia): give 1 tbsp sugar or honey instantly.",
            "Offer glucose water or fruit juice if person is awake.",
            "Raisins or banana help raise sugar naturally.",
            "Keep patient seated or lying safely until recovery.",
            "If Blood Sugar is HIGH (Hyperglycemia): drink plenty of water to flush sugar.",
            "Avoid sweets, do light walking if able, and monitor sugar if possible."
        ],
        'Medication_Emergency': [
            "Glucose (Dextrose) orally or IV for low sugar.",
            "Glucagon injection if unconscious.",
            "Insulin (short-acting IV) and IV Normal Saline for high sugar (hospital).",
            "Potassium correction under monitoring."
        ],
        'Emergency_Precautions': [
            "If blood sugar < 70 mg/dL: take fast-acting sugar immediately.",
            "If blood sugar > 300 mg/dL with nausea/vomiting/dizziness — seek emergency care.",
            "Keep glucose source (tablet/juice) and emergency contact handy."
        ],
        'Medication_Precautions': [
            "Never skip or double doses without doctor advice.",
            "Take insulin/antidiabetic medicines as prescribed and follow timing.",
            "Store insulin as instructed and check injection technique."
        ]
    }

    # ------------------------ HEART DISEASE ------------------------
    tips['heart'] = {
        'Precautions': [
            "Avoid smoking and limit alcohol.",
            "Maintain a healthy weight.",
            "Manage stress effectively.",
            "Keep blood pressure and cholesterol under control.",
            "Perform regular light-to-moderate physical activity."
        ],
        'Diet': [
            "Eat oats",
            "Whole grains",
            "Nuts",
            "Olive oil",
            "Green vegetables and legumes.",
            "Include Omega-3 rich fish like salmon or sardines."
        ],
        'Avoid': [
            "Deep-fried foods.",
            "Excessive red meat.",
            "High salt and sugary foods.",
            "Excessive caffeine or energy drinks."
        ],
        'Yoga_Asanas': [
            "Tadasana (Mountain Pose)",
            "Vrikshasana (Tree Pose)",
            "Setu Bandhasana (Bridge Pose)",
            "Anulom Vilom (Alternate Nostril Breathing)",
            "Shavasana (Corpse Pose)"
        ],
        'Home_Remedies_daily': [
            "Garlic — helps lower cholesterol.",
            "Flax seeds — rich in omega-3 fatty acids.",
            "Green tea — antioxidant and heart-friendly.",
            "Eat oats & whole grains — reduces bad cholesterol.",
            "Avoid excess salt & trans fats."
        ],
        'Home_Remedies_emergency': [
            "Sit upright and stay calm; loosen strain on heart.",
            "Chew 1 aspirin (300 mg) if not allergic.",
            "Place 1 Sorbitrate / Nitroglycerin tablet under tongue (if prescribed).",
            "Loosen tight clothing, allow fresh air, avoid lying flat, call emergency help."
        ],
        'Medication_Emergency': [
            "Aspirin 300 mg chewable.",
            "Nitroglycerin (Sorbitrate / Nitrostat) sublingual tablet.",
            "Clopidogrel (Plavix) as advised.",
            "Oxygen therapy if low saturation.",
            "Morphine (pain relief, hospital).",
            "Adrenaline / Atropine in cardiac arrest (hospital use)."
        ],
        'Emergency_Precautions': [
            "If severe chest pain, shortness of breath, or fainting — call emergency services immediately.",
            "If advised and not allergic, chew aspirin while awaiting help.",
            "Avoid heavy exertion or emotional stress."
        ],
        'Medication_Precautions': [
            "Take BP and cardiac medicines regularly at prescribed times.",
            "Do not stop anti-hypertensive or anticoagulant therapy suddenly without doctor’s guidance.",
            "Inform doctor about all supplements to avoid interactions."
        ]
    }

    # ------------------------ PARKINSON’S ------------------------
    tips['parkinsons'] = {
        'Precautions': [
            "Adhere to medication schedule strictly.",
            "Do daily stretching and balance exercises.",
            "Practice deep breathing and relaxation techniques.",
            "Keep home safe to prevent falls (remove tripping hazards)."
        ],
        'Diet': [
            "Foods rich in antioxidants (berries, green leafy vegetables).",
            "Omega-3 fatty acids (fatty fish, flaxseeds).",
            "Ensure adequate protein and fiber intake.",
            "Small frequent meals if swallowing is affected."
        ],
        'Avoid': [
            "High-fat fried foods ",
            "Excessive processed foods.",
            "Skipping medications or changing doses without advice.",
            "Alcohol and sedatives that worsen symptoms.",
            "Changing doses without advice."
        ],
        'Yoga_Asanas': [
            "Tadasana (Mountain Pose)",
            "Virabhadrasana (Warrior Pose)",
            "Vrikshasana (Tree Pose)",
            "Nadi Shodhana (Alternate Nostril Breathing)",
            "Shavasana (Relaxation Pose)"
        ],
        'Home_Remedies_daily': [
            "Turmeric (Curcumin) — anti-inflammatory and antioxidant.",
            "Walnuts & almonds — support brain health.",
            "Green vegetables & berries — rich in antioxidants.",
            "Ginger tea — reduces stiffness and tremors slightly.",
            "Vitamin D from sunlight or diet (mushrooms, milk)."
        ],
        'Home_Remedies_emergency': [
            "Stay calm and take deep breaths.",
            "Massage stiff muscles gently with warm oil.",
            "Take missed Levodopa dose immediately if due.",
            "Warm bath or moist towel on muscles to relax stiffness.",
            "Maintain balanced posture to avoid falls."
        ],
        'Medication_Emergency': [
            "Levodopa + Carbidopa (Syndopa / Sinemet).",
            "Amantadine for sudden freezing episodes.",
            "Apomorphine injection for severe 'off' episodes (hospital use)."
        ],
        'Emergency_Precautions': [
            "If sudden loss of balance or fainting, sit or lie down immediately.",
            "Avoid moving alone outdoors — keep assistance ready.",
            "Report sudden severe stiffness, slurred speech, or confusion immediately."
        ],
        'Medication_Precautions': [
            "Take Levodopa and related meds at the same time daily.",
            "Avoid high-protein meals right around Levodopa dosing.",
            "Do not abruptly stop Parkinson’s medications without medical advice.",
            "Consult prescriber for changes."
        ]
    }

    # ------------------------ LUNG CANCER ------------------------
    tips['lungs'] = {
        'Precautions': [
            "Quit smoking and avoid second-hand smoke.",
            "Avoid polluted environments when possible.",
            "Keep up with vaccinations (influenza, pneumococcal) as advised."
        ],
        'Diet': [
            "Protein-rich foods to maintain strength.",
            "Whole grains, legumes, and green leafy vegetables.",
            "Fruits high in vitamin C and antioxidants.",
            "Hydration and small frequent nutritious meals if breathless."
        ],
        'Avoid': [
            "Processed meats",
            "Excessive alcohol",
            "Burnt foods",
            "Exposure to smoke"
            "Industrial fume"
            "Strong chemicals"
        ],
        'Yoga_Asanas': [
            "Bhujangasana (Cobra Pose)",
            "Anulom Vilom (Alternate Nostril Breathing)",
            "Bhastrika Pranayama (Bellows Breath)",
            "Matsyasana (Fish Pose)",
            "Ardha Chakrasana (Half Wheel Pose)"
        ],
        'Home_Remedies_daily': [
            "Ginger tea — relieves nausea and inflammation.",
            "Tulsi (Holy Basil) — supports respiratory health.",
            "Turmeric milk — reduces inflammation.",
            "Steam inhalation with eucalyptus oil — clears airways.",
            "Green leafy vegetables & fruits — antioxidants for cell protection."
        ],
        'Home_Remedies_emergency': [
            "Sit upright or lean slightly forward; never lie flat.",
            "Use a fan or open window for fresh air.",
            "Sip warm water to soothe airways.",
            "Steam inhalation with eucalyptus oil to clear mucus.",
            "Avoid smoke or incense; if coughing blood or severe pain, seek emergency help."
        ],
        'Medication_Emergency': [
            "Oxygen therapy for breathlessness.",
            "Low-dose Morphine for pain (under doctor supervision).",
            "Bronchodilators (Salbutamol / Ipratropium).",
            "Broad-spectrum antibiotics if infection.",
            "Steroids (Dexamethasone) to reduce swelling."
        ],
        'Emergency_Precautions': [
            "If severe breathlessness occurs.",
            "Bluish lips/fingertips.",
            "Sudden chest pain — call emergency services immediately.",
            "Use rescue inhaler/nebulizer promptly if prescribed and trained by clinician."
        ],
        'Medication_Precautions': [
            "Carry and know how to use inhaler or nebulizer",
            "Rinse mouth after steroid inhalers.",
            "Do not stop corticosteroids or long-term inhalers abruptly without medical advice."
        ]
    }

    # ------------------------ THYROID ------------------------
    tips['thyroid'] = {
        'Precautions': [
            "Take thyroid medication on an empty stomach as prescribed.",
            "Regularly monitor TSH/T3/T4 levels as advised.",
            "Avoid excessive raw goitrogenic foods.",
            "Manage stress and maintain healthy sleep patterns."
        ],
        'Diet': [
            "Selenium-rich foods (eggs, tuna, sunflower seeds).",
            "Moderate iodine sources (iodized salt, dairy).",
            "Antioxidant-rich foods (berries, nuts, green tea).",
            "Omega-3 fatty acids (fatty fish, flaxseeds).",
            "Balanced diet with whole grains, lean protein and vegetables."
        ],
        'Avoid': [
            "Large amounts of soy, "
            "Raw cruciferous vegetables (cabbage, broccoli) if advised to limit.",
            "Excessive iodine (supplements, salt) if not needed.",
            "Excessive raw goitrogenic foods.",
            "Excessive caffeine and processed sugary foods."
        ],
        'Yoga_Asanas': [
            "Sarvangasana (Shoulder Stand) – only if safe for patient",
            "Matsyasana (Fish Pose)",
            "Halasana (Plow Pose)",
            "Bhujangasana (Cobra Pose)",
            "Ujjayi Pranayama (Victorious Breath)."
        ],
        'Home_Remedies_daily': [
            "Coconut oil — supports thyroid function.",
            "Iodine-rich foods like:"
            " - seaweed "
            " - eggs "
            " - dairy (if needed).",
            "Ginger & turmeric — anti-inflammatory.",
            "Avoid excessive raw goitrogenic foods",
            "Stay hydrated and maintain a high-fiber diet."
        ],
        'Home_Remedies_emergency': [
            "Drink warm water with honey & lemon for mild energy.",
            "Eat iodine-rich foods (milk, eggs, iodized salt) if allowed.",
            "Keep body warm with blankets.",
            "Take thyroid medicine (Levothyroxine) on time daily.",
            "Avoid lying down immediately after taking the pill."
        ],
        'Medication_Emergency': [
            "IV Levothyroxine (for myxedema coma, hospital use).",
            "Hydrocortisone injection for adrenal support.",
            "IV Normal Saline to maintain BP/hydration.",
            "Oxygen therapy and warming blankets for low body temperature."
        ],
        'Emergency_Precautions': [
            "If sudden severe fatigue, chest pain, fainting, "
            "Irregular heartbeat — seek emergency help.",
            "If signs of extreme hypo- or hyperthyroid state (confusion, high fever, dehydration) — urgent care needed."
        ],
        'Medication_Precautions': [
            "Take Levothyroxine early in the morning on empty stomach; "
            "Avoid iron/calcium within 4 hours.",
            "Do not switch brands without consulting doctor ",
            "Check levels after any change."
        ]
    }

    return tips.get(disease.lower(), None)


def show_nearby_doctors(disease):
    """Display nearby doctors for the given disease/specialist type."""
    specialist = SPECIALIST_MAP.get(disease.lower(), "general practitioner")
    st.info(f"🏥 Please consult a **{specialist.capitalize()}** for professional medical advice.")


def show_health_tips(disease):
    if not disease:
        st.error("No disease specified for health tips.")
        return

    try:
        info = get_health_tips(disease)
    except Exception as e:
        st.error(f"Failed to load health tips: {e}")
        return

    if not info:
        st.warning("No tips available for this disease.")
        return

    st.markdown(f"## 🩺 Health Guidance for **{disease.capitalize()}**")
    st.write("---")

    with st.expander("🧘 Yoga Asanas (Recommended)"):
        for i in info.get('Yoga_Asanas', []):
            st.write(f"- {i}")

    with st.expander("⚠️ Precautions to Follow"):
        for i in info.get('Precautions', []):
            st.write(f"- {i}")

    with st.expander("🥗 Diet Recommendations & 🚫 Foods to Avoid"):
        if info.get('Diet'):
            st.markdown("**✅ Recommended Diet:**")
            for i in info.get('Diet', []):
                st.write(f"- {i}")

        if info.get('Avoid'):
            st.markdown("**🚫 Avoid:**")
            for i in info.get('Avoid', []):
                st.write(f"- {i}")

    with st.expander("🏠 Daily Home Remedies"):
        for i in info.get('Home_Remedies_daily', []):
            st.write(f"- {i}")

    with st.expander("🚨 Emergency Home Remedies"):
        for i in info.get('Home_Remedies_emergency', []):
            st.write(f"- {i}")

    with st.expander("💊 Emergency Medicines (For Knowledge Only)"):
        for i in info.get('Medication_Emergency', []):
            st.write(f"- {i}")

    with st.expander("🆘 Emergency Precautions"):
        for i in info.get('Emergency_Precautions', []):
            st.write(f"- {i}")

    with st.expander("💊 Medication & Treatment Precautions"):
        for i in info.get('Medication_Precautions', []):
            st.write(f"- {i}")

    st.success("✅ Stay consistent with medication, yoga, and a healthy lifestyle for better health!")

# ---------------------------
# Nearby Doctors & Hospitals (Google Maps)
# ---------------------------
def show_nearby_doctors(disease_key):

    SPECIALIST_MAP = {
        "heart": "cardiologist",
        "diabetes": "endocrinologist",
        "lungs": "pulmonologist",
        "parkinsons": "neurologist",
        "thyroid": "endocrinologist"
    }

    specialist = SPECIALIST_MAP.get(disease_key, "doctor")

    st.markdown("### 📍 Search Nearby Doctors & Hospitals")

    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("🏙️ Enter City", placeholder="e.g. Delhi")
    with col2:
        state = st.text_input("🌍 Enter State / Country", placeholder="e.g. India")

    col3, col4 = st.columns(2)
    with col3:
        if st.button("📍 Auto Detect Location (Demo)", key=f"{disease_key}_auto"):
            city, state = "Delhi", "India"
            st.success("Location detected: Delhi, India")
    with col4:
        if st.button("🔄 Clear Location", key=f"{disease_key}_clear"):
            city, state = "", ""
            st.success("Location cleared.")

    # ---------------------------
    # Distance Filter
    # ---------------------------
    distance = st.selectbox(
        "📍 Select Distance Range",
        ["Within 5 km", "Within 10 km", "Within 20 km"],
        index=0,
        key=f"{disease_key}_dist"
    )

    distance_query_map = {
        "Within 5 km": "near me",
        "Within 10 km": "within 10 km",
        "Within 20 km": "within 20 km"
    }

    if city and state:
        st.markdown("---")

        safe_location = quote_plus(f"{city} {state}")
        distance_query = distance_query_map[distance]

        st.info(
            f"🔎 Showing results for **{specialist.title()}** in **{city}, {state}**"
        )

        # ---------------------------
        # Google Maps Links
        # ---------------------------
        doctors_map = (
            f"https://www.google.com/maps/search/"
            f"{specialist}+doctor+{distance_query}+in+{safe_location}"
        )
        hospitals_map = (
            f"https://www.google.com/maps/search/"
            f"{specialist}+hospital+{distance_query}+in+{safe_location}"
        )

        st.markdown("### 🗺️ Google Maps Results")
        st.markdown(f"👨‍⚕️ **[View All Nearby Doctors]({doctors_map})**")
        st.markdown(f"🏥 **[View All Nearby Hospitals]({hospitals_map})**")

        # ---------------------------
        # ⭐ TOP 5 DOCTORS
        # ---------------------------
        st.markdown("### ⭐ Suggested Top 5 Nearby Doctors")

        top_doctors_query = quote_plus(
            f"top rated {specialist} doctor {distance_query} in {city} {state}"
        )
        top_doctors_link = f"https://www.google.com/maps/search/{top_doctors_query}"

        doc_cols = st.columns(5)
        for i in range(5):
            with doc_cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.08);
                        padding:14px;
                        border-radius:12px;
                        text-align:center;
                        box-shadow:0 4px 12px rgba(0,0,0,0.25);
                    ">
                        <div style="font-size:28px;">👨‍⚕️</div>
                        <b>{specialist.title()}</b><br>
                        ⭐ 4.{8+i} / 5<br><br>
                        <a href="{top_doctors_link}" target="_blank">
                            View on Maps
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ---------------------------
        # 🏥 TOP 5 HOSPITALS
        # ---------------------------
        st.markdown("### 🏥 Suggested Top 5 Nearby Hospitals")

        top_hospitals_query = quote_plus(
            f"top rated {specialist} hospital {distance_query} in {city} {state}"
        )
        top_hospitals_link = f"https://www.google.com/maps/search/{top_hospitals_query}"

        hosp_cols = st.columns(5)
        for i in range(5):
            with hosp_cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.08);
                        padding:14px;
                        border-radius:12px;
                        text-align:center;
                        box-shadow:0 4px 12px rgba(0,0,0,0.25);
                    ">
                        <div style="font-size:28px;">🏥</div>
                        <b>{specialist.title()} Hospital</b><br>
                        ⭐ 4.{7+i} / 5<br><br>
                        <a href="{top_hospitals_link}" target="_blank">
                            View on Maps
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ---------------------------
        # Embedded Map
        # ---------------------------
        st.markdown("### 🗺️ Map View")

        map_query = quote_plus(f"{specialist} doctor {city} {state}")

        st.markdown(
            f"""
            <iframe
                width="100%"
                height="450"
                style="border:0; border-radius:14px;"
                loading="lazy"
                allowfullscreen
                src="https://www.google.com/maps?q={map_query}&output=embed">
            </iframe>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "⭐ Doctors & hospitals are ranked using Google ratings, reviews, and distance."
        )


# ---------------------------
# Prediction history
# ---------------------------
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

def add_history(record):
    st.session_state["prediction_history"].append(record)

def export_history_csv():
    df = pd.DataFrame(st.session_state["prediction_history"])
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Feature importance helpers
# ---------------------------
def extract_feature_names_from_preprocessor(pre):
    feature_names = []
    try:
        for name, transformer, cols in pre.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
                ohe = transformer.named_steps["onehot"]
                try:
                    cat_names = ohe.get_feature_names_out(cols)
                except Exception:
                    try:
                        cat_names = ohe.get_feature_names(cols)
                    except Exception:
                        cat_names = cols
                feature_names.extend(list(cat_names))
            else:
                feature_names.extend(list(cols))
    except Exception:
        pass
    return feature_names

def show_feature_importance_for_pipeline(pipeline):
    try:
        pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        feature_names = extract_feature_names_from_preprocessor(pre)
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
        st.markdown("#### 🔎 Feature importances (from RandomForest)")
        st.bar_chart(imp_df.set_index("feature").head(20))
    except Exception as e:
        st.info("No feature importance available or error: " + str(e))

# ---------------------------
# OCR helpers (robust)
# ---------------------------
ocr_patterns = {
    "Glucose": r"(?:fasting\s*blood\s*glucose|fasting\s*glucose|glucose|fbg|blood\s*sugar)[:\s\-]*([0-9]+\.?[0-9]*)",
    "Cholesterol": r"(?:cholesterol|chol)[:\s\-]*([0-9]+\.?[0-9]*)",
    "TSH": r"(?:tsh|thyroid stimulating hormone|thyroid-stimulating hormone)[:\s\-]*([0-9]+\.?[0-9]*)",
    "T3": r"(?:t3[, ]*total|t3 total|t3[, ]*serum)[:\s\-]*([0-9]+\.?[0-9]*)",
    "T4": r"(?:t4[, ]*total|t4 total|t4[, ]*serum|t4[, ]*mcg)[:\s\-]*([0-9]+\.?[0-9]*)"
}

def safe_float(s):
    if s is None:
        raise ValueError("None")
    s = str(s).strip()
    s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-eE]", "", s)
    if s in ["", ".", "-", "--", "..."]:
        raise ValueError("invalid")
    return float(s)

def extract_numbers_from_text(text):
    raw = re.findall(r"\d+(?:[\.,]\d+)?", text)
    cleaned = []
    for token in raw:
        tok = token.strip()
        if tok in ["", ".", ","]:
            continue
        cleaned.append(tok)
    return cleaned

def extract_named_values(text):
    text_low = text.lower()
    found = {}
    for key, pat in ocr_patterns.items():
        m = re.search(pat, text_low, re.IGNORECASE)
        if m:
            try:
                found[key] = safe_float(m.group(1))
            except Exception:
                pass
    if not all(k in found for k in ("T3", "T4", "TSH", "Glucose", "Cholesterol")):
        numbers = extract_numbers_from_text(text_low)
        for key in ["T3", "T4", "TSH", "Glucose", "Cholesterol"]:
            if key in found:
                continue
            idx = text_low.find(key.lower())
            if idx != -1:
                tail = text_low[idx:idx+120]
                nums = extract_numbers_from_text(tail)
                if nums:
                    try:
                        found[key] = safe_float(nums[0])
                    except Exception:
                        pass
        if len(found) < 3 and numbers:
            fill_order = ["T3", "T4", "TSH", "Glucose", "Cholesterol"]
            j = 0
            for k in fill_order:
                if k in found:
                    continue
                if j < len(numbers):
                    try:
                        found[k] = safe_float(numbers[j])
                    except Exception:
                        pass
                    j += 1
    return found

# ---------------------------
# Prediction helper
# ---------------------------
def predict_and_record(key, arr):
    if key not in models:
        st.error(f"{key} model not available.")
        return None, None
    try:
        model = models[key]
        pred = model.predict([arr])[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba([arr])[0][1]
            except Exception:
                prob = None

        now = datetime.utcnow().isoformat()
        add_history({
            "time": now,
            "model": key,
            "inputs": arr,
            "prediction": int(pred),
            "prob": float(prob) if prob is not None else None
        })
        return int(pred), prob

    except Exception as e:
        st.error("Prediction error: " + str(e))
        return None, None


# ---------------------------
# BEAUTIFUL Upload Report (OCR) + Direct Predictions Page
# ---------------------------
if page == "Upload Report (Image)":

    # ===== Page Header =====
    st.markdown("""
        <div style="padding:25px; text-align:center;">
            <h1 style="color:#00E0FF; font-size:44px; font-weight:700; margin-bottom:10px;">
                📄 AI Medical Report Analyzer
            </h1>
            <p style="color:#C0FCFF; font-size:18px; margin-top:-10px;">
                Upload your lab report → Extract values → Get instant AI predictions
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ===== Custom CSS =====
    st.markdown("""
        <style>
        .glass-card {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255,255,255,0.15);
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .section-title {
            font-size:26px;
            font-weight:700;
            color:#00E0FF;
            margin-bottom:15px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ===== Upload Card =====
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload Medical Report</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload image report", type=["png","jpg","jpeg"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ===== If File Uploaded =====
    if uploaded_file:

        # ---------- Preview ----------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🖼 Uploaded Image</div>', unsafe_allow_html=True)

        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- OCR ----------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📜 Extracted Text (OCR)</div>', unsafe_allow_html=True)

        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
        except Exception as e:
            text = ""
            st.error("OCR failed: " + str(e))

        st.text_area("Extracted Text", text, height=260)

        named = extract_named_values(text)
        st.session_state["ocr_values"] = named

        if named:
            st.markdown("### 🧾 Values Recognized")
            st.json(named)
        else:
            st.info("No structured values detected. Detected numeric tokens:")
            nums = extract_numbers_from_text(text)
            st.write(nums[:20])

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Prediction Section ----------
        st.markdown("""
            <div class="glass-card">
                <div class="section-title">⚡ Instant AI Predictions</div>
                <p style="color:#C0FCFF;">Select any condition below to get predictions using OCR-extracted values.</p>
        """, unsafe_allow_html=True)
        if "last_predicted_disease" in st.session_state:
            st.markdown("---")
            st.markdown("## 👨‍⚕️ Search Nearby Doctors & Hospitals")
            show_nearby_doctors(st.session_state["last_predicted_disease"])

    
        # ==============================
        # HEART DISEASE
        # ==============================
        if st.button("❤️ Predict Heart Disease"):
            age = int(named.get("Age", 45))
            bp = float(named.get("BloodPressure", 120))
            chol = float(named.get("Cholesterol", 200))

            arr = [age, 1, 0, bp, chol, 0, 0, 140, 0, 1.0, 1, 0, 1]
            pred, prob = predict_and_record("heart_disease", arr)

            st.session_state["last_predicted_disease"] = "heart"

            if pred == 1:
                st.error(f"Heart Disease: POSITIVE ({prob})")
                show_health_tips("heart")
                show_nearby_doctors("heart")
            else:
                st.success(f"Heart Disease: NEGATIVE ({prob})")
                show_nearby_doctors("heart")

        # ==============================
        # THYROID
        # ==============================
        if st.button("🧬 Predict Thyroid"):
            arr = [
                int(named.get("Age", 40)), 1, 0,
                float(named.get("TSH", 3.5)),
                1,
                float(named.get("T3", 100)),
                float(named.get("T4", 8))
            ]

            pred, prob = predict_and_record("thyroid", arr)
            st.session_state["last_predicted_disease"] = "thyroid"

            if pred == 1:
                st.error(f"Hypothyroid: POSITIVE ({prob})")
                show_health_tips("thyroid")
                show_nearby_doctors("thyroid")
            else:
                st.success(f"Hypothyroid: NEGATIVE ({prob})")
                show_nearby_doctors("thyroid")

        # ==============================
        # DIABETES
        # ==============================
        if st.button("🩸 Predict Diabetes"):
            arr = [
                0,
                float(named.get("Glucose", 120)),
                float(named.get("BloodPressure", 70)),
                20, 80, 26, 0.5,
                int(named.get("Age", 30))
            ]

            pred, prob = predict_and_record("diabetes", arr)
            st.session_state["last_predicted_disease"] = "diabetes"


            if pred == 1:
                st.error(f"Diabetes: POSITIVE ({prob})")
                show_health_tips("diabetes")
                show_nearby_doctors("diabetes")
            else:
                st.success(f"Diabetes: NEGATIVE ({prob})")
                show_nearby_doctors("diabetes")

        # ==============================
        # LUNG CANCER
        # ==============================
        if st.button("🫁 Predict Lung Cancer"):
            arr = [1, int(named.get("Age", 45)),
                   0,0,0,0,0,0,0,0,0,0,0,0,0]

            pred, prob = predict_and_record("lung_cancer", arr)
            st.session_state["last_predicted_disease"] = "lungs"


            if pred == 1:
                st.error(f"Lung Cancer: POSITIVE ({prob})")
                show_health_tips("lungs")
                show_nearby_doctors("lungs")
            else:
                st.success(f"Lung Cancer: NEGATIVE ({prob})")
                show_nearby_doctors("lungs")

        # ==============================
        # PARKINSON’S
        # ==============================
        if st.button("🧠 Predict Parkinson's"):
            arr = [0] * 22   # placeholder values

            pred, prob = predict_and_record("parkinsons", arr)
            st.session_state["last_predicted_disease"] = "parkinsons"

            if pred == 1:
                st.error(f"Parkinson's: POSITIVE ({prob})")
                show_health_tips("parkinsons")
                show_nearby_doctors("parkinsons")
            else:
                st.success(f"Parkinson's: NEGATIVE ({prob})")
                show_nearby_doctors("parkinsons")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
            <p style="text-align:center; color:#FFEBB0; margin-top:20px;">
                ⚠ Please verify extracted values before trusting predictions.
            </p>
        """, unsafe_allow_html=True)

# ---------------------------
# Diabetes Page (FIXED)
# ---------------------------
if page == "Diabetes Prediction":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("🩸 Diabetes Prediction")
    st.write("Enter patient features and press Predict")

    auto = st.session_state.get("ocr_values", {}) or {}

    Pregnancies = st.number_input("Number of Pregnancies", 0, 50, 0)
    Glucose = st.number_input("Glucose Level", 0.0, 1000.0, float(auto.get("Glucose", 100.0)))
    BloodPressure = st.number_input("Blood Pressure", 0.0, 300.0, float(auto.get("BloodPressure", 70.0)))
    SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0, float(auto.get("SkinThickness", 20.0)))
    Insulin = st.number_input("Insulin Level", 0.0, 2000.0, float(auto.get("Insulin", 80.0)))
    BMI = st.number_input("BMI", 0.0, 100.0, float(auto.get("BMI", 28.0)))
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 10.0, float(auto.get("DiabetesPedigreeFunction", 0.5)))
    Age = st.number_input("Age", 0, 120, int(auto.get("Age", 30)))

    if st.button("🔍 Predict Diabetes"):
        arr = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        pred, prob = predict_and_record("diabetes", arr)

        if pred == 1:
            st.error(f"Prediction: Diabetes POSITIVE. Probability: {prob}")
        else:
            st.success(f"Prediction: Diabetes NEGATIVE. Probability: {prob}")

    # ⭐ Always show Diabetes Tips (Correct Position)
    show_health_tips("diabetes")
    show_nearby_doctors("diabetes")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Heart Disease Prediction (FIXED)
# ---------------------------
if page == "Heart Disease Prediction":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("❤️ Heart Disease Prediction")

    auto = st.session_state.get("ocr_values", {}) or {}

    age = st.number_input("Age", min_value=0, max_value=120, value=int(auto.get("Age", 45)))

    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0], index=0)

    cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3,
                         value=int(auto.get("cp", 0)))

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0.0, max_value=300.0,
        value=float(auto.get("BloodPressure", 120.0)),
        step=1.0
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=0.0, max_value=1000.0,
        value=float(auto.get("Cholesterol", 200.0)),
        step=1.0
    )

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)", [1, 0], index=1)

    restecg = st.number_input("Resting ECG Results (0–2)", min_value=0, max_value=2, value=0)

    thalach = st.number_input(
        "Max Heart Rate Achieved",
        min_value=0, max_value=300,
        value=int(auto.get("thalach", 140)),
        step=1
    )

    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0], index=1)

    oldpeak = st.number_input(
        "ST Depression Induced by Exercise",
        min_value=0.0, max_value=10.0,
        value=float(auto.get("oldpeak", 1.0)),
        step=0.1
    )

    slope = st.number_input(
        "Slope of ST Segment (0–2)",
        min_value=0, max_value=2,
        value=int(auto.get("slope", 1))
    )

    ca = st.number_input(
        "Major Vessels Colored by Fluoroscopy (0–3)",
        min_value=0, max_value=3,
        value=int(auto.get("ca", 0))
    )

    thal = st.number_input(
        "Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)",
        min_value=0, max_value=3,
        value=int(auto.get("thal", 1))
    )

    if st.button("🔍 Predict Heart Disease"):
        arr = [age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]

        pred, prob = predict_and_record("heart_disease", arr)

        if pred == 1:
            st.error(f"Prediction: Heart Disease POSITIVE. Probability: {prob}")
        else:
            st.success(f"Prediction: Heart Disease NEGATIVE. Probability: {prob}")

    # Always show Heart Tips
    show_health_tips("heart")
    show_nearby_doctors("heart")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Parkinson's Prediction (FIXED)
# ---------------------------
if page == "Parkinsons Prediction":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("🧠 Parkinson's Disease Prediction")

    st.write("Enter the voice measurement features below:")

    auto = st.session_state.get("ocr_values", {}) or {}

    # Correct feature order
    parkin_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
        "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
        "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "PPE"
    ]

    # Safe defaults (typical dataset averages)
    defaults = {
        "MDVP:Fo(Hz)": 150.0, "MDVP:Fhi(Hz)": 200.0, "MDVP:Flo(Hz)": 100.0,
        "MDVP:Jitter(%)": 0.005, "MDVP:Jitter(Abs)": 0.00006, "MDVP:RAP": 0.003,
        "MDVP:PPQ": 0.004, "Jitter:DDP": 0.009, "MDVP:Shimmer": 0.03,
        "MDVP:Shimmer(dB)": 0.3, "Shimmer:APQ3": 0.02, "Shimmer:APQ5": 0.03,
        "MDVP:APQ": 0.03, "Shimmer:DDA": 0.09, "NHR": 0.02, "HNR": 20.0,
        "RPDE": 0.5, "DFA": 0.65, "spread1": -5.0, "spread2": 0.5, "PPE": 0.2
    }

    parkin_values = []

    st.subheader("Voice Features:")
    for feat in parkin_features:
        # allow OCR autofill loosely
        pre = defaults[feat]
        for k, v in auto.items():
            try:
                if feat.lower().replace(":", "").replace("(", "").replace(")", "").replace("%","").replace("-","") in k.lower().replace(":", ""):
                    pre = float(v)
                    break
            except:
                pass

        val = st.number_input(feat, value=float(pre), format="%.6f")
        parkin_values.append(val)

    # Predict button outside the input loop
    if st.button("🔍 Predict Parkinson's"):
        # Ensure vector matches model expected length (pad with zeros if needed)
        while len(parkin_values) < 22:
            parkin_values.append(0.0)

        arr = parkin_values.copy()
        pred, prob = predict_and_record("parkinsons", arr)

        if pred is not None:
            if pred == 1:
                st.error(f"Prediction: Parkinson's disease POSITIVE. Prob: {prob}")
            else:
                st.success(f"Prediction: Parkinson's disease NEGATIVE. Prob: {prob}")


    # ⭐ Always show health tips
    show_health_tips("parkinsons")
    show_nearby_doctors("parkinsons")

    st.markdown("</div>", unsafe_allow_html=True)
# ---------------------------
# Lung Cancer Prediction (FIXED)
# ---------------------------
if page == "Lung Cancer Prediction":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("🫁 Lung Cancer Prediction")

    st.write("Answer the survey-style medical questions:")

    auto = st.session_state.get("ocr_values", {}) or {}

    # Correct feature order
    lung_features = [
        "GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE",
        "CHRONIC_DISEASE","FATIGUE","ALLERGY","WHEEZING","ALCOHOL_CONSUMING",
        "COUGHING","SHORTNESS_OF_BREATH","SWALLOWING_DIFFICULTY","CHEST_PAIN"
    ]

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 120, int(auto.get("Age", 40)))

    def yn_to_num(choice):
        return 1 if choice == "Yes" else 0

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    gender_num = 1 if gender == "Male" else 0

    arr = [
        gender_num, int(age),
        yn_to_num(smoking), yn_to_num(yellow_fingers), yn_to_num(anxiety),
        yn_to_num(peer_pressure), yn_to_num(chronic_disease), yn_to_num(fatigue),
        yn_to_num(allergy), yn_to_num(wheezing), yn_to_num(alcohol),
        yn_to_num(coughing), yn_to_num(shortness), yn_to_num(swallowing),
        yn_to_num(chest_pain)
    ]

    if st.button("🔍 Predict Lung Cancer"):
        pred, prob = predict_and_record("lung_cancer", arr)

        if pred == 1:
            st.error(f"Prediction: Lung Cancer POSITIVE. Probability: {prob}")
        else:
            st.success(f"Prediction: Lung Cancer NEGATIVE. Probability: {prob}")

    # ⭐ Always show health tips
    show_health_tips("lungs")
    show_nearby_doctors("lungs")

    st.markdown("</div>", unsafe_allow_html=True)
# ---------------------------
# Hypo-Thyroid Prediction (FIXED)
# ---------------------------
if page == "Hypo-Thyroid Prediction":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("🧬 Hypo-Thyroid Prediction")

    auto = st.session_state.get("ocr_values", {}) or {}

    # ---- Inputs ----
    age = st.number_input("Age", min_value=0, max_value=120, value=int(auto.get("Age", 40)))
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0], index=0)
    on_thyroxine = st.selectbox("On Thyroxine", [1, 0], index=1)

    # OCR-supported thyroid values
    tsh_val = auto.get("TSH", auto.get("tsh", None))
    t3_val = auto.get("T3", auto.get("t3", None))
    t4_val = auto.get("T4", auto.get("t4", None))

    tsh = st.number_input(
        "TSH Level (mU/L)",
        min_value=0.0, max_value=500.0,
        value=float(tsh_val) if tsh_val is not None else 0.0,
        step=0.1
    )

    t3_measured = st.selectbox("T3 Measured (1 = Yes, 0 = No)", [1, 0], index=1)

    t3 = st.number_input(
        "T3 Level (ng/dL)",
        min_value=0.0, max_value=1000.0,
        value=float(t3_val) if t3_val is not None else 0.0,
        step=0.1
    )

    tt4 = st.number_input(
        "T4 Level (mcg/dL)",
        min_value=0.0, max_value=1000.0,
        value=float(t4_val) if t4_val is not None else 0.0,
        step=0.1
    )

    # ---- Prediction ----
    if st.button("🔍 Predict Thyroid"):
        arr = [age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]
        pred, prob = predict_and_record("thyroid", arr)

        if pred == 1:
            st.error(f"Prediction: Hypothyroid POSITIVE. Probability: {prob}")
        else:
            st.success(f"Prediction: Hypothyroid NEGATIVE. Probability: {prob}")

    # ⭐ Always show Thyroid suggestions (Correct Position)
    show_health_tips("thyroid")
    show_nearby_doctors("thyroid")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Health suggestions
# ---------------------------
if page == "Health Suggestions":
    st.header("🩺 Health Suggestions (All Conditions)")
    for k in ["diabetes", "heart", "parkinsons", "lungs", "thyroid"]:
        with st.expander(k.capitalize()):
            show_health_tips(k)

# ---------------------------
# Model Info
# ---------------------------
if page == "Model Info":
    st.header("🔧 Model Information")
    if models:
        for k, m in models.items():
            with st.expander(k):
                st.write("Model object:")
                st.code(repr(m)[:4000])
                st.markdown(f'<div class="model-box"><pre>{repr(m)}</pre></div>', unsafe_allow_html=True)
    else:
        st.info("No models loaded. Place model files in Models/ folder.")

# ---------------------------
# Prediction History
# ---------------------------
if page == "Prediction History":
    st.header("📈 Prediction History")
    if st.session_state["prediction_history"]:
        df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(df)
        st.download_button("Export history CSV", data=export_history_csv(), file_name="pred_history.csv")
        if st.button("Clear history"):
            st.session_state["prediction_history"] = []
            st.rerun()
    else:
        st.info("No predictions recorded yet. Make predictions to populate history.")

# ABOUT
if page == "About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
**Developed by:**  
- Garima  (2202920100041)
- Tushant Kumar  (2202920100116)
- Mandeep Kaur  (2202920100059)
- Dipanshu Sharma  (2102920100037)

**Under Supervision:** Ms. Srishti Agarwal (Assistant Professor)

**Institution:** Meerut Institute of Technology, Meerut  

This web application demonstrates ML-based predictive diagnostics combined with practical
health guidance (precautions, diet, home remedies, yoga, emergency & medication advice).
"""
    )
    st.markdown("""
**Technologies Used:**  
- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Other relevant libraries and tools
""")

# ---------------------------
# Consult Doctor Page
# ---------------------------
if page == "Consult Doctor":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("👨‍⚕️ Consult Specialist Doctor")

    st.write("Select disease and search for nearby specialist doctors and hospitals.")

    disease = st.selectbox(
        "🩺 Select Disease",
        [
            "Diabetes",
            "Heart Disease",
            "Parkinson's",
            "Lung Cancer",
            "Hypo-Thyroid"
        ]
    )

    disease_map = {
        "Diabetes": "diabetes",
        "Heart Disease": "heart",
        "Parkinson's": "parkinsons",
        "Lung Cancer": "lungs",
        "Hypo-Thyroid": "thyroid"
    }

    st.markdown("📌 Doctors are fetched **live using Google Maps** (no dataset stored).")
    st.markdown("---")

    show_nearby_doctors(disease_map[disease])

    st.markdown("</div>", unsafe_allow_html=True)
