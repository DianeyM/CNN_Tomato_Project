import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from PIL import Image
import io

# --- 1. CONFIGURACIÓN DE LA APLICACIÓN ---
# Desactivar advertencias de Keras (opcional)
tf.get_logger().setLevel('ERROR') 

# Título y encabezado
st.set_page_config(
    page_title="Diagnóstico de Tomate (90.45%)",
    page_icon="🍅"
)

st.title("🍅 PROTOTIPO WEB: DIAGNÓSTICO DE ENFERMEDADES DEL TOMATE")
st.markdown("Clasificación multiclase de 10 condiciones de la hoja de tomate usando MobileNetV2 (Precisión: 90.45%).")
st.markdown("---")

# --- 2. PARÁMETROS CLAVE DEL MODELO ---
MODEL_PATH = 'MobileNetV2_Tomato_Classifier.h5' 
CLASSES_PATH = 'class_names.pkl' 
IMG_SIZE = (224, 224) 
UMBRAL_CONFIANZA = 0.70 # Estrategia de Rechazo: 70%

# 🚨 FUNCIÓN DE FORMATEO CORREGIDA 🚨
# Solo reemplaza guiones bajos por espacios y usa título de caso.
# Esto asegura que el nombre predicho coincida con la clave del diccionario (Ej: "Tomato Bacterial Spot").
def format_class_name(name):
    name = name.replace("_", " ") 
    name = name.title()
    return name

# Mapeo de resultados para visualización y recomendaciones
# 🚨 CLAVES VERIFICADAS PARA COINCIDIR CON LA SALIDA DEL FORMATO 🚨
CLASS_MAPPING = {
    "Tomato Healthy": ("Hoja Sana", "✅", "La hoja de tomate no presenta síntomas visibles de plaga o enfermedad. Mantenimiento rutinario."),
    "Tomato Bacterial Spot": ("Mancha Bacteriana", "⚠️", "Causada por la bacteria Xanthomonas spp. Requiere aplicación de bactericidas a base de cobre."),
    "Tomato Early Blight": ("Tizón Temprano", "⚠️", "Causado por el hongo Alternaria solani. Aplicar fungicidas preventivos y rotación de cultivos."),
    "Tomato Late Blight": ("Tizón Tardío", "🚨", "Causado por Phytophthora infestans. Es una enfermedad destructiva. Aislar y eliminar las plantas infectadas."),
    "Tomato Leaf Mold": ("Moho de la Hoja", "⚠️", "Causado por Passalora fulva. Mejorar la ventilación y reducir la humedad. Usar fungicidas."),
    "Tomato Septoria Leaf Spot": ("Mancha Foliar Por Septoria", "⚠️", "Causado por Septoria lycopersici. Usar fungicidas y evitar mojar el follaje."),
    "Tomato Spider Mites Two Spotted Spider Mite": ("Ácaros (Araña Roja)", "⚠️", "Causado por la plaga Tetranychus urticae. Aplicar acaricidas o depredadores naturales."),
    "Tomato Target Spot": ("Mancha En Diana", "⚠️", "Causado por Corynespora cassiicola. Usar fungicidas y eliminar restos de plantas infectadas."),
    
    "Tomato Tomato Mosaic Virus": ("Virus del Mosaico (ToMV)", "🚨", "Enfermedad viral. No tiene cura. Eliminar y destruir la planta para evitar la propagación."),
    "Tomato Tomato Yellowleaf Curl Virus": ("Virus del Enrollamiento de la Hoja (TYLCV)", "🚨", "Enfermedad viral. No tiene cura. El control se centra en el vector (mosca blanca).")
}

# --- 3. CARGA DE MODELO Y CLASES (Caching para eficiencia) ---

@st.cache_resource
def load_assets():
    try:
        model = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'rb') as f:
            class_names = pickle.load(f)
            
        return model, class_names
        
    except FileNotFoundError as e:
        st.error(f"Error al cargar archivos: {e}. Asegúrate de que '{MODEL_PATH}' y '{CLASSES_PATH}' estén en la misma carpeta que 'streamlit_app.py'.")
        return None, None

model, class_names = load_assets()

# --- 4. FUNCIÓN DE PREDICCIÓN ---

def predict_image(img_bytes, model, class_names):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array, verbose=0)
    
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_class = class_names[predicted_index]
    
    return predicted_class, confidence

# --- 5. INTERFAZ Y LÓGICA DE LA APP ---

if model:
    uploaded_file = st.file_uploader(
        "Sube una imagen de una hoja de tomate para diagnosticar (JPG/PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagen Subida', use_column_width=True, width=300)
        st.markdown("---")

        with st.spinner('Analizando la imagen...'):
            predicted_class_raw, confidence = predict_image(
                uploaded_file.getvalue(), 
                model, 
                class_names
            )
            
            # Formateo y Umbral de Confianza
            predicted_class_formatted = format_class_name(predicted_class_raw)
            confidence_percent = confidence * 100
            
            if confidence >= UMBRAL_CONFIANZA:
                
                display_name, emoji, recommendation = CLASS_MAPPING.get(predicted_class_formatted, ("Diagnóstico Desconocido", "❓", "Información no disponible."))
                
                # Mostrar resultado de ACEPTACIÓN
                st.subheader(f"{emoji} Resultado del Diagnóstico: {display_name}")
                st.metric(
                    label="Confianza del Modelo", 
                    value=f"{confidence_percent:.2f}%",
                    delta=f"Umbral de Aceptación: {UMBRAL_CONFIANZA*100:.0f}%"
                )
                st.info(f"**Recomendación:** {recommendation}")
                
            else:
                # Mostrar resultado de RECHAZO (Umbral no superado)
                st.error("❌ IMAGEN NO VÁLIDA O INSUFICIENTE CERTEZA")
                st.warning(f"""
                El modelo no pudo identificar una hoja de tomate o la certeza de la predicción 
                ({confidence_percent:.2f}%) es inferior al umbral mínimo requerido del 
                **{UMBRAL_CONFIANZA*100:.0f}%**.
                """)
                st.info("Por favor, sube una imagen clara de una hoja de tomate.")

st.markdown("---")
st.markdown("Desarrollado con Python y Streamlit para la UPTC con base en las CNN.")

