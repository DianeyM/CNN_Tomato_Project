import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from PIL import Image
import io
import pandas as pd

# --- 1. CONFIGURACIÓN DE LA APLICACIÓN ---
# Suprimir advertencias de TensorFlow para mantener la interfaz limpia
tf.get_logger().setLevel('ERROR') 

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
UMBRAL_CONFIANZA = 0.70 # Umbral de Aceptación

def format_class_name(name):
    """
    Función CRUCIAL para estandarizar el nombre de la clase, saneando caracteres problemáticos 
    como los que se encuentran en el archivo class_names.pkl.
    """
    # 1. Eliminar caracteres no alfanuméricos que NO sean guiones bajos (ej: el '%' en el TYLCV).
    name = re.sub(r'[^\w_]', '', name)
    
    # 2. Reemplaza cualquier secuencia de guiones bajos con un solo espacio.
    #    Esto maneja tanto "_" como "__" (corrigiendo Target Spot y Virus).
    name = re.sub(r'_{1,}', ' ', name)
    
    # 3. Capitaliza la primera letra de cada palabra
    name = name.title()
    
       
    # 5. Limpia cualquier espacio sobrante al principio o al final (¡SOLUCIÓN FINAL!).
    name = name.strip() 
    
    return name

# Mapeo de resultados para visualización y recomendaciones
# Las CLAVES deben usar ESPACIOS y CAPITALIZACIÓN DE TÍTULO para coincidir con el
# formato de 'format_class_name'.
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

# --- 4. FUNCIÓN DE PREDICCIÓN (Modificada para devolver todas las probabilidades) ---

def predict_image(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 

    # predictions es un array de 10 probabilidades
    predictions = model.predict(img_array, verbose=0)[0]
    
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return predicted_index, confidence, predictions

# --- 5. FUNCIÓN PARA MOSTRAR LA TABLA DE PROBABILIDADES ---
def display_top_n_probabilities(predictions, class_names, n=5):
    # Crear un DataFrame con probabilidades
    results = pd.DataFrame({
        # Se formatea el nombre para la presentación en la tabla
        'Clase': [format_class_name(name) for name in class_names], 
        'Probabilidad': predictions
    })
    
    # Ordenar por probabilidad descendente
    results = results.sort_values(by='Probabilidad', ascending=False)
    
    # Mostrar solo las N más altas (o todas si son menos de N)
    results_display = results.head(n).copy()
    
    # Formatear la probabilidad a porcentaje para la vista
    results_display['Probabilidad'] = (results_display['Probabilidad'] * 100).map('{:.2f}%'.format)
    
    st.markdown("##### 🔍 Distribución de Probabilidades (Top 5)")
    st.table(results_display.reset_index(drop=True))

# --- 6. INTERFAZ Y LÓGICA DE LA APP ---

if model:
    uploaded_file = st.file_uploader(
        "Sube una imagen de una hoja de tomate para diagnosticar (JPG/PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Imagen Subida', use_column_width=True, width=300)
        st.markdown("---")

        with st.spinner('Analizando la imagen...'):
            predicted_index, confidence, all_predictions = predict_image(
                uploaded_file.getvalue(), 
                model
            )
            
            # 1. Obtener nombre de la clase RAW (e.g., 'tomato_late_blight')
            predicted_class_raw = class_names[predicted_index]
            
            # *************************************************************************
            # *** CORRECCIÓN DEFINITIVA APLICADA AQUÍ ***
            # La clave de búsqueda se formatea COMPLETAMENTE (guiones bajos a espacios, minúsculas a Título)
            # para coincidir con el formato de las CLAVES en CLASS_MAPPING.
            lookup_key = format_class_name(predicted_class_raw)
            # *************************************************************************
            
            confidence_percent = confidence * 100
            
            # --- LÓGICA DEL UMBRAL DE CONFIANZA ---
            if confidence >= UMBRAL_CONFIANZA:
                
                # Se usa la clave formateada para obtener el diagnóstico y la recomendación
                display_name, emoji, recommendation = CLASS_MAPPING.get(
                    lookup_key, 
                    # Valor de reserva en caso de que la clave formateada no exista (¡no debería pasar ahora!)
                    ("Diagnóstico Desconocido", "❓", "Información no disponible.")
                )
                
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

            # Mostrar la tabla de probabilidades en ambos casos (acepte o rechace)
            display_top_n_probabilities(all_predictions, class_names)


st.markdown("---")
st.markdown("Desarrollado con Python y Streamlit para la UPTC v5.")







