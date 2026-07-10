"""
Demo interactiva: Regresión Logística vs Random Forest
========================================================

INSTALACIÓN (una sola vez):
    pip install shiny scikit-learn joblib pandas numpy

ARCHIVOS NECESARIOS en la misma carpeta que este script:
    - logreg.pkl          (joblib.dump(best_logreg_model_auc, 'logreg.pkl'))
    - random_forest.pkl   (joblib.dump(best_rf_model_auc, 'random_forest.pkl'))

EJECUCIÓN:
    shiny run app.py
    (o "shiny run --reload app.py" 

"""

import re
import joblib
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

# ============================================================
# 1. MAPEOS: valor numérico -> texto legible para el usuario
# ============================================================
mapeos = {
    'tecnica': {0.0: "Estándar", 1.0: "Mini-perc", 2.0: "Combinado flexible anterógrado",
                3.0: "Combinado flexible retrógrado", 4.0: "Combinado semirrígido retrógrado",
                5.0: "Combinado anterógrado retrógrado", 6.0: "Micro-perc"},
    'Sexo': {0: 'Hombre', 1: 'Mujer'},
    'Diabetes': {0: 'No', 1: 'No insulinodependiente', 2: 'Insulinodependiente'},
    'HTA': {0: 'No', 1: 'Sí'},
    'Fumador_tabaco': {0: 'No', 1: 'Sí', 2: 'Exfumador'},
    'ASA': {1: 'ASA I', 2: 'ASA II', 3: 'ASA III', 4: 'ASA IV'},
    'Cultresul': {0: 'Negativo', 1: 'Positivo', 2: 'Contaminado'},
    'Microorganismo': {0: 'Negativo', 1: 'Ureasa negativo', 2: 'Ureasa positivo'},
    'tipotto': {0: 'Primario', 1: 'Post-ESWL', 2: 'Post-URS', 3: 'Post-NLP'},
    'lado': {0: 'Derecho', 1: 'Izquierdo', 2: 'Bilateral', 3: 'Trasplante'},
    'Guy_Score': {1: 'I', 2: 'II', 3: 'III', 4: 'IV'},
    'posicion_paciente': {0: 'Prono', 1: 'Supino'},
    'acceso_puncion': {0: 'Bajo escopia', 1: 'Ecografía', 2: 'Ambas'},
    'cateterismo_ureteral': {0: 'No', 1: 'Sí'},
    'contraste': {0: 'No', 1: 'Sí'},
    'metodo_dilatacion': {0: 'Amplatz', 1: 'Balón', 2: 'Ambos', 3: 'Metálicos'},
    'multitrayecto': {0: 'No', 1: 'Sí'},
    'fuente_fragmentacion': {0: 'Lithoclast', 1: 'Ultrasonido', 2: 'Lithoclast + Ultrasonido',
                             3: 'Láser holmium', 4: 'Cesta', 5: 'Pinzas',
                             6: 'Lithoclast + Láser', 7: 'Irrigación'},
    'facilidad_localizacion': {0: 'Fácil', 1: 'Media', 2: 'Difícil'},
    'drenajes': {0: 'No', 1: 'Nefrostomía', 2: 'Doble J', 3: 'Doble J + nefrostomía',
                 4: 'Catéter ureteral 24h', 5: 'Catéter ureteral 24h + nefrostomía'},
    'tubeless': {0: 'No', 1: 'Sí'},
    'Procalcitonina': {0.0: 'Rango normal', 1.0: 'Fuera del rango normal'},
    'LeucocitosTotales': {0.0: 'Rango normal', 1.0: 'Fuera del rango normal'},
    'ProcentajeNeutrofilos': {0.0: 'Rango normal', 1.0: 'Fuera del rango normal'},
}

# Localización: one-hot "manual" con nombres de columna, no números
GRUPO_LOCALIZACION = {
    'Pelvis': 'loc_pelvis',
    'Cáliz superior': 'loc_caliz_sup',
    'Cáliz medio': 'loc_caliz_med',
    'Cáliz inferior': 'loc_caliz_inf',
    'Renal y ureteral': 'loc_renal y ureteral',
    'Ureteral': 'ureteral',
}

# ============================================================
# 2. CASOS ESPECIALES: categorías "ambos" que activan 2 columnas
# ============================================================
COMBOS = {
    ('acceso_puncion', '2'): ['acceso_puncion_0', 'acceso_puncion_1'],
    ('fuente_fragmentacion', '2'): ['fuente_fragmentacion_0', 'fuente_fragmentacion_1'],
    ('fuente_fragmentacion', '6'): ['fuente_fragmentacion_0', 'fuente_fragmentacion_3'],
    ('drenajes', '3'): ['drenajes_1', 'drenajes_2'],
    ('drenajes', '5'): ['drenajes_1', 'drenajes_4'],  # 
}

# ============================================================
# 3. CLASIFICACIÓN DE VARIABLES
# ============================================================
# Grupos one-hot numéricos (nombre_base -> columnas nombre_base_N)
VARIABLES_ONEHOT_NUM = ['Cultresul', 'Microorganismo', 'lado', 'tecnica',
                         'posicion_paciente', 'acceso_puncion', 'metodo_dilatacion',
                         'fuente_fragmentacion', 'drenajes', 'Fumador_tabaco']

# Variables categóricas simples (una sola columna, valor directo)
VARIABLES_SIMPLES_CATEGORICAS = ['Sexo', 'Diabetes', 'HTA', 'ASA', 'tipotto',
                                  'cateterismo_ureteral', 'contraste', 'multitrayecto',
                                  'facilidad_localizacion', 'tubeless',
                                  'Procalcitonina', 'LeucocitosTotales',
                                  'ProcentajeNeutrofilos', 'Guy_Score']

# Variables numéricas continuas (sin mapeo, el usuario mete el número directamente)
VARIABLES_NUMERICAS = {
    'BMI': (14.0, 48.0, 27.0),
    'diametro_mayor': (0.0, 90.0, 22.0),
    'diametro_menor': (0.0, 70.0, 13.0),
    'numero': (1, 20, 1),
    'UH': (0.0, 2500.0, 1000.0),
    'distanciapielcalculo': (0.0, 150.0, 10.0),
    'EDAD': (0, 100, 56),
    'duracion_tto': (0, 300, 105),
    'calibre_vaina': (10.0, 30.0, 24.0),
}
# (min, max, valor_por_defecto) — 

# Etiquetas más legibles para mostrar en pantalla
ETIQUETAS = {
    'BMI': 'BMI (kg/m²)', 'diametro_mayor': 'Diámetro mayor (mm)',
    'diametro_menor': 'Diámetro menor (mm)', 'numero': 'Número de cálculos',
    'UH': 'Unidades Hounsfield', 'distanciapielcalculo': 'Distancia piel-cálculo (mm)',
    'EDAD': 'Edad', 'duracion_tto': 'Duración del tratamiento (min)',
    'localizacion': 'Localización', 'calibre_vaina': 'Calibre de vaina (Fr)',
}

# ============================================================
# 4. CARGA DE MODELOS
# ============================================================
logreg_pipeline = joblib.load('logreg.pkl')
rf_model = joblib.load('random_forest.pkl')
COLUMNAS_MODELO = list(logreg_pipeline.feature_names_in_)

# Umbrales óptimos encontrados con TunedThresholdClassifierCV . Se aplican sobre predict_proba().
UMBRAL_LOGREG = 0.3065
UMBRAL_RF = 0.2676


# ============================================================
# 5. RECONSTRUCCIÓN DEL VECTOR DE ENTRADA
# ============================================================
def construir_input(valores_numericos: dict, valores_categoricos: dict) -> pd.DataFrame:
    fila = {col: 0 for col in COLUMNAS_MODELO}

    # --- Localización (caso especial, one-hot por nombre) ---
    if 'localizacion' in valores_categoricos:
        columna_activar = GRUPO_LOCALIZACION[valores_categoricos['localizacion']]
        fila[columna_activar] = 1

    # --- Resto de variables categóricas ---
    for base, categoria_elegida in valores_categoricos.items():
        if base == 'localizacion':
            continue

        if (base, categoria_elegida) in COMBOS:
            for col_activar in COMBOS[(base, categoria_elegida)]:
                fila[col_activar] = 1
            continue

        col_onehot = f"{base}_{categoria_elegida}"
        if col_onehot in fila:
            fila[col_onehot] = 1
        elif base in fila:
            fila[base] = float(categoria_elegida)

    # --- Variables numéricas continuas ---
    for var, valor in valores_numericos.items():
        if var in fila:
            fila[var] = valor

    return pd.DataFrame([fila])[COLUMNAS_MODELO]


# ============================================================
# 6. INTERFAZ (UI)
# ============================================================
def select_categoria(nombre_var, label=None):
    choices = {str(k): v for k, v in mapeos[nombre_var].items()}
    return ui.input_select(nombre_var, label or nombre_var.replace('_', ' ').capitalize(), choices=choices)


panel_preop = ui.card(
    ui.h5("Datos preoperatorios"),
    select_categoria('Sexo'),
    select_categoria('Diabetes'),
    select_categoria('HTA',"HTA"),
    select_categoria('Fumador_tabaco', "Tabaquismo"),
    select_categoria('ASA',"ASA"),
    ui.input_numeric('EDAD', ETIQUETAS['EDAD'], value=VARIABLES_NUMERICAS['EDAD'][2]),
    ui.input_numeric('BMI', ETIQUETAS['BMI'], value=VARIABLES_NUMERICAS['BMI'][2]),
    select_categoria('Cultresul', "Resultado de cultivo"),
    select_categoria('Microorganismo'),
    select_categoria('lado'),
    ui.input_select('localizacion', ETIQUETAS['localizacion'], choices=list(GRUPO_LOCALIZACION.keys())),
    select_categoria('Guy_Score'),
    ui.input_numeric('diametro_mayor', ETIQUETAS['diametro_mayor'], value=VARIABLES_NUMERICAS['diametro_mayor'][2]),
    ui.input_numeric('diametro_menor', ETIQUETAS['diametro_menor'], value=VARIABLES_NUMERICAS['diametro_menor'][2]),
    ui.input_numeric('numero', ETIQUETAS['numero'], value=VARIABLES_NUMERICAS['numero'][2]),
    ui.input_numeric('UH', ETIQUETAS['UH'], value=VARIABLES_NUMERICAS['UH'][2]),
    ui.input_numeric('distanciapielcalculo', ETIQUETAS['distanciapielcalculo'], value=VARIABLES_NUMERICAS['distanciapielcalculo'][2]),
)

panel_intraop = ui.card(
    ui.h5("Datos intraoperatorios"),
    select_categoria('tipotto', "Tipo de litiasis"),
    select_categoria('tecnica'),
    select_categoria('posicion_paciente', "Posición del paciente"),
    select_categoria('acceso_puncion', "Acceso de punción"),
    select_categoria('cateterismo_ureteral'),
    select_categoria('contraste'),
    select_categoria('metodo_dilatacion', "Método de dilatación"),
    select_categoria('multitrayecto'),
    ui.input_numeric('calibre_vaina', ETIQUETAS['calibre_vaina'], value=VARIABLES_NUMERICAS['calibre_vaina'][2]),
    select_categoria('fuente_fragmentacion', "Fuente de fragmentación"),
    select_categoria('facilidad_localizacion', "Facilidad de localización"),
    ui.input_numeric('duracion_tto', ETIQUETAS['duracion_tto'], value=VARIABLES_NUMERICAS['duracion_tto'][2]),
    select_categoria('drenajes'),
    select_categoria('tubeless'),
)

panel_postop = ui.card(
    ui.h5("Datos analíticos / postoperatorios"),
    select_categoria('Procalcitonina'),
    select_categoria('LeucocitosTotales', "Leucocitos totales"),
    select_categoria('ProcentajeNeutrofilos', "Porcentaje de neutrófilos"),
)

app_ui = ui.page_fluid(
    ui.panel_title("Predicción de complicación de origen infeccioso tras NLP"),
    ui.navset_tab(
        ui.nav_panel("Preoperatorio", panel_preop),
        ui.nav_panel("Intraoperatorio", panel_intraop),
        ui.nav_panel("Postoperatorio / Analítica", panel_postop),
    ),
    ui.br(),
    ui.input_action_button("predecir_btn", "Predecir", class_="btn-primary"),
    ui.br(), ui.br(),
    ui.output_text_verbatim("resultado"),
)


# ============================================================
# 7. LÓGICA (SERVER)
# ============================================================
def server(input, output, session):

    @output
    @render.text
    @reactive.event(input.predecir_btn)
    def resultado():

        valores_numericos = {
            var: input[var]() for var in VARIABLES_NUMERICAS.keys()
        }

        valores_categoricos = {}
        for var in VARIABLES_SIMPLES_CATEGORICAS + VARIABLES_ONEHOT_NUM:
            valores_categoricos[var] = input[var]()
        valores_categoricos['localizacion'] = input['localizacion']()

        X = construir_input(valores_numericos, valores_categoricos)

        proba_logreg = logreg_pipeline.predict_proba(X)[0][1]
        proba_rf = rf_model.predict_proba(X)[0][1]

        decision_logreg = "COMPLICACIÓN DE ORIGEN INFECCIOSO" if proba_logreg >= UMBRAL_LOGREG else "No complicación de origen infeccioso"
        decision_rf = "COMPLICACIÓN DE ORIGEN INFECCIOSO" if proba_rf >= UMBRAL_RF else "No complicación de origen infeccioso"

        return (
            f"Regresión Logística  →  Probabilidad: {proba_logreg:.2%}  "
            f"(umbral: {UMBRAL_LOGREG:.2%})  →  {decision_logreg}\n"
            f"Random Forest        →  Probabilidad: {proba_rf:.2%}  "
            f"(umbral: {UMBRAL_RF:.2%})  →  {decision_rf}\n\n"

        )


app = App(app_ui, server)
