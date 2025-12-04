import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuración de semilla para reproducibilidad
np.random.seed(42)

# ==========================================
# 1. GENERACIÓN DE DATOS HISTÓRICOS (MOCK DATA)
# ==========================================
# Simulamos una base de datos de jugadores que ya alcanzaron su potencial
# Variables: Edad, Rating Actual, Potencial Físico, Inteligencia Táctica -> Rating Futuro

n_samples = 500
data = {
    'edad_captacion': np.random.randint(16, 21, n_samples),
    'rating_actual': np.random.normal(65, 5, n_samples),
    'fisico': np.random.normal(70, 10, n_samples),
    'iq_tactico': np.random.normal(70, 10, n_samples),
}
df_historico = pd.DataFrame(data)

# La variable objetivo (Rating Futuro) se genera con una fórmula con algo de ruido aleatorio
# Lógica: Más joven con alto rating actual = mayor futuro.
df_historico['rating_futuro_real'] = (
    (df_historico['rating_actual'] * 1.2) + 
    (df_historico['fisico'] * 0.1) + 
    (df_historico['iq_tactico'] * 0.2) - 
    (df_historico['edad_captacion'] * 1.5) + 
    np.random.normal(0, 3, n_samples) # Ruido aleatorio (lesiones, suerte)
)

# ==========================================
# 2. ENTRENAMIENTO DEL MODELO DE REGRESIÓN
# ==========================================
X = df_historico[['edad_captacion', 'rating_actual', 'fisico', 'iq_tactico']]
y = df_historico['rating_futuro_real']

# Entrenamos un modelo simple para establecer la proyección base
modelo = LinearRegression()
modelo.fit(X, y)

print(f"✅ Modelo entrenado. R2 Score: {modelo.score(X, y):.4f}")

# ==========================================
# 3. NUEVOS PROSPECTOS (SCOUTING)
# ==========================================
# Estos son los jugadores que queremos evaluar hoy
prospectos_data = [
    {'nombre': 'Jugador A (Joven promesa)', 'edad_captacion': 16, 'rating_actual': 70, 'fisico': 80, 'iq_tactico': 75},
    {'nombre': 'Jugador B (Físico puro)', 'edad_captacion': 18, 'rating_actual': 72, 'fisico': 90, 'iq_tactico': 60},
    {'nombre': 'Jugador C (Técnico maduro)', 'edad_captacion': 20, 'rating_actual': 76, 'fisico': 65, 'iq_tactico': 85},
    {'nombre': 'Jugador D (Promedio)', 'edad_captacion': 19, 'rating_actual': 68, 'fisico': 70, 'iq_tactico': 70},
]
df_prospectos = pd.DataFrame(prospectos_data)

# ==========================================
# 4. SIMULACIÓN DE ESCENARIOS (MONTE CARLO)
# ==========================================
# No nos basta con una predicción, queremos ver la varianza.
# Simularemos 10,000 carreras posibles para cada jugador.

n_simulaciones = 10000
umbral_exito = 85  # Definimos "Éxito" como llegar a un rating de 85 (Clase Mundial)
resultados = []

print("\n🔄 Ejecutando simulaciones de escenarios...")

for index, row in df_prospectos.iterrows():
    features = row[['edad_captacion', 'rating_actual', 'fisico', 'iq_tactico']].values.reshape(1, -1)
    
    # 1. Predicción base del modelo
    prediccion_base = modelo.predict(features)[0]
    
    # 2. Generar escenarios (Simulación)
    # Asumimos una desviación estándar de 4 puntos (incertidumbre del desarrollo)
    escenarios = np.random.normal(prediccion_base, 4, n_simulaciones)
    
    # 3. Cálculo de métricas probabilísticas
    prob_exito = np.mean(escenarios > umbral_exito) * 100
    piso = np.percentile(escenarios, 5)   # El 5% peor caso (riesgo de fracaso)
    techo = np.percentile(escenarios, 95) # El 5% mejor caso (potencial máximo)
    
    resultados.append({
        'Nombre': row['nombre'],
        'Edad': row['edad_captacion'],
        'Rating Actual': row['rating_actual'],
        'Potencial Esperado (Media)': round(np.mean(escenarios), 1),
        'Piso (Peor Caso)': round(piso, 1),
        'Techo (Mejor Caso)': round(techo, 1),
        'Probabilidad Éxito (>85)': f"{round(prob_exito, 1)}%"
    })

# ==========================================
# 5. RESULTADOS Y VISUALIZACIÓN
# ==========================================
df_resultados = pd.DataFrame(resultados)

print("\n📊 TABLA DE POTENCIAL Y RIESGO:")
print("-" * 80)
print(df_resultados.to_markdown(index=False))

# Gráfico de distribución para el Jugador A vs Jugador C
plt.figure(figsize=(10, 6))

# Re-simular para graficar (solo visualización)
feat_A = df_prospectos.iloc[0][['edad_captacion', 'rating_actual', 'fisico', 'iq_tactico']].values.reshape(1, -1)
sim_A = np.random.normal(modelo.predict(feat_A)[0], 4, n_simulaciones)

feat_C = df_prospectos.iloc[2][['edad_captacion', 'rating_actual', 'fisico', 'iq_tactico']].values.reshape(1, -1)
sim_C = np.random.normal(modelo.predict(feat_C)[0], 4, n_simulaciones)

plt.hist(sim_A, bins=50, alpha=0.6, label='Jugador A (16 años)', color='blue', density=True)
plt.hist(sim_C, bins=50, alpha=0.6, label='Jugador C (20 años)', color='orange', density=True)
plt.axvline(umbral_exito, color='red', linestyle='dashed', linewidth=2, label=f'Umbral Éxito ({umbral_exito})')

plt.title('Simulación de Desarrollo: Distribución de Probabilidad')
plt.xlabel('Rating Futuro Simulado')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()