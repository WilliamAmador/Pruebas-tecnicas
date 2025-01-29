import sklearn
import imblearn
import xgboost
import lightgbm

print("Scikit-learn version:", sklearn.__version__)
print("Imbalanced-learn version:", imblearn.__version__)
print("XGBoost version:", xgboost.__version__)
print("LightGBM version:", lightgbm.__version__)


# %%
# Importar librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo casv
archivo = "MercadoLibre Data Scientist Technical Challenge - Dataset.csv"
data = pd.read_csv(archivo)

# Tamano del dataFrame
dimensiones = data.shape
print(f"El DataFrame tiene {dimensiones[0]} filas y {dimensiones[1]} columnas.")
# %%
# Explorar información general
print("Información general:")
data.info()

# Contar valores faltantes
valores_perdidos = data.isnull().sum()
print("\nValores faltantes:")
print(valores_perdidos[valores_perdidos > 0])
# %%
# Proporción de transacciones fraudulentas
por_fraude = data['fraude'].value_counts(normalize=True)
print("Proporción de transacciones fraudulentas:")
por_fraude
# %%
# Calcular ganancia y perdida por transacción
data['ganancia_real'] = data.apply(
    lambda row: 0.25 * row['monto'] if row['fraude'] == 0 else -1.0 * row['monto'], axis=1
)
# print(data.head(10))

# Ganancia total
ganancia_total = data['ganancia_real'].sum()
print(f"Ganancia total actual: ${ganancia_total:.2f}")

# Calcular la perdida
perdida_fraudes = data[data['fraude'] == 1]['monto'].sum()
print(f"Perdida fraudes: ${perdida_fraudes:.2f}")
# %%
# Columnas no numericas
col_numericas = data.select_dtypes(exclude=['float64', 'int64']).columns
print("Columnas no numericas:")
print(col_numericas)

for col in col_numericas:
    print(f"\nColumna: {col}")

    # Mostrar el número de categorías únicas
    num_unicas = data[col].nunique()
    print(f"Número de categorías únicas: {num_unicas}")
# %% 
data['monto_bin']=pd.qcut(data['monto'], 4)
    
# %%
## Numero de categorias unicas de J de acuerdo al label
data.groupby('fraude')['j'].nunique()
# %%
## Interseccion de categorias de J entre el 1 y 0
mask = (data['fraude']==1)
j_fraude = set(data.loc[mask,'j'].unique())
j_no_fr = set(data.loc[~mask,'j'].unique())
inter = j_fraude.intersection(j_no_fr)
len(inter)
### Casi todas las categorias de J de la categoria 1 estan contenidas en la categoria 0, la variable no es informativa
# %%
# G es el codigo iso de pais
data['g'].unique()
# %%
## Numero de categorias unicas de G de acuerdo al label
## Hay paises con muy poca representacion en la muestra, se recomienda agruparlos en una clase "otros"
data.groupby(['g','fraude'])['j'].count()
# %%
## Mapear el Pais a su continente para agrupar
iso_codes = pd.read_csv('continents2.csv')
iso_codes = {iso:cont for iso,cont in iso_codes[['alpha-2','sub-region']].values.tolist()}
data['g_sub_region'] = data['g'].map(iso_codes)
data.groupby(['g_sub_region','fraude'])['j'].count()
# %%
## Se decide usar solo los datos de latam, hay poca representatividad de los demas paises
data = data[data['g_sub_region']=='Latin America and the Caribbean'].copy()
# %%
print(data['g'])

# %%
# Crear el one-hot encoding de la columna 'region_category' con valores 1 y 0
one_hot_encoded = pd.get_dummies(data['g'], prefix='region', dtype=int)

# Añadir las nuevas columnas al DataFrame original
data = pd.concat([data, one_hot_encoded], axis=1)

# Verificar el resultado
print(data.head())

# %%
print(data.columns)

# %%
from sklearn.preprocessing import LabelEncoder

# Crear un LabelEncoder
le = LabelEncoder()

data['g_numerica'] = le.fit_transform(data['g'])
#data['j_numerica'] = le.fit_transform(data['j']) #J se elimina porque es poco informativa
data['o_numerica'] = le.fit_transform(data['o'])
data['p_numerica'] = le.fit_transform(data['p'])
# %%
# Deja las columnas numericas
col_numericas = data.drop(columns=['ganancia_real']).select_dtypes(include=['float64', 'int64']).columns

# Calcul0 correlacion con 'fraude'
fraude_correlacion = data[col_numericas].corr(numeric_only=True)['fraude'].sort_values()

# Mostrar las correlaciones
print("Correlacion 'fraude':")
print(fraude_correlacion)

# Grafica correlacion
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(data[col_numericas].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Grafica correlacion")
plt.show()
# %%
data_significativa = data[
    ['a', 'b', 'c', 'd', 'e', 'f', 'region_AR', 'region_BO', 'region_BR', 'region_BS',
    'region_CL', 'region_CO', 'region_CR', 'region_DO', 'region_EC',
    'region_HN', 'region_MX', 'region_NI', 'region_PA', 'region_PE',
    'region_PR', 'region_PY', 'region_UY', 'h', 'k', 'l',
     'm', 'n', 'o_numerica', 'p_numerica','monto','monto_bin', 'score', 'fraude']
]
fraude_correlacion.sort_index().index
# %%
# Contar valores faltantes
valores_perdidos = data_significativa.isnull().sum()
print("\nValores faltantes:")
print(valores_perdidos[valores_perdidos > 0])
# %%
# Filtrar las filas donde 'fraude' es igual a 1
filas_fraude_1 = data_significativa[data_significativa['fraude'] == 1]
filas_con_na_fraude_1 = filas_fraude_1.isna().sum(axis=1)

# Contar cuántas filas tienen al menos un valor faltante
filas_con_na_fraude_1_count = (filas_con_na_fraude_1 > 0).sum()
total_filas_fraude_1 = filas_fraude_1.shape[0]

# Imprimir el resultado con la información adicional
print(f"Cantidad de filas con 'fraude' igual a 1 y datos faltantes: {filas_con_na_fraude_1_count} de un total de {total_filas_fraude_1} filas con 'fraude' igual a 1.")
# %%
## Imputar valores faltantes con el promedio de la columna
for c in valores_perdidos[valores_perdidos > 0].index:
  data_significativa[c] = data_significativa[c].fillna(data_significativa[c].mean()).copy()

dimensiones = data_significativa.shape
print(f"El DataFrame tiene {dimensiones[0]} filas y {dimensiones[1]} columnas.")
# %%
# Proporción de transacciones fraudulentas
por_fraude = data_significativa['fraude'].value_counts(normalize=True)
print("Proporción de transacciones fraudulentas:",por_fraude)
print(data_significativa.head(5))

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

# Separar características y la variable objetivo
X = data_significativa.drop(columns=['fraude','monto_bin'])
y = data_significativa['fraude']

# Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear datasets balanceados
datasets = {}

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=data_significativa[['fraude','monto_bin']])

## Aumentar solo los datos de train
# 1. SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
datasets['SMOTE'] = (X_smote, y_smote)

# 2. ADASYN
adasyn = ADASYN(sampling_strategy=1.0, random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
datasets['ADASYN'] = (X_adasyn, y_adasyn)

# 3. Submuestreo de la clase mayoritaria
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X_train, y_train)
datasets['UnderSampling'] = (X_under, y_under)

# 4. Sin Sub Muestreo
datasets['NoSampling'] = (X_train, y_train)


# Mostrar el tamaño de cada dataset balanceado
for method, (X_res, y_res) in datasets.items():
    print(f"Dataset generado con {method}:")
    print(f"  Número de muestras: {X_res.shape[0]}")
    print(f"  Proporción de clases:\n{pd.Series(y_res).value_counts(normalize=True)}")
    print("-" * 50)


# %%

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Diccionario para almacenar resultados
results = []

# Función para entrenar y evaluar modelos
def train_evaluate_model(X_train, X_test, y_train, y_test, model, params, dataset, model_name):
    print(f"\nEntrenando y evaluando el modelo para el dataset: {dataset}")

    # Ajuste de hiperparámetros
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Reducir la validación cruzada a 3 folds
    grid_search = RandomizedSearchCV(pipeline, params, scoring='f1', cv=3, n_jobs=-1, n_iter=3, random_state=42)
    grid_search.fit(X_train, y_train)

    # Mejor modelo y métricas
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prs, rcls, _ = precision_recall_curve(y_test, y_score)
    prauc = auc(rcls, prs)

    # Guardar resultados en el diccionario
    results.append({
        "Dataset": dataset,
        "Model": model_name,
        "Best Params": grid_search.best_params_,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "F1-Score": f1,
        "Precision-Recall AUC": prauc
    })

    # Imprimir resultados
    print(f"Mejores Hiperparámetros: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision Recall AUC: {prauc:.4f}")
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
    return best_model

# Definir los modelos y sus hiperparámetros
## Incluir XGBoost y LightGBM
models_and_params = {
    "XGBoost": {
    "model": XGBClassifier(random_state=42, eval_metric='logloss'),
    "params": {
        "model__learning_rate": uniform(0.01, 0.1),
        "model__n_estimators": randint(80, 150),
        "model__max_depth": randint(3, 6),
        "model__colsample_bytree": uniform(0.6, 0.4)
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "model__learning_rate": uniform(0.01, 0.1),
            "model__n_estimators": randint(80, 150),
            "model__num_leaves": randint(30, 60)
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "model__C": uniform(0.1, 1),
            "model__penalty": ['l2']
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": randint(100, 110),
            "model__max_depth": randint(5, 7),
            "model__min_samples_split": randint(2, 5)
         }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__learning_rate": uniform(0.01, 0.1),
            "model__n_estimators": randint(50, 100),
            "model__max_depth": randint(3, 6)
        }
    }
}

# Entrenar y evaluar en los datasets generados
for dataset, (X_res, y_res) in datasets.items():
    print(f"\nData: {dataset}\n{'=' * 50}")
    for modelo, config in models_and_params.items():
        print(f"\nModelo: {modelo}")
        train_evaluate_model(X_res, X_test, y_res, y_test, config["model"], config["params"], dataset, modelo)

# Convertir los resultados en un DataFrame para guardar como CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_results_latinoamerica_one.csv", index=False)
print("\nResultados guardados en 'model_results_latinoamerica.csv'")


# %%

import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, precision_recall_curve, auc, precision_recall_fscore_support
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Obtener el dataset SMOTE
X_smote, y_smote = datasets['NoSampling']

# Entrenar el modelo LogisticRegression con los mejores hiperparámetros
best_model = XGBClassifier(model__colsample_bytree= 0.749816047538945, model__learning_rate= 0.10507143064099161, model__max_depth= 5, model__n_estimators= 140)

best_model.fit(X_smote, y_smote)

# Realizar predicciones
y_pred = best_model.predict(X_test)
y_score = best_model.predict_proba(X_test)[:, 1]

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label=1)
precision = precision_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
prs, rcls, _ = precision_recall_curve(y_test, y_score)
prauc = auc(rcls, prs)

# Mostrar resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
print(f"Precision Recall AUC: {prauc:.4f}")
# %%
# Lista de nombres de las columnas
feature_names = ['a', 'b', 'c', 'd', 'e', 'f', 'region_AR', 'region_BO', 'region_BR', 'region_BS',
'region_CL', 'region_CO', 'region_CR', 'region_DO', 'region_EC',
'region_HN', 'region_MX', 'region_NI', 'region_PA', 'region_PE',
'region_PR', 'region_PY', 'region_UY', 'h', 'k', 'l',
 'm', 'n', 'o_numerica', 'p_numerica','monto', 'score']

# Crear SHAP con TreeExplainer y habilitar la aproximación
# Usamos el modelo entrenado con LogisticRegression, pero LightGBM requiere TreeExplainer
# Para LogisticRegression no es aplicable. En su lugar, solo haz una visualización de coeficientes.

# Si quieres usar SHAP para LogisticRegression, deberías usar `KernelExplainer` o `LinearExplainer`
explainer = shap.KernelExplainer(best_model.predict_proba, X_smote[:100])  # Usar una muestra representativa para SHAP
shap_values = explainer.shap_values(X_test)

# Mostrar el gráfico de resumen de SHAP
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
plt.show()
# %%


def cost_gain_threshold_optimization(
        score: list, y_true: list, 
        weights: list = [],
        tp_gain: float = 1, fn_gain: float = 1,
        tn_gain: float = 1, fp_gain: float = 1
    ):
        
    thresholds = np.linspace(start=0.1,stop=0.9,num=98).round(2)
    res = []

    type_gain_mapping = {
        'tp':tp_gain,
        'fn':fn_gain,
        'tn':tn_gain,
        'fp':fp_gain
    }
    
    if len(weights)==0:
        weights = [1]*len(y_true)

    t_data = pd.DataFrame({
        "y_true":y_true,
        "score":score,
        "weight": weights
    })

    for thresh in thresholds:
        y_pred = np.where(score>thresh,1,0)
        t_data['y_pred'] = y_pred
        t_data['type'] = 'tp'
        t_data.loc[(t_data['y_true']==1) & (t_data['y_pred']==0), 'type'] = 'fn'
        t_data.loc[(t_data['y_true']==0) & (t_data['y_pred']==1), 'type'] = 'fp'
        t_data.loc[(t_data['y_true']==0) & (t_data['y_pred']==0), 'type'] = 'tn'
        t_data['type_gain'] = t_data['type'].map(type_gain_mapping)
        t_data['gain'] = t_data[['type_gain','weight']].apply(lambda x: x.prod(),axis=1)


        precision, recall, f1, _= precision_recall_fscore_support(
            y_true,y_pred,average = "binary",
            #zero_division=0
        )
        res.append(
            {
                'Precision':precision,
                'Recall':recall,
                'F1-score':f1,
                'Gain':t_data['gain'].sum(),
                'Threshold':thresh
            }
        )
    res = pd.DataFrame(res)
    
    best_score = res[res['Gain']==res['Gain'].max()]
    if best_score.shape[0]>0:
        best_score = best_score[best_score['F1-score']==best_score['F1-score'].max()]
    best_score = best_score.loc[best_score['Threshold'].idxmax(),:].to_dict()
    
    return res, best_score

score = best_model.predict_proba(X_test)[:,1]
thresh_optim_results, best_score = cost_gain_threshold_optimization(
    score=score, y_true=y_test,
    tp_gain=1 , fn_gain=-1 ,
    tn_gain=.25 , fp_gain=-.75 , weights= X_test[:,-2])
thresh_optim_results


def plot_thresh_optim_results(thresh_optim_results: pd.DataFrame, best_score: dict):

    
    fig, axes = plt.subplots(figsize = (9,5), nrows = 2, ncols = 1)
    axes[0].plot(thresh_optim_results['Threshold'].values,thresh_optim_results['Precision'].values, label = 'Precision')
    axes[0].plot(thresh_optim_results['Threshold'].values,thresh_optim_results['Recall'].values, label = 'Recall')
    axes[0].plot(thresh_optim_results['Threshold'].values,thresh_optim_results['F1-score'].values, label = 'F1-score')
    axes[0].axvline(x=best_score['Threshold'],linestyle = "--",color='black',linewidth=1)
    axes[0].legend()
    axes[1].plot(thresh_optim_results['Threshold'].values,thresh_optim_results['Gain'].values, label = 'Gain')
    axes[1].set_xlabel("Threshold")
    axes[1].axhline(y=best_score['Gain'],linestyle = "--",color='black',linewidth=1)
    axes[1].axvline(x=best_score['Threshold'],linestyle = "--",color='black',linewidth=1)

    ## Annotation box 
    offset = 72
    axes[1].annotate(
        text = "\n".join([f"{k}: {v:.2f}" for k,v in best_score.items()]),
        xy = (best_score['Threshold'],best_score['Gain']),
        xytext = (0.3*offset, -offset),
        textcoords='offset points',
        bbox = dict(boxstyle="round", fc="0.8"),
        arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10")
    )
    axes[1].legend()
    plt.tight_layout()
    
    return fig, axes
fig, axes = plot_thresh_optim_results(thresh_optim_results=thresh_optim_results, best_score=best_score)

plt.show()


# %%
# Identificar falsos positivos y falsos negativos
falsos_positivos = X_test[(y_test == 0) & (y_pred == 1)]
falsos_negativos = X_test[(y_test == 1) & (y_pred == 0)]


# Acceder a la columna 'monto' para falsos positivos y falsos negativos
falsos_positivos_monto = falsos_positivos[:, -2]
falsos_negativos_monto = falsos_negativos[:, -2]

# Calcular los costos de falsos positivos y falsos negativos
total_cost_falsos_negativos = falsos_negativos_monto.sum() 
total_cost_falsos_positivos = falsos_positivos_monto.sum()  

# Mostrar resultados
print(f"Falsos Positivos: {len(falsos_positivos)}")
print(f"Falsos Negativos: {len(falsos_negativos)}")
print(f"Costo total de falsos negativos: ${total_cost_falsos_negativos}")
print(f"Costo total de falsos positivos: ${total_cost_falsos_positivos}")
print(f"Costo total combinado: ${total_cost_falsos_negativos + total_cost_falsos_positivos}")
