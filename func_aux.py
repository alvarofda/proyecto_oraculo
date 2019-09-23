### --- ARCHIVOS FUNCIONES AUXILIARES --- ###

# Librerias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pylab import rcParams
from scipy import stats

# Funcion Correlacion entre atributos y vector objetivo

def fetch_features (df, vec_ob):
    columns = df.columns #extraemos los nombres de las columnas en la base de datos

    # generamos 3 arrays vacíos para guardar los valores
    attr_name = [] # nombre de la variable
    pearson_r = [] # correlación de pearson
    abs_pearson_r = [] # valor absoluto de la correlación

    # para cada columna en el array de columnas
    for col in columns:
        # si la columna no es la dependiente
        if col != vec_ob:
            # adjuntar el nombre de la variable en attr_name
            attr_name.append(col)
            # adjuntar la correlación de pearson
            pearson_r.append(df[col].corr(df[vec_ob]))
            # adjuntar el absoluto de la correlación de pearson
            abs_pearson_r.append(abs(df[col].corr(df[vec_ob])))
        
    # transformamos los arrays en un DataFrame
    features = pd.DataFrame({'attribute': attr_name,'corr':pearson_r,'abs_corr':abs_pearson_r})
    # generamos el index con los nombres de las variables
    features = features.set_index('attribute')
    # ordenamos los valores de forma descendiente
    feat_sort = features.sort_values(by=['abs_corr'], ascending=False)
   
    return feat_sort

# Función para la importancia de atributos

def plot_feature_importance(fit_model, feat_names):
    """
    Plot relative importance of a feature subset given a fitted model.
    """

    # Seteamos el tama{o de nuestro plot
    rcParams['figure.figsize'] = 10, 5
    # Guardamos las columnas de nuestro conjunto de entrenamiento
    features = feat_names
    # Obtenemos la importancia de nuestros atributos desde el modelo entrenado
    importances = fit_model.feature_importances_
    # Ordenamos de maoyr a menor la importancia de nustro atributos
    indices = np.argsort(importances)

    # Graficamos
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('porcentaje de importancia relativa')
    plt.show()

# Función para gráficar metricas de desempeño

def plot_metrics (metrics, models, title):

    length = len(metrics)
    x_labels = models

    # Set plot parameters
    rcParams['figure.figsize'] = 10, 6
    fig, ax = plt.subplots()
    width = 0.2 # width of bar
    x = np.arange(length)

    rects1 = ax.bar(x, metrics[:,0], width, color='#000080', label='MAE s/imputar')
    rects2 = ax.bar(x + width, metrics[:,1], width, color='#0F52BA', label='MAE imputados')
    rects3 = ax.bar(x + (2 * width), metrics[:,2], width, color='#6593F5', label='RMSE s/imputar')
    rects4 = ax.bar(x + (3 * width), metrics[:,3], width, color='#73C2FB', label='RMSE imputados')


    ax.set_ylabel('Minutos')
    ax.set_ylim(0,75)
    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Modelos')
    ax.set_title(title)
    ax.legend()
    plt.grid(True, 'major', 'y', ls='--', lw=.9, c='k', alpha=.3)
    fig.tight_layout()
    plt.show()



