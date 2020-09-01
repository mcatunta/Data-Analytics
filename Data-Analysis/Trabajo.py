# =============================================================================
#                   CREDIT DATA ANALYSIS
# =============================================================================

# Autor: Marcos Catunta Cachi

# Se realizará un análisis sobre una base de datos crediticia de prueba para encontrar el mejor
# modelo de predicción e identificar el poder predictivo de las variables sobre el modelo. 

# Configuracion Inicial
#######################

# Importando librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from pathlib import Path

# Definiendo ruta
path = 'Path Data File'

# Importando data
data = pd.read_csv(path+'data.csv')

# Separando en X e y
data.columns
X=data.loc[:,['atraso','vivienda','edad','dias_lab','exp_sf','nivel_ahorro','ingreso','linea_sf',
              'deuda_sf','zona','clasif_sbs','nivel_educ']]
y=data.loc[:,['mora']]

# Separando data en train y test
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=4)

# Procesamiento de los datos
############################
Xtrain.describe()

# Funciones y ayudas
####################    
def histograma(datos,titulo):
    plt.hist(datos)
    plt.title(titulo)
    plt.show()

# Tratamiento de missings
#########################

# Cuantificando
np.sum(Xtrain.isnull()) # Absoluto
np.mean(Xtrain.isnull()) # Relativo
pd.concat([pd.DataFrame(np.sum(X.isnull())),pd.DataFrame(np.mean(X.isnull()))],axis=1)

# Asignando missings
Xtrain.exp_sf.fillna(np.mean(Xtrain.exp_sf),inplace=True)
Xtrain.linea_sf.fillna(0,inplace=True)
Xtrain['tuvo_deuda']=np.where(Xtrain['deuda_sf']>=0,1,0)
Xtrain.deuda_sf.fillna(0,inplace=True)

# Valores extremos
##################

# Ayuda para determinar el valor de las cotas
# histograma(Xtrain.atraso,'Atraso')
# Xtrain.groupby('atraso').size()
# np.percentile(Xtrain.atraso,range(101))

# Calculando umbrales a aplicar (cotas)
cotas_ingreso=np.nanpercentile(Xtrain.ingreso,[1,98])
cotas_linea_sf=np.nanpercentile(Xtrain.linea_sf,[97])
cotas_deuda_sf=np.nanpercentile(Xtrain.deuda_sf,[97])

# Asignando cotas
Xtrain.loc[Xtrain.atraso>=114,'atraso']=114
Xtrain.loc[Xtrain.edad>=65,'edad']=65
Xtrain.loc[Xtrain.ingreso>=cotas_ingreso[1],'ingreso']=cotas_ingreso[1]
Xtrain.loc[Xtrain.ingreso<=cotas_ingreso[0],'ingreso']=cotas_ingreso[0]
Xtrain.loc[Xtrain.linea_sf>=cotas_linea_sf[0],'linea_sf']=cotas_linea_sf[0]
Xtrain.loc[Xtrain.deuda_sf>=cotas_deuda_sf[0],'deuda_sf']=cotas_deuda_sf[0]


# Tratamiento de variables cualitativas
#######################################

# Uniendo bases X e y
data_train=pd.concat([Xtrain,ytrain],axis=1)

# Funcion: Information Value
def iv(variable):
    base=pd.DataFrame(data_train.groupby([variable])['mora'].agg({
        'Malos':'sum','Total':'count','RD':'mean'}))
    base['Buenos']=base.Total-base.Malos
    base['OR']=base.Buenos/base.Malos
    base['WOE']=np.log((base.Buenos/np.sum(base.Buenos))/(base.Malos/np.sum(base.Malos)))
    base['Diff']=(base.Buenos/np.sum(base.Buenos))-(base.Malos/np.sum(base.Malos))
    print('El IV de la variable',variable,'es:',np.sum(base.WOE*base.Diff))

# Calculando IV en variables cualitativas
iv('vivienda')      # Poder predictivo debil
iv('zona')          # Poder predictivo mediano
iv('nivel_educ')    # Poder predictivo mediano
iv('clasif_sbs')    # Poder predictivo debil

# Vivienda -> Dummies
Xtrain.vivienda.value_counts()
dummy_vivienda=pd.get_dummies(Xtrain.vivienda,prefix='d')
Xtrain=pd.concat([Xtrain,dummy_vivienda],axis=1)
# Zona -> Agrupación
Xtrain.zona.value_counts()
Xtrain['zona_f']=np.where(Xtrain.zona=='Lima',1,0)
# Clasificacion SBS -> Dummies
Xtrain.clasif_sbs.value_counts()
Xtrain['clasif_normal']=np.where(Xtrain.clasif_sbs==0,'Normal',
      np.where(Xtrain.clasif_sbs==1, 'Problemas',
               np.where(Xtrain.clasif_sbs==2,'Deficiente',
                        np.where(Xtrain.clasif_sbs==3,'Dudosa','Perdida'))))
dummy_clasif_sbs=pd.get_dummies(Xtrain.clasif_normal,prefix='d_sbs')
Xtrain=pd.concat([Xtrain,dummy_clasif_sbs],axis=1)
# Nivel educativo -> Dummies
Xtrain.nivel_educ.value_counts()
dummy_nivel=pd.get_dummies(Xtrain.nivel_educ,prefix='d')
Xtrain=pd.concat([Xtrain,dummy_nivel],axis=1)

# Dropeando variables de Xtrain
Xtrain.drop(['vivienda','nivel_educ','zona','clasif_sbs','clasif_normal'],axis=1,inplace=True)


# Aplicar todas las lógicas en test
############################

# Tratamiento de missings
#########################

# Asignando missings
Xtest.exp_sf.fillna(np.mean(Xtrain.exp_sf),inplace=True)
Xtest.linea_sf.fillna(0,inplace=True)
Xtest['tuvo_deuda']=np.where(Xtest['deuda_sf']>=0,1,0)
Xtest.deuda_sf.fillna(0,inplace=True)

# Valores extremos
##################

# Asignando cotas
Xtest.loc[Xtest.atraso>=114,'atraso']=114
Xtest.loc[Xtest.edad>=65,'edad']=65
Xtest.loc[Xtest.ingreso>=cotas_ingreso[1],'ingreso']=cotas_ingreso[1]
Xtest.loc[Xtest.ingreso<=cotas_ingreso[0],'ingreso']=cotas_ingreso[0]
Xtest.loc[Xtest.linea_sf>=cotas_linea_sf[0],'linea_sf']=cotas_linea_sf[0]
Xtest.loc[Xtest.deuda_sf>=cotas_deuda_sf[0],'deuda_sf']=cotas_deuda_sf[0]

# Tratamiento de variables cualitativas
#######################################

# Vivienda -> Dummies
dummy_vivienda=pd.get_dummies(Xtest.vivienda,prefix='d')
Xtest=pd.concat([Xtest,dummy_vivienda],axis=1)
# Zona -> Agrupación
Xtest['zona_f']=np.where(Xtest.zona=='Lima',1,0)
# Clasificacion SBS -> Dummies
Xtest['clasif_normal']=np.where(Xtest.clasif_sbs==0,'Normal',
      np.where(Xtest.clasif_sbs==1, 'Problemas',
               np.where(Xtest.clasif_sbs==2,'Deficiente',
                        np.where(Xtest.clasif_sbs==3,'Dudosa','Perdida'))))
dummy_clasif_sbs=pd.get_dummies(Xtest.clasif_normal,prefix='d_sbs')
Xtest=pd.concat([Xtest,dummy_clasif_sbs],axis=1)
# Nivel educativo -> Dummies
dummy_nivel=pd.get_dummies(Xtest.nivel_educ,prefix='d')
Xtest=pd.concat([Xtest,dummy_nivel],axis=1)

# Dropeando variables de Xtest
Xtest.drop(['vivienda','nivel_educ','zona','clasif_sbs','clasif_normal'],axis=1,inplace=True)


# Modelamiento de datos
#######################
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

columns_model=['Modelo','Score_train','Score_test','Gini_train','Gini_test',
               'Tasa_asierto_train','Tasa_asiento_test','Sensibilidad_train',
               'Sensibilidad_test','Especificidad_train','Especificidad_test']
resultado_modelo=pd.DataFrame(columns=columns_model)

#Funciones
##########
def modelar(nombre,modelo_base,clasifica,gscv):
    if (gscv):
        max_depth_space=np.linspace(5,15,3,dtype=int)
        max_leaf_nodes_space=[2,3,4]
        param_grid={'max_depth':max_depth_space,'max_leaf_nodes':max_leaf_nodes_space}
        modelo=GridSearchCV(modelo_base,param_grid,cv=5)
    else:
        modelo=modelo_base
    modelo.fit(Xtrain,ytrain.mora)
    score_train=modelo.score(Xtrain,ytrain.mora)
    score_test=modelo.score(Xtest,ytest.mora)
    predict_Xtrain=modelo.predict(Xtrain)
    predict_Xtest=modelo.predict(Xtest)
    if (clasifica):
        gini_train=0
        gini_test=0
        cm_train=confusion_matrix(ytrain.mora,predict_Xtrain)
        ta_train=(cm_train[0,0]+cm_train[1,1])/np.sum(cm_train)
        sensi_train=cm_train[1,1]/np.sum(cm_train[1,:])
        espec_train=cm_train[0,0]/np.sum(cm_train[0,:])
        cm_test=confusion_matrix(ytest.mora,predict_Xtest)
        ta_test=(cm_test[0,0]+cm_test[1,1])/np.sum(cm_test)
        sensi_test=cm_test[1,1]/np.sum(cm_test[1,:])
        espec_test=cm_test[0,0]/np.sum(cm_test[0,:])
    else:        
        gini_train=2*roc_auc_score(ytrain.mora,predict_Xtrain)-1
        gini_test=2*roc_auc_score(ytest.mora,predict_Xtest)-1
        ta_train=0
        sensi_train=0
        espec_train=0
        ta_test=0
        sensi_test=0
        espec_test=0
    return pd.DataFrame([[nombre,score_train,score_test,gini_train,gini_test,
                          ta_train,ta_test,sensi_train,sensi_test,espec_train,espec_test]],
                 columns=columns_model)

# Regresion Logistica
from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()
resultado_modelo=resultado_modelo.append(modelar('Regresion Logistica',reg_log,True,False))

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nbayes=MultinomialNB()
resultado_modelo=resultado_modelo.append(modelar('Naive Bayes',nbayes,False,False))

# SVM: Regresion
from sklearn.svm import SVR
svr=SVR()
resultado_modelo=resultado_modelo.append(modelar('SVM Regresion',svr,False,False))

# SVM: Clasificacion
from sklearn.svm import SVC
svc=SVC()
resultado_modelo=resultado_modelo.append(modelar('SVM Clasificacion',svc,True,False))

# Arbol de Regresion
from sklearn.tree import DecisionTreeRegressor
arbol_reg=DecisionTreeRegressor()
resultado_modelo=resultado_modelo.append(modelar('Arbol Regresion',arbol_reg,False,True))

# Arbol de Clasificacion
from sklearn.tree import DecisionTreeClassifier
arbol_cl=DecisionTreeClassifier()
resultado_modelo=resultado_modelo.append(modelar('Arbol Clasificacion',arbol_cl,True,True))

# Random Forest Regresion
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
resultado_modelo=resultado_modelo.append(modelar('Random Forest Regresion',rf_reg,False,True))

# Random Forest Clasificacion
from sklearn.ensemble import RandomForestClassifier
rf_cl=RandomForestClassifier(n_estimators=10,random_state=0)
resultado_modelo=resultado_modelo.append(modelar('Random Forest Clasificacion',rf_cl,True,True))

# XGBoost Regresion
import xgboost as xgb
xgb_reg=xgb.XGBRegressor(objetive='binary:logistic',n_estimators=10,seed=0)
resultado_modelo=resultado_modelo.append(modelar('XGBoost Regresion',xgb_reg,False,True))

# XGBoost Clasificacion
import xgboost as xgb
xgb_cl=xgb.XGBClassifier(objetive='binary:logistic',n_estimators=10,seed=0)
resultado_modelo=resultado_modelo.append(modelar('XGBoost Clasificacion',xgb_cl,True,True))


# Mejor Modelo: XGBoost Clasificacion
#####################################
xgb_cl=xgb.XGBClassifier(objetive='binary:logistic',n_estimators=10,seed=0)
max_depth_space=np.linspace(5,15,3,dtype=int)
max_leaf_nodes_space=[2,3,4]
param_grid={'max_depth':max_depth_space,'max_leaf_nodes':max_leaf_nodes_space}
modelo=GridSearchCV(xgb_cl,param_grid,cv=5)
modelo.fit(Xtrain,ytrain.mora)

# Modelo con parametros optimizados
xgb_cl=xgb.XGBClassifier(objetive='binary:logistic',max_depth=modelo.best_params_['max_depth'],
                         max_leaf_nodes=modelo.best_params_['max_leaf_nodes'],n_estimators=10,seed=4)
xgb_cl.fit(Xtrain,ytrain.mora)
# Information Gain
importancia=pd.DataFrame(xgb_cl.feature_importances_)
importancia.index=Xtrain.columns
