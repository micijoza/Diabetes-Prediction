import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

file = './dataSets/pima-indians-diabetes.data.csv'
columns_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
classes = ['Negative Diabete', 'Positive Diabete']
data = pd.read_csv(file, names=columns_names)


def visualize_data():
    st.markdown('<p style="font-size:30px;color:#df1d4b;font-weight:bold">Visualisation des données de Diabète :</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Dimension :</p>', unsafe_allow_html=True)
    st.write(data.shape)
    st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Informations generales :</p>', unsafe_allow_html=True)
    st.write(data.describe())
    st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Histogramme :</p>', unsafe_allow_html=True)
    plt.figure(figsize=(12, 12))
    plt.rcParams.update({'font.size': 5})
    data.hist()
    st.pyplot(plt)
    st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Box and Whisker Plot :</p>', unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    data.plot(kind='box', subplots=True, sharex=False, layout=(3, 3))
    st.pyplot()
    st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Scatter Plots :</p>', unsafe_allow_html=True)
    plt.figure(figsize=(25, 25))
    scatter_matrix(data, color='r')
    st.pyplot()


def display_prediction_and_metrics(model, df, classes,selectedClassMetrics,X,Y):
    prediction = model.predict(df)
    color = "#3dfe00" if int(prediction) == 0 else "#df1d4b"
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Prediction :</p>', unsafe_allow_html=True)
        st.write(f'<p style="font-size:20px;color:{color};font-weight:bold">{classes[int(prediction)]}</p>', unsafe_allow_html=True)
    with col2:
        ClassMetr(model,selectedClassMetrics, X, Y)


def ClassMetr(model, selectedClassMetrics, X, Y):
    st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Les métriques :</p>', unsafe_allow_html=True)
    if selectedClassMetrics:
        if "La précision" in selectedClassMetrics:
            st.markdown('<p style="font-size:20px;color:#cbf2fa;font-weight:bold">Precision :</p>', unsafe_allow_html=True)
            prec = cross_val_score(model, X, Y, scoring="precision")
            st.write(prec.mean().round(2)*100.0)
        if "Le rappel" in selectedClassMetrics:
            st.markdown('<p style="font-size:20px;color:#cbf2fa;font-weight:bold">Recall :</p>', unsafe_allow_html=True)
            rec = cross_val_score(model, X, Y, scoring="recall")
            st.write((rec.mean() * 100.0).round(2))
        if "La F1-score" in selectedClassMetrics:
            st.markdown('<p style="font-size:20px;color:#cbf2fa;font-weight:bold">F1-Score :</p>', unsafe_allow_html=True)
            f1 = cross_val_score(model, X, Y, scoring="f1")
            st.write((f1.mean()*100.0).round(2))
        if "AUC-ROC" in selectedClassMetrics:
            st.markdown('<p style="font-size:20px;color:#cbf2fa;font-weight:bold">AUC-ROC :</p>', unsafe_allow_html=True)
            auc = cross_val_score(model, X, Y, scoring="roc_auc")
            st.write(auc.mean().round(2)*100.0)
    else:
        st.markdown('<p style="font-size:20px;color:#cbf2fa;font-weight:bold">Veuillez choisir une métrique pour mesurer la qualité des prédictions et à évaluer à quel point le modèle est performant.</p>', unsafe_allow_html=True)
    
def user_input_features():
    st.sidebar.markdown('<p style="font-size:16px;color:#df1d4b;font-weight:bold">Veuillez entrer vos valeurs :</p>', unsafe_allow_html=True)
    preg = st.sidebar.slider('Preg', 0.0, 17.0, 3.8)
    plas = st.sidebar.slider('Plas', 0.0, 199.0, 120.8)
    pres = st.sidebar.slider('Pres', 0.0, 122.0, 69.1)
    skin = st.sidebar.slider('Skin', 0.0, 99.0, 20.5)
    test = st.sidebar.slider('Test', 0.0, 846.0, 79.7)
    mass = st.sidebar.slider('Mass', 0.0, 67.1, 31.9)
    pedi = st.sidebar.slider('Pedi', 0.0, 2.4, 0.4)
    age = st.sidebar.slider('Age', 21, 81, 33)
    data = {
        'preg' : preg,
        'plas' : plas,
        'pres' : pres,
        'skin' : skin,
        'test' : test,
        'mass' : mass,
        'pedi' : pedi,
        'age' : age
    }
    features = pd.DataFrame(data, index=[0])
    return features


def main():
    st.write("""
        # Application de prédiction et de visualisation de cas de diabète
        Cette application vous offre la possibilité de prédire le cas du diabète (Classification(+/-)) 
        """
    )
    options = ["Les options", "Visualisation des données", "Prediction de cas de diabète"]
    st.markdown('<p style="font-size:16px;color:#df1d4b;font-weight:bold">Choisissez une option :</p>', unsafe_allow_html=True)
    selected_option = st.selectbox("", options)

    if selected_option != "Les options":
        if selected_option == "Visualisation des données":
            visualize_data()
        else:
            st.sidebar.header('Prediction : ')
            st.sidebar.markdown('<p style="font-size:16px;color:#df1d4b;font-weight:bold">Choisissez un algorithm :</p>', unsafe_allow_html=True)
            ClassAlgo = ["Les algorithms", "La régression logistique", "Les arbres de décision", "Les forêts aléatoires", "Les machines à vecteurs de support(SVM)"]
            RegAlgo = ["Les algorithms", "La régression linéaire","Le Lasso", "Les arbres de décision" ]
            algo = st.sidebar.selectbox("", ClassAlgo)
            ClassMetric = ["La précision", "Le rappel", "La F1-score", "AUC-ROC"]
            st.sidebar.markdown('<p style="font-size:16px;color:#df1d4b;font-weight:bold">Sélectionnez une ou plusieurs metrics :</p>', unsafe_allow_html=True)
            selectedClassMetrics = st.sidebar.multiselect("", ClassMetric)
            array = data.values
            X = array[ : ,0:-1]
            Y = array[ : , -1]
            if algo != "Les algorithms":
                if algo == "La régression logistique":
                    model = LogisticRegression(solver='newton-cg')
                elif algo == "Les arbres de décision":
                    model = DecisionTreeClassifier()
                elif algo == "Les forêts aléatoires":
                    model = RandomForestClassifier()
                else:
                    model = SVC()
                model.fit(X, Y)
                df = user_input_features()
                st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">User Input Parameteres :</p>', unsafe_allow_html=True)
                st.write(df)
                display_prediction_and_metrics(model, df, classes, selectedClassMetrics,X ,Y)
            else:
                st.markdown('<p style="font-size:25px;color:#dabfc5;font-weight:bold">Prediction :</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-size:20px;color:#cbf2fa;font-weight:bold">Veuillez choisir un algorithme pour prédire</p>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()