# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:06:12 2024

@author: burakt
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

def main():
    st.sidebar.header('VALUES')  

    def user_input_features():
        relative_compactness = st.sidebar.slider('Relative Compactness', 0.764, 0.98, 0.73)
        wall_area = st.sidebar.slider('Wall Area', 240, 420, 315)
        roof_area = st.sidebar.slider('Roof Area', 100, 250, 159)
        overall_height = st.sidebar.slider('Overall Height', 3.5, 7.0, 3.5)
        glazing_area = st.sidebar.slider('Glazing Area', 0.0, 0.4, 0.2)
        glazing_area_distribution = st.sidebar.slider('Glazing Dist', 0, 5, 2)

        data = {'relative_compactness': relative_compactness,
                'wall_area': wall_area,
                'roof_area': roof_area,
                'overall_height': overall_height,
                'glazing_area': glazing_area,
                'glazing_area_distribution': glazing_area_distribution}

        features = pd.DataFrame(data, index=[0])
        return features

    def model_regression_chart(modelname, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
        if modelname == 'LinearRegression':
            model = LinearRegression()
        elif modelname == 'RandomForestRegressor':
            model = RandomForestRegressor()
        elif modelname == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, y_test

    st.title("Isıtma Yükü Tahmini")

    menu = ["Giriş", "Hakkımızda", "Analiz", "Isıtma Yükü Tahmini"]
    choice = st.sidebar.selectbox("Menü", menu)

    if choice == "Giriş":
        st.write("Yeni bir mi yapacaksınız?")
        st.write("Evi ısıtmak için ihtiyacınız olan yükü mü öğrenmek istiyorsunuz?")
        st.write("O zaman Data Boom Sunar.")
    elif choice == "Hakkımızda":
        st.write("Söz Tuana KURŞUN - Bilgisayar Mühendisi")
        st.write("Semih Furkan ÖCEK - İnşaat Mühendisi")
        st.write("Gizem YÖRÜR - Makine Mühendisi")
        st.write("Burak TAŞOVA - Elektrik Mühendisi")
    elif choice == "Analiz":
        st.write("Analiz sayfası içeriği buraya gelecek.")
    elif choice == "Isıtma Yükü Tahmini":
        df = user_input_features()

        ModelType = st.sidebar.selectbox('KURMAK ISTEDIGINIZ MODEL',
                                         ["LinearRegression", "RandomForestRegressor", "KNeighborsRegressor"])

        # Veri ve hedef değişkeni tanımlamak için örnek bir veri oluşturuyorum
        df = pd.read_csv("ENB2012_data.csv")
        df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation',
                'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
        df.drop(['surface_area'], axis=1, inplace=True)
        df.drop('orientation', axis=1, inplace=True)  
        X = df.drop(['heating_load', 'cooling_load'], axis=1)
        y = df['heating_load']

        # Tahmin ve grafik oluşturma işlemleri
        y_pred ,y_test= model_regression_chart(ModelType, X, y)

        # Hata hesaplama
        gercek = y_test
        tahmin = y_pred
        hata = gercek - tahmin
        df2 = user_input_features()
        y_pred2 = model.predict(df2)
        # Grafik oluşturma
        st.write(f"<h1 style='font-weight: bold; font-size: 29px;'>     {ModelType}</h1>", unsafe_allow_html=True)
        plt.figure(figsize=(8, 6))
        plt.scatter(gercek, tahmin, color='red', alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        plt.title('Hata Dağılım Grafiği')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.grid(True)
        st.pyplot()

        # Gerekli enerji miktarını yazdırma
        st.markdown(f"<h1 style='font-weight: bold; font-size: 24px;'>Gerekli enerji miktarı: {y_pred2}</h1>",
                    unsafe_allow_html=True)

if __name__ == "__main__":
    main()
