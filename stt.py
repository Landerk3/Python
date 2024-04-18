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
        return y_pred, y_test,model

    st.title("Isıtma Yükü Tahmini")

    menu = ["Giriş", "Hakkımızda", "Isıtma Yükü Tahmini"]
    choice = st.sidebar.selectbox("Menü", menu)

    if choice == "Giriş":
        st.write("· Yeni bir ev mi yapacaksınız?")
        st.write("· Evi ısıtmak için ihtiyacınız olan yükü öğrenmek mi istiyorsunuz?")
        st.write("· Isıtma Yük tahmini hesabını yapın!")
        image_path = "image.jpg"
        #st.markdown("<div style='float: right;'><img src='image.jpg' alt='Isıtma Yükü Tahmini' width='400'></div>",unsafe_allow_html=True)
        st.image(image_path, caption='Data Boom Boom Logo',use_column_width = True)
    elif choice == "Hakkımızda":
        st.write("Söz Tuana KURŞUN - Bilgisayar Mühendisi")
        st.write("Semih Furkan ÖCEK - İnşaat Mühendisi")
        st.write("Gizem YÖRÜR - Makine Mühendisi")
        st.write("Burak TAŞOVA - Elektrik Mühendisi")
    elif choice == "Isıtma Yükü Tahmini":
        st.sidebar.header('VALUES')
        df2 = user_input_features()

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
        y_pred ,y_test,model= model_regression_chart(ModelType, X, y)

        # Hata hesaplama
        gercek = y_test
        tahmin = y_pred
        hata = gercek - tahmin
        y_pred2 = model.predict(df2)[0]
        # Grafik oluşturma
        col6, col5,col4 = st.columns(3)
        with col6:
            st.write(f"<h1 style='font-weight: bold; font-size: 27px;'> Seçilen Model:</h1>", unsafe_allow_html=True)
        with col5:
            st.write(f"<h1 style='font-weight: bold; font-size: 29px;'>     {ModelType}</h1>", unsafe_allow_html=True)
        with col4:
            st.write(" ", unsafe_allow_html=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(gercek, tahmin, color='red', alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        plt.title('Hata Dağılım Grafiği')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.grid(True)
        st.pyplot()

        # Gerekli enerji miktarını yazdırma
        col1, col2,col13 = st.columns(3)
        with col1:
            st.write(f"<h1 style='font-weight: bold; font-size: 24px;'> Gerekli enerji:</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h1 style='font-weight: bold; font-size: 24px;'> {round(y_pred2,4)} kWh</h1>",unsafe_allow_html=True)
        with col13:
            st.write(" ", unsafe_allow_html=True)
        #st.markdown(f"<h1 style='font-weight: bold; font-size: 24px;'>Gerekli enerji miktarı: {round(y_pred2,4)} kWh</h1>",
              #  unsafe_allow_html=True)

if __name__ == "__main__":
    main()
