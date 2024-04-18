import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

import streamlit as st
st.write("<h1 style='font-weight: bold; font-size: 29px;'>Isıtma Soğutma Yük Tahmini</h1>", unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv("ENB2012_data.csv")
#print(df.head())
#sütun isimlerini değiştirdik.
df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height', 'orientation',
                'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
df.corr()['cooling_load'].sort_values()[:-1].plot.bar()
df.corr()['heating_load'].sort_values()[:-1].plot.bar()
correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
#sns.heatmap(correlation_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix of Features')
plt.show()

import streamlit as st

def main():
    st.title("İzmir Data: Veri Bilimi Yolculuğu")
    
    menu = ["Giriş", "Hakkımızda", "Analiz", "Isıtma Yükü Tahmini"]
    choice = st.sidebar.selectbox("Menü", menu)

    if choice == "Home":
        st.write("Yeni bir mi yapacaksınız?")
        st.write("Evi ısıtmak için ihtiyacınız olan yükü mü öğrenmek istiyorsun ?")
        st.write("O zaman Data Boom Sunar.")
    elif choice == "Hakkımızda":
        st.write(" Söz Tuana KURŞUN - Bilgisayar Mühendisi ")
        st.write(" Semih Furkan ÖCEK - İnşaat Mühendisi ")
        st.write(" Gizem YÖRÜR - Makine Mühendisi ")
        st.write(" Burak TAŞOVA - Elektrik Mühendisi ")
    elif choice == "Analiz":
        st.write("Analiz sayfası içeriği buraya gelecek.")
    elif choice == "Maaş Tahmini":
        st.sidebar.header('VALUES')  
        relative_compactness = st.sidebar.slider('Relative Compactness',0.764,0.98,0.73)
        wall_area = st.sidebar.slider('Wall Area',240,420,315)
        roof_area = st.sidebar.slider('Roof Area',100,250,159)
        overall_height = st.sidebar.slider('Overall Height',3.5,7.0,3.5)
        glazing_area = st.sidebar.slider('Glazing Area',0.0,0.4,0.2)
        glazing_area_distribution = st.sidebar.slider('Glazing Dist',0,5,2)
       
        

if __name__ == "__main__":
    main()






#tekrarı önlemek için x1 ve x2 den birini kaldırmamız gerekiyordu. X2 diğer featurelar ile daha ilişkili olduğundan x1 i kaldırdım.
df.drop(['surface_area'], axis=1, inplace=True)
df.drop('orientation', axis=1, inplace=True)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any()
#outliers
#there are no outliers

num_columns = df.shape[1]
num_rows = num_columns // 2 if num_columns % 2 == 0 else (num_columns // 2) + 1
plt.figure(figsize=(15, num_rows * 5))


for i, col in enumerate(df.columns):
    plt.subplot(num_rows, 2, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()


df.corr()['cooling_load'].sort_values()[:-1].plot.bar()

df.corr()['heating_load'].sort_values()[:-1].plot.bar()



correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Matrix of Features')
plt.show()

#df.drop(['relative_compactness'], axis=1, inplace=True)

#df.drop('orientation', axis=1, inplace=True)



X = df.drop(['heating_load', 'cooling_load'], axis=1)
y = df['heating_load']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21)



# def train_models(X_train, X_test, y_train, y_test):
#
#
#     models = [
#         ("Linear Regression", LinearRegression()),
#         ("Random Forest", RandomForestRegressor()),
#         ("SVM", SVR()),
#         ("K-Nearest Neighbors", KNeighborsRegressor())
#     ]
#
#
#     results = []
#
#     for model_name, model in models:
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         results.append({
#             'Model': model_name,
#             'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
#             'Root Mean Squared Error': mean_squared_error(y_test, y_pred, squared=False),
#             'Mean Squared Error': mean_squared_error(y_test, y_pred),
#             'R-squared (R2)': r2_score(y_test, y_pred)
#         })
#
#     return pd.DataFrame(results)




# model = RandomForestRegressor()
# model.fit(X,y)
# print(model.feature_importances_)
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.plot(kind='bar')
# plt.show()



st.sidebar.header('VALUES')  
def user_input_features():
    relative_compactness = st.sidebar.slider('Relative Compactness',0.764,0.98,0.73)
    wall_area = st.sidebar.slider('Wall Area',240,420,315)
    roof_area = st.sidebar.slider('Roof Area',100,250,159)
    overall_height = st.sidebar.slider('Overall Height',3.5,7.0,3.5)
    glazing_area = st.sidebar.slider('Glazing Area',0.0,0.4,0.2)
    glazing_area_distribution = st.sidebar.slider('Glazing Dist',0,5,2)

    data = {'relative_compactness' : relative_compactness,
            'wall_area' : wall_area,
            'roof_area' :roof_area,
            'overall_height' :overall_height,
            'glazing_area' :glazing_area,
            'glazing_area_distribution' :glazing_area_distribution}

    features = pd.DataFrame(data,index=[0])
    return features
ModelType = st.sidebar.selectbox('KURMAK ISTEDIGINIZ MODEL',["LinearRegression","RandomForestRegressor","KNeighborsRegressor"])

#LinearRegression
#RandomForestRegressor
#SVR
#KNeighborsRegressor

def model_regression_chart(modelname):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
    global model
    if modelname == 'LinearRegression':
            model = LinearRegression()
    if modelname == 'RandomForestRegressor':
            model = RandomForestRegressor()
    if modelname == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred

# Test verileri üzerinde tahmin yapma
a = model_regression_chart(ModelType)
# Gerçek ve tahmini değerler
gercek = y_test
tahmin = a

# Hata hesaplama
hata = gercek - tahmin
col1, col2,col13 = st.columns(3)
with col1:
    st.write(" ", unsafe_allow_html=True)
with col2:
    st.write(f"<h1 style='font-weight: bold; font-size: 29px;'>     {ModelType}</h1>", unsafe_allow_html=True)
with col13:
    st.write(" ", unsafe_allow_html=True)
# Hata dağılım grafiği
plt.figure(figsize=(8, 6))
plt.scatter(gercek, tahmin, color='red', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
#plt.axhline(, color='black', linestyle='--')
plt.title('Hata Dağılım Grafiği')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.grid(True)
st.pyplot()

#df2 = user_input_features()
#st.write(df2)
y_pred2 = model.predict(df2)
col3, col4,col5 = st.columns(3)
with col3:
    st.markdown(f"<h1 style='font-weight: bold; font-size: 24px;'>Gerekli enerji miktarı </h1>", unsafe_allow_html=True)
with col4:
    st.write(f"<h1 style='font-weight: bold; font-size: 24px;'>{y_pred2}</h1>", unsafe_allow_html=True)
with col5:
    st.write(f"<h1 style='font-weight: bold; font-size: 24px;'>kWh</h1>", unsafe_allow_html=True)

#st.write(y_pred2)
#plt.figure(figsize=(8, 6))
#sns.heatmap(correlation_matrix, annot=True,fmt=".2f", cmap="coolwarm", cbar=True)
#st.pyplot()

