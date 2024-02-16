import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image

# Load your image
image = Image.open("tehran.jpg")

df = pd.read_csv("cleaned_data.csv")

adress = pd.read_csv("columns_dataframe.csv")

cat = Pipeline([("enc",OrdinalEncoder())])
one = Pipeline([("one",OneHotEncoder(handle_unknown="ignore"))])
scaler = Pipeline([("scaler",StandardScaler())])

transformer = ColumnTransformer([("cat",cat,["Parking","Warehouse","Elevator"]),
                        ("one",one,["Address"]),
                        ("scaler",scaler,["Area","Room"])])

ml_pipe = Pipeline([("preprocess",transformer),
                    ("xgb",GradientBoostingRegressor(learning_rate= 0.5,max_depth= 2,n_estimators= 200))])


X = df.drop("Price",axis="columns")
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

ml_pipe.fit(X_train,y_train)

st.title("""Tehran House Price Prediction App (2021) """)
st.write("""
* #### The Data has scraped from [Divar](https://divar.ir/) website
* #### The model has trained with *Graident Boosting* with *86 %* accuracy
* #### For more details check my [kaggle profile ](https://www.kaggle.com/code/mahdialfred/accuracy-85-xgboosting)

""")

st.image(image,use_column_width=True)

st.sidebar.title("User Input Features")

def user_input_features():

    area = st.sidebar.slider("Area",X.Area.min(),X["Area"].max(),int(X["Area"].mean()))
    room = st.sidebar.slider("Room",X.Room.min(),X.Room.max(),1)
    parking = st.sidebar.selectbox("Parking",[True,False])
    warehouse = st.sidebar.selectbox("Warehouse",[True,False])
    elevator = st.sidebar.selectbox("Elevator",[True,False])
    address = st.sidebar.selectbox("Address",adress)

    data = {
        "Area":area,
        "Room":room,
        "Parking":parking,
        "Warehouse":warehouse,
        "Elevator":elevator,
        "Address":address
    }
    input_features = pd.DataFrame(data,index=[0])
    return input_features

user_inpt = user_input_features()
st.write("---")
st.header("User input feature")
st.write(user_inpt)
st.write("---")

st.header("Predicted House Price")
st.write(pd.DataFrame(ml_pipe.predict(user_inpt).round(0),columns=["Price"]))

st.write("""
#### Trained and Developed my [Mahdi Zare](https://www.linkedin.com/in/mahdizare22/)
          """)
