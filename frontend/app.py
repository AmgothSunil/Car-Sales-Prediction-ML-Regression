import streamlit as st
import requests

st.title("Car Price Prediction")
st.write("Enter car details to predict selling price")

# Input fields
symboling = st.number_input("Symboling", -3, 3, 0)
fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.selectbox("Aspiration", ["std", "turbo"])
doornumber = st.selectbox("Door Number", ["two", "four"])
carbody = st.selectbox("Car Body", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
enginelocation = st.selectbox("Engine Location", ["front", "rear"])
wheelbase = st.number_input("Wheel Base", 80.0, 120.0, 100.0)
carlength = st.number_input("Car Length", 140.0, 210.0, 170.0)
carwidth = st.number_input("Car Width", 60.0, 75.0, 65.0)
carheight = st.number_input("Car Height", 47.0, 60.0, 53.0)
curbweight = st.number_input("Curb Weight", 1500, 4500, 2500)
enginetype = st.selectbox("Engine Type", ["dohc", "ohcv", "ohc", "l", "rotor"])
cylindernumber = st.selectbox("Cylinder Number", ["two", "three", "four", "five", "six", "eight", "twelve"])
enginesize = st.number_input("Engine Size", 50, 400, 120)
fuelsystem = st.selectbox("Fuel System", ["mpfi", "2bbl", "mfi", "1bbl", "spfi", "4bbl", "idi", "spdi"])
boreratio = st.number_input("Bore Ratio", 2.0, 4.0, 3.0)
stroke = st.number_input("Stroke", 2.0, 5.0, 3.2)
compressionratio = st.number_input("Compression Ratio", 7.0, 23.0, 9.0)
horsepower = st.number_input("Horsepower", 50, 300, 100)
peakrpm = st.number_input("Peak RPM", 4000, 7000, 5000)
citympg = st.number_input("City MPG", 10, 50, 25)
highwaympg = st.number_input("Highway MPG", 10, 60, 30)

if st.button("Predict Price"):
    features = {
        "symboling": symboling,
        "fueltype": fueltype,
        "aspiration": aspiration,
        "doornumber": doornumber,
        "carbody": carbody,
        "drivewheel": drivewheel,
        "enginelocation": enginelocation,
        "wheelbase": wheelbase,
        "carlength": carlength,
        "carwidth": carwidth,
        "carheight": carheight,
        "curbweight": curbweight,
        "enginetype": enginetype,
        "cylindernumber": cylindernumber,
        "enginesize": enginesize,
        "fuelsystem": fuelsystem,
        "boreratio": boreratio,
        "stroke": stroke,
        "compressionratio": compressionratio,
        "horsepower": horsepower,
        "peakrpm": peakrpm,
        "citympg": citympg,
        "highwaympg": highwaympg
    }

    # Call FastAPI
    response = requests.post("https://car-sales-prediction-ml-regression.onrender.com/predict", json=features)

    if response.status_code == 200:
        price = response.json()["predicted_price"]
        st.success(f"💰 Predicted Car Price: ${price}")
    else:
        st.error("Error connecting to API")
