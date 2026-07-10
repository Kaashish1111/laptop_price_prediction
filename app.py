import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("elastic_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
columns = pickle.load(open("model_columns.pkl","rb"))

st.title("AI Laptop Price Predictor")

# ---------------- Sidebar ---------------- #

st.sidebar.title("Laptop Price Predictor")

st.sidebar.markdown("---")

st.sidebar.header("About")

st.sidebar.write("""
This application predicts the price of a laptop based on its specifications using a Machine Learning model.
""")

st.sidebar.markdown("---")

st.sidebar.header("Model")

st.sidebar.write("""
- **Algorithm:** ElasticNetCV Regressor
- **Framework:** Scikit-Learn
- **Features:** 119
""")

st.sidebar.markdown("---")

st.sidebar.header("Dataset")

st.sidebar.write("""
- **Samples:** 893 Laptops
- **Target:** Laptop Price
- **Task:** Regression
""")

st.sidebar.markdown("---")

st.sidebar.header("Developed By")

st.sidebar.write("**Kashish Goel**")

st.sidebar.markdown("---")

st.sidebar.success("Built with Streamlit")

brand = st.selectbox(
    "Brand",
    ["HP","Dell","Lenovo","Asus","Acer","Apple","MSI"]
)

ram = st.selectbox("RAM",[4,8,16,32,64])

rom = st.selectbox("Storage",[256,512,1024,2048])

display = st.slider("Display Size",13.0,18.0,15.6)

spec = st.slider("Spec Rating",50,100,80)

warranty = st.selectbox("Warranty",[1,2,3])

processor_brand = st.selectbox(
    "Processor Brand",
    ["Intel","AMD","Apple","MediaTek"]
)

processor_family = st.selectbox(
    "Processor Family",
    ["Core i3","Core i5","Core i7","Core i9",
     "Ryzen 3","Ryzen 5","Ryzen 7","Ryzen 9",
     "M1","M2","Athlon","Celeron"]
)

generation = st.slider("Generation",3,13,12)

core_count = st.selectbox(
    "CPU Cores",
    [2,4,6,8,10,12,14]
)

thread_count = st.selectbox(
    "Threads",
    [2,4,8,12,16,20,24]
)

gpu_brand = st.selectbox(
    "GPU Brand",
    ["Intel","AMD","NVIDIA"]
)

vram = st.selectbox(
    "VRAM",
    [0,2,4,6,8,12,16]
)

dedicated_gpu = st.selectbox(
    "Dedicated GPU",
    [0,1]
)

ram_type = st.selectbox(
    "RAM Type",
    ["DDR4","DDR5","LPDDR5","LPDDR4X"]
)

rom_type = st.selectbox(
    "Storage Type",
    ["SSD","Hard-Disk"]
)

os = st.selectbox(
    "Operating System",
    ["Windows 11 OS","Windows 10 OS","DOS OS","Mac OS","Ubuntu OS"]
)

laptop_category = st.selectbox(
    "Laptop Category",
    ["Normal","Gaming","Business","Convertible","Chromebook","MacBook"]
)

pixels = st.selectbox(
    "Resolution",
    [2073600,2304000,3686400]
)
import numpy as np

if st.button("🔮 Predict Price"):

    input_df = pd.DataFrame(0.0, index=[0], columns=columns)

    # Numerical features
    input_df.loc[0, "Ram"] = ram
    input_df.loc[0, "ROM"] = rom
    input_df.loc[0, "display_size"] = display
    input_df.loc[0, "spec_rating"] = spec
    input_df.loc[0, "warranty"] = warranty
    input_df.loc[0, "generation"] = generation
    input_df.loc[0, "core_count"] = core_count
    input_df.loc[0, "thread_count"] = thread_count
    input_df.loc[0, "vram"] = vram
    input_df.loc[0, "dedicated_gpu"] = dedicated_gpu
    input_df.loc[0, "pixels"] = pixels

    # One-hot encoded features
    categorical = {
        "brand": brand,
        "Ram_type": ram_type,
        "ROM_type": rom_type,
        "OS": os,
        "processor_brand": processor_brand,
        "processor_family": processor_family,
        "gpu_brand": gpu_brand,
        "laptop_category": laptop_category
    }

    for prefix, value in categorical.items():
        col = f"{prefix}_{value}"
        if col in input_df.columns:
            input_df.loc[0, col] = 1

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Convert back if you trained on log(price)
    prediction = np.expm1(prediction)

    st.success(f"Estimated Laptop Price: ₹{prediction[0]:,.0f}")

    st.balloons()