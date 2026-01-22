import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

# Define the model architecture
class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = HousePriceModel()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Load model and scaler
model, scaler = load_model_and_scaler()

# Streamlit UI
st.title("🏠 House Price Prediction")
st.markdown("Enter property details to predict the house price")

col1, col2 = st.columns(2)

with col1:
    median_income = st.number_input("Median Income (in $10K)", min_value=0.0, value=3.0, step=0.1)

with col2:
    avg_rooms = st.number_input("Average Rooms", min_value=0.0, value=5.0, step=0.1)

if st.button("🔍 Predict Price", use_container_width=True):
    # Normalize input
    input_data = np.array([[median_income, avg_rooms]])
    input_normalized = scaler.transform(input_data)
    
    # Make prediction
    input_tensor = torch.FloatTensor(input_normalized)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    # Display result
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"💰 **Estimated Price: ${prediction*1000:.2f}**")
    
    with col2:
        st.metric("Price (in $1000s)", f"${prediction:.2f}")
