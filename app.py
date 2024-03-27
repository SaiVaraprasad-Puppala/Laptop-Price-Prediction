import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and DataFrame
pipe = pickle.load(open('S:\Odin Final Projects\Laptop Price Predictor\pipe.pkl','rb'))
df = pickle.load(open('S:\Odin Final Projects\Laptop Price Predictor\df.pkl','rb'))

# Set the title of the Streamlit app
st.title("Laptop Price Predictor")

# Dropdown menu for selecting laptop brand
company = st.selectbox('Brand',df['Company'].unique())

# Dropdown menu for selecting laptop type
type = st.selectbox('Type',df['TypeName'].unique())

# Dropdown menu for selecting RAM size
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# Numeric input for laptop weight
weight = st.number_input('Weight of the Laptop')

# Dropdown menu for selecting touchscreen option
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# Dropdown menu for selecting IPS display option
ips = st.selectbox('IPS',['No','Yes'])

# Numeric input for screen size
screen_size = st.number_input('Screen Size')

# Dropdown menu for selecting screen resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# Dropdown menu for selecting CPU brand
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())

# Dropdown menu for selecting HDD size
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# Dropdown menu for selecting SSD size
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# Dropdown menu for selecting GPU brand
gpu = st.selectbox('GPU',df['Gpu Brand Name'].unique())

# Dropdown menu for selecting operating system
os = st.selectbox('OS',df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # Preprocess the user input and prepare it for prediction
    # Convert categorical variables to numerical representation
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    # Calculate PPI (Pixels Per Inch) based on screen resolution and size
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    if screen_size != 0:
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    else:
        ppi = 0

    # Prepare the input features as a numpy array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)  # Reshape the array to match the model input shape

    # Make the price prediction using the pre-trained model
    predicted_price = int(np.exp(pipe.predict(query)[0]))  # Predicted price after transforming it back from log scale

    # Display the predicted price
    st.title("The predicted price of this configuration is " + str(predicted_price))



# Define the trademark text
trademark_text = " This project was done by Sai Varaprasad Puppala."

# Add a text component with the trademark text
# Set the text alignment to 'center' horizontally, font size to a smaller value, and bottom margin
st.markdown(f'<p style="font-size: 10px; text-align: center; margin-bottom: 10px;">{trademark_text}</p>', unsafe_allow_html=True)
