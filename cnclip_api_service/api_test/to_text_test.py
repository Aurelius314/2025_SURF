import requests  
import base64  
import os  

# image_path = r'C:\Users\Administrator\Downloads\1120.jpg'  
# C:\Users\Administrator\Downloads\1120.jpg  
image_path = input("Please enter the image path to convert: ").strip()  

# Check file type and extension  
if os.path.isfile(image_path):  
    with open(image_path, 'rb') as img_file:  
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')  

    data = {  
        "img_base64": img_base64,  
        "api_key": "your_demo_api_key"  # Keep the API Key active  
    }  

    response = requests.post('http://127.0.0.1:8000/image-to-text/', json=data)  
    print(response.json())  
else:  
    print("File not found. Please check the file path.")