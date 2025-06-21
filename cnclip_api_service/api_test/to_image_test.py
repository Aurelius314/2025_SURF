import requests  

# Input search text  
query_text = input("Please enter the text to search: ").strip()  

if query_text:  
    data = {  
        "query_text": query_text,  
        "api_key": "your_demo_api_key"  # Keep the API Key active  
    }  

    # Send a POST request to the server  
    response = requests.post('http://127.0.0.1:8000/text-to-image/', json=data)  
    
    # Check the return status and print the results  
    if response.status_code == 200:  
        result = response.json()  
        print("Most relevant image index:", result.get("most_similar_image_index"))  
        print("Most relevant image path:", result.get("most_similar_image_path"))  
        
        # Output Base64 encoded image content  
        img_base64 = result.get("most_similar_image_base64")  
        if img_base64:  
            print("Base64 encoding of the most relevant image has been generated and can be used for rendering.")  
        else:  
            print("Base64 encoding of the relevant image not found.")  
    else:  
        print("Request failed, error code:", response.status_code)  
        print("Error message:", response.text)  
else:  
    print("Please provide valid input.")