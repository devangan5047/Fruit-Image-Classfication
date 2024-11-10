# app.py
import os
import requests
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import logging
from PIL import Image
import io

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']


load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

SYSTEM_PROMPT = """strictly answer in 2 words not more than that. """
MAX_PIXELS = 1700000  

def compress_image(image_data):
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    width, height = image.size
    num_pixels = width * height
    
    if num_pixels > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / num_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)  
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            app.logger.error("Request data is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400 

        data = request.get_json() 
        user_message = data.get('message', '').strip()
        image = data.get('image')

        app.logger.info(f"Received request. Image: {'Yes' if image else 'No'}")

        if image:
            try:
                processed_image = compress_image(image)
                
                # For image inputs, incorporate system prompt into the user message
                enhanced_user_message = f"{SYSTEM_PROMPT}\n\njust give the number of fruits or vegetables in the image and which image or vegetable in the image. strictly give the response in not more than 2 words"
                
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{processed_image}"
                            }
                        },
                        {"type": "text", "text": enhanced_user_message}
                    ]
                }]

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEEPINFRA_API_KEY}"
                }

                payload = {
                    "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                    "messages": messages,
                    "temperature": 0.4,
                    "max_tokens": 400
                }

                app.logger.info(f"Sending request to DeepInfra API. Payload: {payload}")
                response = requests.post(DEEPINFRA_API_URL, headers=headers, json=payload, timeout=30)

                if response.status_code == 200:
                    app.logger.info(f"Received response from DeepInfra API: {response.json()}")
                    generated_text = response.json()['choices'][0]['message']['content']
                    
                    return jsonify({
                        "output": generated_text.strip()
                    })
                else:
                    app.logger.error(f"Error from DeepInfra API. Status code: {response.status_code}, Response: {response.text}")
                    return jsonify({"error": f"Failed to get response from the model. Status code: {response.status_code}"}), 500
            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                return jsonify({"error": f"Error processing image: {str(e)}"}), 400
        else:
            return jsonify({"error": "No image provided"}), 400

    except requests.Timeout:
        app.logger.error("Request to DeepInfra API timed out")
        return jsonify({"error": "Request to AI model timed out. Please try again."}), 504
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
