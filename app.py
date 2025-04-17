from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Configuração do cliente OpenAI para a API da NVIDIA
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plan', methods=['POST'])
def generate_plan():
    data = request.json
    user_input = data.get('input', '')
    
    if not user_input:
        return jsonify({'error': 'Nenhuma entrada fornecida'}), 400
    
    try:
        # Personalize o prompt para planejamento diário
        messages = [
            {
                "role": "system",
                "content": "Você é um assistente de planejamento diário especializado. Ajude o usuário a criar um plano eficiente para o dia, considerando produtividade, equilíbrio vida-trabalho e metas pessoais. Seja conciso e organizado."
            },
            {
                "role": "user",
                "content": f"Me ajude a planejar meu dia com base no seguinte: {user_input}"
            }
        ]
        
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=messages,
            temperature=0.6,
            top_p=0.7,
            max_tokens=1024,
            stream=False
        )
        
        response = completion.choices[0].message.content
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
