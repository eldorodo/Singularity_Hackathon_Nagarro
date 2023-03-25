from flask import Flask, request
from test import Chatbot
CONFIG_PATH = "."
chatbot = Chatbot(CONFIG_PATH)


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def predict_T5():
    data = request.get_json()
    usr_input = data['usr_input']
    predict_sentence = chatbot.predict(usr_input)   
    return predict_sentence



if __name__ == '__main__':
    app.run()