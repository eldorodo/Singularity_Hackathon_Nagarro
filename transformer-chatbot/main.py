from test import Chatbot
CONFIG_PATH = "."
chatbot = Chatbot(CONFIG_PATH)

def predict_T5(usr_input):
    predict_sentence = chatbot.predict(usr_input)
    print(predict_sentence)
    return predict_sentence

usr_input = 'Hi Iam feeling quite down today'
# for i in range(1):
predict_T5(usr_input)