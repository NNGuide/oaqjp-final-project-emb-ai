import requests

def emotion_detector(text_to_analyse):

    # Input parameters
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    text_json = { "raw_document": { "text": text_to_analyse } }
    
    # Send a POST request to the Watson NLP library
    response = requests.post(url, json = text_json, headers = header)

    # return the response text from the library
    return response.text
