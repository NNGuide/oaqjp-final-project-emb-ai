import requests
import json

def emotion_detector(text_to_analyse):

    # Input parameters
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    text_json = { "raw_document": { "text": text_to_analyse } }
    
    # Send a POST request to the Watson NLP library
    response = requests.post(url, json = text_json, headers = header)

    # Convert the response text to json format
    formatted_response = json.loads(response.text)

    # Extract the required set of emotions' score
    anger_score = formatted_response['emotionPredictions'][0]['emotion']['anger']
    disgust_score = formatted_response['emotionPredictions'][0]['emotion']['disgust']
    fear_score = formatted_response['emotionPredictions'][0]['emotion']['fear']
    joy_score = formatted_response['emotionPredictions'][0]['emotion']['joy']
    sadness_score = formatted_response['emotionPredictions'][0]['emotion']['sadness']

    # Find the dominant_emotion
    def find_emotion_key(d, target_value):
        keys = [key for key, value in d.items() if value == target_value]
        return keys

    dominant_score = max(anger_score, disgust_score, fear_score, joy_score, sadness_score)
    dominant_emotion = find_emotion_key(formatted_response["emotionPredictions"][0]["emotion"], 
                                        dominant_score)

    # Get a string of the dominant emotion from the list result
    dominant_emotion = dominant_emotion[0]

    # return the json format output
    return {'anger': anger_score, 'disgust': disgust_score, 'fear': fear_score, 'joy': joy_score,
            'sadness': sadness_score, 'dominant_emotion': dominant_emotion}
