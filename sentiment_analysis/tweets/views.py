# # from django.shortcuts import render

# # tweets/views.py
# import pickle
# from django.shortcuts import render
# from django.http import HttpResponse
# import os

# # Load the model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# with open(MODEL_PATH, 'rb') as f:
#     model = pickle.load(f)

# def sentiment_view(request):
#     if request.method == 'POST':
#         text_value = request.POST.get('text_value', '')
#         if text_value:
#             # Predict sentiment
#             prediction = model.predict([text_value])[0]
#             output_sentiment = 'Harmful' if prediction == 1 else 'Not Harmful'
#         else:
#             output_sentiment = 'No tweet provided'

#         return render(request, 'base.html', {'output_sentiment': output_sentiment, 'text_value': text_value})
#     else:
#         return render(request, 'base.html')

# tweets/views.py
# tweets/views.py
# tweets/views.py
import pickle
from django.shortcuts import render
import os

# Load the model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def sentiment_view(request):
    if request.method == 'POST':
        text_value = request.POST.get('text_value', '')
        if text_value:
            # Transform the input text using the loaded vectorizer
            text_transformed = tfidf_vectorizer.transform([text_value])
            # Predict sentiment
            prediction = model.predict(text_transformed)[0]

            # Map predictions to corresponding classes
            class_mapping = {
                1: 'not_cyberbullying',
                2: 'gender',
                3: 'religion',
                4: 'other_cyberbullying',
                5: 'age',
                6: 'ethnicity'
            }
            output_sentiment = class_mapping.get(prediction, 'Unknown')
        else:
            output_sentiment = 'No tweet provided'

        return render(request, 'base.html', {'output_sentiment': output_sentiment, 'text_value': text_value})
    else:
        return render(request, 'base.html')
