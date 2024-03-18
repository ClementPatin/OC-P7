# imports
import uvicorn
from fastapi import FastAPI
import myFunctionsForApp as mf


app = FastAPI()




@app.get('/')
def index() :
    return {"message" : "welcome to the Air Paradis API _  test"}


text_vect_loaded, interpreter_loaded = mf.load_prod_advanced_model(load_path="TfLite")


@app.post('/predict')
def predict_sentiment(data : dict) :
    score = mf.predict_with_TFLite_loaded_model(
        text_vectorizer=text_vect_loaded,
        interpreter=interpreter_loaded,
        X=data["text"],
        proba=True
    )[0]
    if score >= 0.5 :
        sentiment = "negative"
    else :
        sentiment = "positive"
    
    return {"score" : str(score), "sentiment" : sentiment}



# if __name__ == '__main__' :
#     uvicorn.run(app, host="127.0.0.1", port = 8000)

# uvicorn main:app --reload






