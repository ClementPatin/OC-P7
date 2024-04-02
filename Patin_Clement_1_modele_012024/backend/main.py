# imports
import uvicorn
from fastapi import FastAPI, HTTPException
import myFunctionsForApp as mf

# initiate the app
app = FastAPI()




# create index
@app.get('/')
def index() :
    return {"message" : "welcome to the Air Paradis API - TEST CI/CD"}

# load text_vectorization layer and model interpreter form TfLite folder
text_vect_loaded, interpreter_loaded = mf.load_prod_advanced_model(load_path="TfLite")


# create predict
@app.post('/predict')
def predict_sentiment(data : dict) :
    # handle errors
    print(type(data))
    if "text" not in data.keys() :
        raise HTTPException(status_code=400, detail="key should be 'text'")
    if type(data["text"]) != str :
        raise HTTPException(status_code=400, detail="input value should be in string format")
    if data["text"] == "" :
        raise HTTPException(status_code=400, detail="empty input")
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






