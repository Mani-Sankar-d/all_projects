from fastapi import FastAPI
from ytchat import run
app=FastAPI()

@app.get('/')
def home():
    return {'message':'home'}
@app.get('/{id}')
def give(query:str,id:str):
    return run(query,id)