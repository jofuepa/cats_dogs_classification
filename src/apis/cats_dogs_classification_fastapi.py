# FastAPI service to classify an image as a cat or dog
# activate Pyenv: $ source venv/bin/activate
# run locally: $ uvicorn cats_dogs_classification_fastapi:app --reload
# test terminal: $ 
# test browser: http://localhost:8000/docs 
# kill TCP connections on port 8080: sudo lsof -t -i tcp:8000 | xargs kill -9
# Request body example: