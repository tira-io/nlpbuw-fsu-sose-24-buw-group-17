FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py

RUN apt-get update && apt-get install -y curl

RUN curl -L -o /code/vectorizer.joblib "https://drive.google.com/uc?export=download&id=1D2wHOdRQG1bV9hrNuxmuXxcykSnHLB3D"
RUN curl -L -o /code/tfidf_matrix.joblib "https://drive.google.com/uc?export=download&id=153PbgJ0QHIBXTHh9ee3B3AMFQLSky65z"

RUN pip install numpy
RUN pip install scikit-learn
RUN pip install joblib
RUN pip install tira

ENTRYPOINT [ "python3", "/code/run.py" ]
