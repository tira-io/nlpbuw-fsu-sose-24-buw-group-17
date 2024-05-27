FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD logistic_regression_model.pkl /code/logistic_regression_model.pkl
ADD tfidf_vectorizer.pkl /code/tfidf_vectorizer.pkl

ENTRYPOINT [ "python3", "/code/run.py" ]
