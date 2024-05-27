FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD model.joblib /code/model.joblib
ADD best_threshold.txt /code/best_threshold.txt
ADD model.py /code/model.py

ENTRYPOINT [ "python3", "/code/run.py" ]
