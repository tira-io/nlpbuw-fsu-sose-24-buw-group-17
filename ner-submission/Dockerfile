# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD run.py /code/run.py
ADD crf_model.joblib /code/crf_model.joblib

ENTRYPOINT [ "python3", "/code/run.py" ]
