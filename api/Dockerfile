#*'''#runtime or python application 
FROM python:3.11.0  

#copy the requrements to a tmp file
COPY requirements.txt  tmp/requirements.txt 

# Install the dependencies specified in the requirements.txt file
RUN python -m pip install --timeout 3000 -r /tmp/requirements.txt

COPY  . /app 

WORKDIR /app 

EXPOSE 80

CMD ["uvicorn","api:app", "--host","0.0.0.0","--port","80"]