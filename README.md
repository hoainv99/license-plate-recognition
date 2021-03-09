# INSTALL

``` pip install -r requirement.txt```
# RUN
```uvicorn main:app --reload```
# API
```curl -X POST "http://localhost:8000/plate_recognition/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@{file_name};type=image/jpeg"```
# OUTPUT SAMPLE
```{
  "status": true,
  "file_name": "23.jpg",
  "format": ".jpg",
  "data": {
      "{plate_number_1}": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wB...",
      "{plate_number_2}": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wB...",
      ...
      "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBA..."
  }
}```
