# REQUIREMENT
To install dependencies and weight. Flow this:

``` pip install -r requirement.txt```

``` gdown --id 1HL68JCUqGKopjk3w_gosA6nC4-Mk_72j```

``` unzip -q weights_license_plate_recognition.zip ```

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
