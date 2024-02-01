# How to run

```
docker compose up --build --detach
```

Open this in your browser
```
http://localhost:8000/detect?url=<image-url>
```

Or try the api from the docs
```
http://localhost:8000/docs
```

Example:

URL
```
http://localhost:8000/detect?url=https://farm3.staticflickr.com/2666/4155270341_eecd47de5a_z.jpg
```

Response:
```
{
    "objects": [
        {
            "object_name": "zebra",
            "confidence": 0.999,
            "location": [
                269.34,
                186.35,
                341.77,
                317.4
            ]
        },
        {
            "object_name": "zebra",
            "confidence": 0.999,
            "location": [
                103.23,
                153.98,
                275.79,
                324.67
            ]
        }
    ],
    "latency": 0.034041643142700195,
    "device": "cuda:0"
}
```


You need to configure CUDA for GPU support, and also configure NVIDIA for container toolkit.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)