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

You need to configure CUDA for GPU support, and also configure NVIDIA for container toolkit.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)