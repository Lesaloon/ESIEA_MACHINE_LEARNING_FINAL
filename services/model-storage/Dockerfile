FROM alpine:3.21

WORKDIR /models

COPY storage-init/ /storage-init/

CMD ["sh", "-c", "mkdir -p /models/registry /models/artifacts /models/metadata && sleep infinity"]
