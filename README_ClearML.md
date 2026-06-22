# ClearML

## Server installation via Docker Desktop in Windows 10
https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_win

## Run docker container
docker-compose -f c:\opt\clearml\docker-compose-win10.yml up

## Initialize ClearML
clearml-init

## Paste credentials from ClearML server
Example:
```
api {
  web_server: http://192.168.1.173:8080
  api_server: http://192.168.1.173:8008
  files_server: http://192.168.1.173:8081
  credentials {
    "access_key" = "****"
    "secret_key" = "***"
  }
}
```

## Initialize ClearML agent in remote machine
clearml-agent init
