set WORKER_MODE=tournament
set WRITER_MODE=tournament
docker-compose --profile tournament up --build --scale worker=1 --scale trainer=0
pause