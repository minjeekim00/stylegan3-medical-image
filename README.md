# stylegan3-medical-image

## Docker setting

1. Build docker image via Dockerfile.
```
    docker build -t stylegan3 .
```
2. Run docker container with the script below.

```
docker run -it \
        --restart always \
        --gpus all \
        --name container_stylegan3 \
        --workdir /workspace \
        -v /mnt:/mnt \
        -v $PWD:/workspace \
        --shm-size 16G \
        -p 3336-3338:3336-3338 \
        stylegan3 /bin/bash
```
or

```
source run_docker.sh
```


## Usage

1. Create PNG dataset using the notebook
```
    python3 dcm_to_png.py --src-dir /mnt/dataset/Synthesis_Study/2022/AbdomenCT \
                          --dst-dir /mnt/dataset/Synthesis_Study/2022/AbdomenCT_png \
                          --modality abdomenct \
                          --resolution 512
```
2. Start train with provided hyperparameters

```
cd stylegan3
source train.sh
```


2-1. Presumably, your hyperparamters will be:
```
python3 train.py --outdir ./training-runs \
                --cfg stylegan2 \
                --data /mnt/dataset/Synthesis_Study/2022/AbdomenCT_png \
                --gpus 4 \
                --batch-gpu 80 \
                --batch 320 \
                --gamma 20 \
                --mirror False \ ## should be False
                --aug ada \
                --kimg 20000 \
                --snap 15 \
                --metrics none
```
