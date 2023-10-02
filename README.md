
## Create and activate conda environment
`conda env create -f environment.yml`
`conda activate master`


## Run project
- Experiments are defined by a directory containing a `config.ini`, `dataset_config.ini`, and
optionally a `runprops.txt`
- Experiment can then be executed by running the command:
    `python src/run-pipeline.py -dir {path-to-experiment-directory} --generate --train --eval -p {profiles} --device {device (cpu or cuda)}`
