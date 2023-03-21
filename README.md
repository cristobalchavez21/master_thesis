# DPJ

For branching the git repo follow the name convetion:
{name}/identifier

for example:
    mhaacke/xgboost_cut_2



## Getting started

### Runing in batch

For instalation from root of the proyect:

```bash
make venv
source .venv/bin/activate
make install
make run
```

note: you need to put the path to your samples in the config.yaml

if more indepth you can use the notebook to test

### Runing in jupyter

at the first time run 

```bash
make venv
source .venv/bin/activate
make install
pip install ipykernel

```

After doing this, in vscode should appear as a valid kernel and must be chosen


