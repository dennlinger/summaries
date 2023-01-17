# Reproducibility Challenge for BTW'23

Author: Dennis Aumiller

## System Requirements

* OS: Tested on Ubuntu 20.04
* Python3.7 or later
* 7 GB disk memory for datasets
* CUDA-compatible GPU preferable for reproduction of inference runs


## Installation Notes
**Note: This has been tested on Ubuntu 20.04 LTS. A successful reproduction on other systems is not guaranteed!**  
Furthermore, this assumes you have Python3.7+ installed on your system.

To reproduce the content necessary for the paper's experiments,
simply install the library according to the `README.md`'s instructions:  
First, install the necessary dependencies with
```bash
python3 setup.py install
```
and then set up the library with `pip`:
```bash
python3 -m pip install .
```

Also ensure that you have the `unzip` shell command installed:
```bash
sudo apt-get install unzip
```


## Execution of Master Script
The master script produces the majority of artifacts required to reproduce this work.
To execute it, navigate to the respective experimental folder and run these two commands:

```bash
cd experiments/german_summarization_datasets
./master_script.sh
```

This will automatically download all the necessary data for the experiments (careful, this is close to 7 GB)
and execute the individual experiments.

Also see the comments in `master_script.sh` for more information on the respective commands and tools.
