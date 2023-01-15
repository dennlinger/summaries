# Reproducibility Challenge for BTW'23

Author: Dennis Aumiller

**Note: This has been tested on Ubuntu 20.04 LTS. A successful reproduction on other systems is not guaranteed!**  
Furthermore, this assumes you have Python3.7+ installed on your system.

To reproduce the content necessary for the paper's experiments,
simply install the library according to the `README.md`'s instructions:
```bash
python3 -m pip install .
```

Also ensure that you have the `unzip` shell command installed:
```bash
sudo apt-get install unzip
```

Navigate to the respective experimental folder and execute the master script:

```bash
cd experiments/german_summarization_datasets
./master_script.sh
```

This will automatically all the necessary data for the experiments (careful, this is close to 10 GB)
and execute the individual experiments.

Also see the comments in `master_script.sh` for more information on the respective commands and tools.
