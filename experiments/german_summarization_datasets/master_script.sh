#!/bin/bash

# This script requires the use of `unzip`. If your system does not have access to it, install it by running
# sudo apt-get install unzip

##################
# Acquire datasets
##################

# EUR-Lex German split (this slightly differs in the processing from the Huggingface version)
curl -O -J -L https://heibox.uni-heidelberg.de/f/e94dabecb4864ff296bf/?dl=1 > eurlexsum/german_eurlexsum.json

# MassiveSumm
curl -O -J -L

# SwissText dataset for the 2019 challenge:
curl -O -J -L https://drive.switch.ch/index.php/s/YoyW9S8yml7wVhN/download
unzip swisstext.zip

# LegalSum
curl -O -J -L https://www.dropbox.com/s/23mrviv5396rdl0/LegalSum.zip?dl=1
unzip LegalSum.zip -d legalsum/

# Additional LegalSum split files
wget https://raw.githubusercontent.com/sebimo/LegalSum/master/model/train_files.txt
mv train_files.txt legalsum/
wget https://raw.githubusercontent.com/sebimo/LegalSum/master/model/val_files.txt
mv val_files.txt legalsum/
wget https://raw.githubusercontent.com/sebimo/LegalSum/master/model/test_files.txt
mv test_files.txt legalsum/

# Klexikon, Wikilingua and MLSUM can be acquired directly through the `datasets` library.


# Will run the stats on the various datasets to get to the results in Table 2
python3 eurlexsum/clean_eurlexsum.py
python3 klexikon/clean_kelxikon.py
python3 legalsum/clean_legalsum.py
python3 massivesumm/clean_massivesumm.py
python3 mlsum_experiments/clean_mlsum.py
python3 swisstext/clean_swisstext.py
python3 wikilingua/clean_wiki_lingua.py

# Plot the Violin Plots of Figure 2. Careful: Due to the conflation of density estimates and counts, it looks
# like the filtering of MassiveSumm has no (inverse?) effect. This is not the case, and only due to different scaling.
python3 plot_violin_reference.py
python3 plot_violin_summary.py

# Execute baseline experiments in Table 3 for the baseline runs
python3 baselines_generation/lead3.py
python3 baselines_generation/leadk.py
python3 baselines_generation/lexrank_st.py

# Execute the individual model runs of Table 4 for MLSUM specifically
# NOTE: This requires a GPU to be run efficiently
python3 baselines_generation/inference_mlsum.py
