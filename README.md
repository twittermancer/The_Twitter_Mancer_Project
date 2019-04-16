# The TwitterMancer Project
Code and dataset for the paper:</br>
**TwitterMancer:Predicting Interactions on Twitter Accurately**
Konstantinos Sotiropoulos<sup>1</sup>, John W. Byers<sup>1</sup>, Polyvios Pratikakis<sup>2</sup>, and Charalampos E. Tsourakakis<sup>1</sup></br>
<sup>1</sup>Boston University, USA <sup>2</sup>University of Crete, Greece </br>

## Dataset
Whole dataset is in the "data/" folder, in a compressed format and split in different files
in order for files to be <50MB per github rules. Dataset is anonymized.

## In order to reproduce results...
  Run sequentially:
  ```
  python create_followSet.py
  python construct_features.py {start_date} {end_date}
  python prediction.py {start_date} {end_date} > results/prediction_task.txt
  python degree_precision.py {start_data} {end_date}
  ```
  where *start_date* and *end_date* are arguments that define the time window we want to use in our dataset </br>
  e.g. for using the whole dataset feb 1- feb 28 we have to run:
  ```
  python create_followSet.py
  python construct_features.py 1 28 
  python prediction.py 1 28 > results/prediction_task.txt
  python degree_precision.py 1 28
  ```

## Read results from jupyter notebook
We have created a jupyter notebook (called **read_results.ipynb** ), which reads the output of the prediction.py
script (which is saved in a txt file under "results/prediction.txt") and a pickle file, where we have saved the 
prediction accuracy per embeddedness results and reproduces the main figures and plots from our paper. </br>
 
 **Important!**
  1. Triangles were listed using [MACE](http://research.nii.ac.jp/~uno/code/mace.html) package
      Our scripts require a MACE executable inside a _"mace/"_ folder.
  2. code is written in Python2.7 and the scikit-learn version used in the experiments is 0.20.2.
  3. Dataset is anonymized, so given user IDs do not represent real twitter users.
