# Asthma exacerbation prediction
This is the codebase for the paper https://www.medrxiv.org/content/medrxiv/early/2019/11/29/19012161.full.pdf <br>
Asthma Exacerbation Prediction and Interpretation based on Time-sensitive Attentive Neural Network: A Retrospective Cohort Study
, which is used to predict the risk of asthma exacerbation among asthma patients according to a history of EHR data. The model can be also scaled to other sequential prediction tasks. Please refer to the paper for more details

## Environment
Python 3.0+ <br>
Tensorflow 1.14+, GPU <br>
Numpy, Scikit-learn, etc.

## Demo script
cd src/ <br>
sh run.sh

## Sample data format
code: a list (list of visits) of list (list of code in each visit) <br>
time: a list (list of visits) of list (time to the first visit, time to the previous visit, time to the last visit) <br>
You can choose any time you want or choose all <br>
There is also a sample data in the data/ directory (compressed using pickle), you can load it using pkl.load
