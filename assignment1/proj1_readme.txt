'''
:Name: Ryan C. Adams
:Course: CS487_Applied_Machine_Learning
:Implementation: read me file
readme for how to run files
'''
Important:
The files are set to run based on the following format:
Also please note that my program checks to see that the dataset ends with file extension .csv

python3 main.py <alogrithm> <dataset> <eta_value> <number_iterations>

Here is a list that can be individually copy and pasted to run each

python3 main.py perceptron iris.csv .01 10
python3 main.py adaline iris.csv .001 100
python3 main.py sgd iris.csv .01 10
python3 main.py percptron <dataset2> .01 10