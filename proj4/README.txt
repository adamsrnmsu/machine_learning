:Name: Ryan C. Adams
:Course: CS487 Applied Machine Learning I
:Project 4: Regression 

The usage of this file is in the form of :
main.py [-h] [-f LOAD_FROM_FILE] [-s START] [-t TARGET]

  -f LOAD_FROM_FILE, --load_from_file LOAD_FROM_FILE
                        specify path to data file
  -s START, --start START
                        column number of where training data (X) starts
  -t TARGET, --target TARGET
                        column number of answer column (Y)

CLI INSTRUCTIONS (RUN THESE):

*** NOTE *********************************************************************
Neural MLP Regressor Results takes a second a bit to run and throws a warning
be patient.
****************************************************************************

python3 main.py -f all_breakdown.csv -s 1 -t 7
python3 main.py -f housing.data.txt -s 1 -t 2