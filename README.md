## WHAT IS THIS?
A detailed description of our hypothesis, methotodology and findings can be found in our final report.
The final report is availible in this repository in two file formats: paper.pdf and paper.odt


A very brief summary of what we're trying to achieve here:
___________________________________________________________________

Based on the assumption, that live Internet communication is, in it’s nature, self-
similar, we asserted that it is possible to build a machine learning model, that would
identify and correct errors in Internet transmissions.

We decided to test this theory by attempting to train a neural network to correct errors in damaged Ethernet 2 frames.


A very brief summary of our findings:
___________________________________________________________________

We found, that our assertion (that machine learning models could be used to correct
transmission errors based on the data being transmitted) to be true in a strictly factual
sense. However, low effectiveness of this method of error correction makes it
unfeasible as a standalone solution.


If that sounds interesting - we'd like to invite you to read the [full paper](docs/paper.pdf) 


## NAVIGATION
The project repository is organized as follows:
- convert_compile.sh - basic script for compiling and running the data preprocessing pipeline, uses g++
- docs - our final report, detailed project navigation and data storage format explanation
- data - raw (Wireshark capture file) and preprocessed data - not included in this repository due to size constraints 
- src - actual source code
  - prep - data preprocessing in C++
  - ml - everything related to machine learning - datasets, modules etc

More detailed map [here](docs/navigation.md)

## STORAGE FORMAT
Single set of preprocessed frames is stored in the following format:
- a binary file containing all the preprocessed frames
- a binary file containing the XORs of all the preprocessed frames before and after the errors were introduced 
- a CSV file containing the number and position of errors introduced into each frame

More detailed explanation [here](docs/data_format.md)
