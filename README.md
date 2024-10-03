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


If that sounds interesting - we'd like to invite you to read the full paper. 


## NAVIGATION
- REAMDE.md - you are here
- paper.odt - final report in .odt format
- paper.pdf - final report in .pdf format
- requirements.txt - required Python modules, for `pip install -r requirements.txt`
- .gitignore
- gpp.sh - g++ compiler script for C++ files
- gpp2.sh - another g++ script for compiling the newer version of data preparation program
- clang.sh - same as above, but for clang OUTDATED
- bin - compiled binaries from C++ (output from gpp.sh goes here)
- data
  - raw - raw data ie. Wireshark capture files
  - prep - preprocessed data goes here, format below
    - train - training data
    - test - test data
- src - actual source code
  - prep - data preprocessing code - written in C++
  - ml - all Python machine learning code
    - datasets - Dataset classes, inheriting torch.util.data.Dataset, for loading data into tensor format
    - util - various utilities
    - models - saved neural network models
    - trainer.py - simple class that takes and trains torch.Module
    - test_datasets.py - simple testing for data loading, MOVE OUT / REMOVE
    - network.py - neural network class
    - training.py - code for training and assesment of the model using the Trainer class


## STORAGE FORMAT
Single set of preprocessed frames is stored in the following format:
- `name.dat` - binary file containing all the preprocessed frames (randomized MAC / IP addresses, CRC32 appended, padded, encoded and with randomized errors). Each frame is stored as a sequence of 1518 bytes.
- `name_xor.dat` - binary file containing the XORs of all the preprocessed frames before and after the errors were introduced - ie. 1s only on the positions of changed bits. Each XOR is stored as a sequence of 1518 bytes. Each block of 1518 bytes corresponds to a single frame (also 1518 block) in `name.dat` file.
- `name_errDesc.csv` - CSV file containing the number of errors introduced into each frame. Each row contains the number of errors introduced into different parts the corresponding frame (block of 1518 bytes) in the `name.dat` file. The columns are as follows:
    - `0` - number of errors introduced into the Destination MAC address
    - `1` - number of errors introduced into the Source MAC address
    - `2` - number of errors introduced into the EtherType
    - `3` - number of errors introduced into the IPv4 header
    - `4` - number of errors introduced into the IPv4 payload
    - `5` - number of errors introduced into the CRC32
