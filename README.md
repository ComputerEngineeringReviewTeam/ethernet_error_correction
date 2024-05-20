## NAVIGATION
- REAMDE.md - you are here
- requirements.txt - required Python modules, for `pip install -r requirements.txt`
- .gitignore
- .gitattributes
- gpp.sh - g++ compiler script for C++ files
- clang.sh - same as above, but for clang OUTDATED
- ./venv - Python venv
- bin - compiled binaries from C++ (output from gpp.sh goes here)
- data
  - raw - raw data ie. Wireshark capture files
  - prep - preprocessed data goes here, format below
  - train - training data
    - capture.txt - the big capture file
      - capture_test.txt - smaller subset, for testing purposes
      - capture_xor.txt - xors of the correct and damaged frames
      - capture_test_xor.txt - smaller subset, xors of the correct and damaged frames
      - capture_ErrDesc.csv - frame description (location of errors)
      - capture_test_ErrDesc.csv - smaller subset, frame description (location of errors)
    - test - test data
      - capture.txt - the big capture file
      - capture_test.txt - smaller subset, for testing purposes
      - capture_xor.txt - xors of the correct and damaged frames
      - capture_test_xor.txt - smaller subset, xors of the correct and damaged frames
      - capture_ErrDesc.csv - frame description (location of errors)
      - capture_test_ErrDesc.csv - smaller subset, frame description (location of errors)
- src - actual source code
  - prep - data preprocessing code - written in C++
    - convert - main program(s) to preprocess the frame data
    - include - all header files and encoding tables
    - src - function definition files
    - test - small pseudo-tests or playgrounds for testing various bits of code
  - ml - all Python machine learning code
    - datasets - Dataset classes, inheriting torch.util.data.Dataset, for loading data into tensor format
      - EtherBits.py - tensor of bits (as torch.tensor.bool)
      - EtherBytes.py - tensor of bytes (as torch.tensor.uint8)
      - funcs.py - some functions for converting from binary into torch.tensor
    - util - various utilities
      - trainer.py - simple class that takes and trains torch.Module, to avoid repetitive code, probably not the best for real usage
    - test_datasets.py - simple testing for data loading, MOVE OUT / REMOVE


### Global TODO
- [ ] better error handling in `encoding.cpp`
- [ ] convert all data

### Global MAYBE
- [ ] randomize IPv4 payloads ?
- [ ] move bitify() function from Python into C for speed

## NEW STORAGE FORMAT
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
