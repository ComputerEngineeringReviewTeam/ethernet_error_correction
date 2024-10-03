## NAVIGATION
- requirements.txt - required Python modules
- convert_compile.sh - g++ compiler script for data preparation, outputs to /bin/convertdata
- docs/ - project docs
  - navigation.md - you are here
  - data_format.md - detailed explanation of our data storage format
  - paper.odt / paper.pdf - final paper in .odt / .pdf format
- data/ - due to big file sizes (up to 1GB) data is NOT stored on GitHub
  - raw/ - raw data i.e. Wireshark capture files
  - prep/ - prepared data would be here, for detailed format [here](data_format.md) 
    - train/ - training data
    - test/  - test data
- src/ - actual source code
  - prep/ - data preprocessing code in C++
    - convert/ - main program(s) to prepare the frame data
    - include/ - all header files, constants and reference tables
      - constants.hpp - different constants and aliases used all throughout the project
      - crc32.hpp - generating CRC32 checksum of a vector of bytes, credit to https://github.com/timepp 
      for reference table generation function
      - encoding.hpp - software-level re-implementation of 8b/10b encoding for vectors of bytes
      - encodingtables.hpp - reference tables used in 8b/10b encoding and decoding
      - filesaver.hpp - class for managing file I/O during data preprocessing
      - frame.hpp - various operations on Ethernet II frames - reading, parsing from Wireshark, 
      saving to file etc.
      - transerrors.hpp - introducing single-bit errors into Ethernet II frames
    - src/ - function definitions 
  - ml/ - all Python machine learning code
    - datasets/ - Dataset classes, inheriting torch.util.data.Dataset
      - EtherBits.py - frames are expressed as tensors of bits (each as torch.tensor.bool)
      - EtherBytes.py - frames are expressed as tensors of bytes (each as torch.tensor.uint8)
    - util/ - various utilities
    - modules/ - examples of networks used in the projects, for exhaustive list refer to 
    [paper](paper.pdf)
    - trainer.py - simple class that takes and trains torch.Module
    - training.py - code for training and assesment of the model using the Trainer class
