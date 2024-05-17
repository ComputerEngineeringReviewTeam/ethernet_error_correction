### Global TODO
- [ ] better error handling in `encoding.cpp`
- [ ] redo pytorch datasets to load data from new storage format
- [ ] convert all data

### Global MAYBE
- [ ] randomize IPv4 payloads ?

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