// Author: Marek Szymański
// errDescription: Convert Wireshark capture to binary data file

#include "../include/crc32.hpp"
#include "../include/encoding.hpp"
#include "../include/transerrors.hpp"
#include "../include/frame.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

const int allToTestRatio = 6;

int main(int argc, char* argv [])
{

    std::string wiresharkFile;

    // === TRAINING DATASET ===
    std::string dataTrainFile;
    std::string xorTrainFile;
    std::string errDescTrainFile;

    // === TESTING DATASET ===
    std::string dataTestFile;
    std::string xorTestFile;
    std::string errDescTestFile;


    if(argc>1 && argv[1][0]=='t')
    {
        wiresharkFile = "../data/raw/capture_test.txt";        // .txt file with Wireshark capture

        // === TRAINING DATASET ===
        dataTrainFile = "../data/prep/train/capture_test.dat";            // .dat binary file with encoded training frames with errors
        xorTrainFile = "../data/prep/train/capture_test_xor.dat";         // .dat binary file with error vectors for training
        errDescTrainFile = "../data/prep/train/capture_test_errDesc.csv"; // .csv file with error classes for training

        // === TESTING DATASET ===
        dataTestFile = "../data/prep/test/capture_test.dat";            // .dat binary file with encoded testing frames with errors
        xorTestFile = "../data/prep/test/capture_test_xor.dat";         // .dat binary file with error vectors for testing
        errDescTestFile = "../data/prep/test/capture_test_errDesc.csv"; // .csv file with error classes for testing
    }
    else
    {
        wiresharkFile = "../data/raw/capture1.txt";        // .txt file with Wireshark capture

        // === TRAINING DATASET ===
        dataTrainFile = "../data/prep/train/capture.dat";            // .dat binary file with encoded training frames with errors
        xorTrainFile = "../data/prep/train/capture_xor.dat";         // .dat binary file with error vectors for training
        errDescTrainFile = "../data/prep/train/capture_errDesc.csv"; // .csv file with error classes for training

        // === TESTING DATASET ===
        dataTestFile = "../data/prep/test/capture.dat";            // .dat binary file with encoded testing frames with errors
        xorTestFile = "../data/prep/test/capture_xor.dat";         // .dat binary file with error vectors for testing
        errDescTestFile = "../data/prep/test/capture_errDesc.csv"; // .csv file with error classes for testing
    }



    std::ifstream ifile(wiresharkFile);

    std::ofstream ofileTrain(dataTrainFile, std::ios::binary);
    std::ofstream ofileXorTrain(xorTrainFile, std::ios::binary);
    std::ofstream ofileErrDescTrain(errDescTrainFile);

    std::ofstream ofileTest(dataTestFile, std::ios::binary);
    std::ofstream ofileXorTest(xorTestFile, std::ios::binary);
    std::ofstream ofileErrDescTest(errDescTestFile);

    if (!ifile.is_open())
    {
        std::cout << "Error: could not open: " << wiresharkFile << std::endl;
        return 1;
    }

    if (!ofileTrain.is_open())
    {
        std::cout << "Error: could not open: " << dataTrainFile << std::endl;
        return 1;
    }

    if (!ofileXorTrain.is_open())
    {
        std::cout << "Error: could not open: " << xorTrainFile << std::endl;
        return 1;
    }

    if (!ofileErrDescTrain.is_open())
    {
        std::cout << "Error: could not open: " << errDescTrainFile << std::endl;
        return 1;
    }

    if (!ofileTest.is_open())
    {
        std::cout << "Error: could not open: " << dataTestFile << std::endl;
        return 1;
    }

    if (!ofileXorTest.is_open())
    {
        std::cout << "Error: could not open: " << xorTestFile << std::endl;
        return 1;
    }

    if (!ofileErrDescTest.is_open())
    {
        std::cout << "Error: could not open: " << errDescTestFile << std::endl;
        return 1;
    }

    std::string wsline;
    bytesVec frame;
    std::uint16_t type;
    int complement4 = 0;
    bytesVec crcVec;
    uint32_t crcTable[256];
    crc32::generate_table(crcTable);
    int info[] = {0, 0, 0, 0, 0, 0, 0, 0};
    /*
    Info table meaning:
        0 - Total frames read from raw data file.
        1 - Total IPv4 frames read from raw data file.
        2 - Total encoding fails.
        3 - Total frames properly encoded and written to output files.
        4 - Total frames assigned to testing data.
        5 - Total frames assigned to training data.
        6 - Undefined
        7 - Undefined
    */
    std::vector<double> errors = {1.0, 0.1};
    std::mt19937 gen;
    
    while (getline(ifile, wsline))
    {
        if (wsline[0] != '|' || wsline[1] != '0')
        {
            continue;
        }

        wsline.erase(0, 6);

        // Parse a new frame
        frame = parseFrame(wsline);
        info[0]++;

        type = (frame[ETH2_TYPE_OFFSET] << BYTE_SIZE) + frame[ETH2_TYPE_OFFSET + 1];
        if (type != ETH2_TYPE_IP4)
        {
            continue;
        }
        info[1]++;
        // Randomize MAC addresses and IP addresses
        frame = randomizeMAC(frame, 2);
        frame = randomizeIPv4Addr(frame, 2);

        // Calculate and append CRC32 checksum
        crcVec = crc32::toBytesVec(crc32::crc(crcTable, frame));
        frame.insert(frame.end(), crcVec.begin(), crcVec.end());

        // Encode the frame using 8b/10b encoding
        // For the encoding, the frame size must be a multiple of 4
        frame = rightPadBytesVec(frame, frame.size() + (4 - (frame.size() % 4)), 0);
        frame = encodings::encodeBytesVec8b10b(frame);

        // If the frame could not be encoded, skip it
        if (frame.size() == 0)
        {
            std::cout << "Error: encoding failed" << std::endl;
            info[2]++;
            continue;
        }
        info[3]++;

        // Padd the encoded frame with 0s to the maximum size (1518 bytes)
        // Save the original size of the frame for later use (error introduction)
        int unpaddedSize = frame.size();
        frame = rightPadBytesVec(frame, ETH2_MAX_FRAME_SIZE, 0);

        // At this point we also create an error-free copy of the encoded frame - the xorFrame - to be XORed with the frame with errors
        bytesVec xorFrame = frame;

        // Introduce errors to the encoded frame
        // First we get random positions of bits to flip, according to the given probabilities (errors vector)
        // Then we classify the positions to the fields of the Ethernet II frame (getting map of {field: nr. of errors})
        // We use `unpaddedSize` to get the positions in the original frame, before the padding 0s were added
        // Finally we flip the bits in the frame according to the positions
        auto errorsPos = transerrors::getRandomPositions(errors, unpaddedSize, gen);
        auto errorsPosMap = transerrors::classifyPositions(errorsPos, unpaddedSize);
        frame = transerrors::flipBits(frame, errorsPos);

        //Pointers pointing to the file that will be used to write this frame
        std::ofstream* ofile = nullptr;
        std::ofstream* ofileXor = nullptr;
        std::ofstream* ofileErrDesc = nullptr;

        //Strings representing directories that will be used to write this frame
        std::string dataFile;
        std::string xorFile;
        std::string errDescFile;

        std::uniform_int_distribution<int> dist(0, allToTestRatio-1);
        int roll = dist(gen);

        //Assigning correct values to pointers and strings (assigning the frame to either test or training data)
        if(!roll)
        {
            ofile = &ofileTest;
            ofileXor = &ofileXorTest;
            ofileErrDesc = &ofileErrDescTest;

            dataFile = dataTestFile;
            xorFile = xorTestFile;
            errDescFile = errDescTestFile;

            info[4]++;
        }
        else
        {
            ofile = &ofileTrain;
            ofileXor = &ofileXorTrain;
            ofileErrDesc = &ofileErrDescTrain;

            dataFile = dataTrainFile;
            xorFile = xorTrainFile;
            errDescFile = errDescTrainFile;

            info[5]++;
        }

        // Write the new frame (with errors) to the output file
        ofile->write((char *)frame.data(), frame.size());
        if (ofile->fail())
        {
            std::cout << "Error: writing to file: " << dataFile << " failed on frame " << info[3] << std::endl;
            return 1;
        }

        // Perform byte-wise XOR operation on the error-free frame copy and the frame with errors
        // Thus we get the error vector, which will be used for error detection
        for (int i = 0; i < frame.size(); i++)
        {
            xorFrame[i] = xorFrame[i] ^ frame[i];
        }

        // Write the error vector to the output file
        ofileXor->write((char *)xorFrame.data(), xorFrame.size());
        if (ofileXor->fail())
        {
            std::cout << "Error: writing to file: " << xorFile << " failed on xor-frame " << info[3] << std::endl;
            return 1;
        }

        // Write the error class (errorsPosMap) of the frame to the errDescription file
        for (auto it = errorsPosMap.begin(); it != errorsPosMap.end(); it++)
        {
            if (std::next(it) == errorsPosMap.end())
            {
                (*ofileErrDesc) << it->second << "\n";
            }
            else
            {
                (*ofileErrDesc) << it->second << ",";
            }
        }
        if (ofileErrDesc->fail())
        {
            std::cout << "Error: writing to file: " << errDescFile << " failed on the desc of frame " << info[3] << std::endl;
            return 1;
        }
    }

    ifile.close();
    ofileTrain.close();
    ofileXorTrain.close();
    ofileErrDescTrain.close();
    ofileTest.close();
    ofileXorTest.close();
    ofileErrDescTest.close();

    std::cout << "Frames read: " << info[0] << std::endl;
    std::cout << "IPv4 frames: " << info[1] << std::endl;
    std::cout << "Encoding fails: " << info[2] << std::endl;
    std::cout << "Frames encoded, processed and written: " << info[3] << std::endl;
    std::cout << "Total frames in training data: " << info[5] << std::endl;
    std::cout << "Total frames in test data: " << info[4] << std::endl;

    return 0;
}