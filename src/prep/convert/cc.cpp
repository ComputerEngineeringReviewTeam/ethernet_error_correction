// Author: Marek Szymański
// errDescription: Convert Wireshark capture to binary data file

#include "../include/crc32.hpp"
#include "../include/encoding.hpp"
#include "../include/transerrors.hpp"
#include "../include/frame.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;
using symbolVec = std::vector<symbol10>;

const int allToTestRatio = 6;


int main()
{
    // === CONTROLS ===
    std::string filename = "f100";                                                        // Filename of the output files
    int allToTestRatio = 6;                                                             // Ratio of all frames to be assigned to test data
    int maxFrameSize = 100;                                                            // Maximum size of the frame we want to process
    bool toXOR = true;                                                                  // Write error vectors to the output file
    bool toErrDesc = true;                                                              // Write error descriptions to the output file
    bool toOg = true;                                                                   // Write original frames to the output file
    std::uint16_t etherType = ETH2_TYPE_IP4;                                            // Ethernet II type of the frames we want to process
    std::string wiresharkFile = "../data/raw/capture1.txt";                         // .txt file with Wireshark capture


    // === FILENAMES ===
    std::string trainDirPath = "../data/prep/train/";
    std::string testDirPath = "../data/prep/test/";
    std::string rawDataDirPath = "../data/raw";

    std::string binaryExt = ".dat";
    std::string csvExt = ".csv";

    std::string xorLabel = "_xor";
    std::string errDescLabel = "_errDesc";
    std::string ogLabel = "_og";

    // === TRAINING DATASET ===
    std::string dataTrainFile = trainDirPath + filename + binaryExt;                    // .dat binary file with encoded training frames with errors
    std::string xorTrainFile = trainDirPath + filename + xorLabel + binaryExt;          // .dat binary file with error vectors for training
    std::string errDescTrainFile = trainDirPath + filename + errDescLabel + csvExt;     // .csv file with error classes for training
    std::string ogTrainFile = trainDirPath + filename + ogLabel + binaryExt;            // .dat binary file with original frames

    // === TESTING DATASET ===
    std::string dataTestFile = testDirPath + filename + binaryExt;                      // .dat binary file with encoded testing frames with errors
    std::string xorTestFile = testDirPath + filename + xorLabel + binaryExt;            // .dat binary file with error vectors for testing
    std::string errDescTestFile = testDirPath + filename + errDescLabel + binaryExt;    // .csv file with error classes for testing
    std::string ogTestFile = testDirPath + filename + ogLabel + binaryExt;              // .dat binary file with original frames

    std::ifstream ifile(wiresharkFile);

    std::ofstream ofileTrain(dataTrainFile, std::ios::binary);
    std::ofstream ofileXorTrain(xorTrainFile, std::ios::binary);
    std::ofstream ofileErrDescTrain(errDescTrainFile);
    std::ofstream ofileOgTrain(ogTrainFile, std::ios::binary);

    std::ofstream ofileTest(dataTestFile, std::ios::binary);
    std::ofstream ofileXorTest(xorTestFile, std::ios::binary);
    std::ofstream ofileErrDescTest(errDescTestFile);
    std::ofstream ofileOgTest(ogTestFile, std::ios::binary);

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

    if (!ofileOgTrain.is_open())
    {
        std::cout << "Error: could not open: " << ogTrainFile << std::endl;
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

    if (!ofileOgTest.is_open())
    {
        std::cout << "Error: could not open: " << ogTestFile << std::endl;
        return 1;
    }

    std::string wsline;
    bytesVec frame;
    symbolVec encodedFrame;
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
        4 - Total frames assigned to training data.
        5 - Total frames assigned to testing data.
        6 - IPv4 frames longer than given bytes
        7 - Undefined
    */
    std::vector<double> errors = {1.0, 0.1};
    std::mt19937 gen;
    
    while (getline(ifile, wsline))
    {
        // Filter off non-data lines
        if (wsline[0] != '|' || wsline[1] != '0')
        {
            continue;
        }

        // Remove the first 6 characters - in Wireshark format they are always "|0   " before the actuall frame
        wsline.erase(0, 6);

        // Parse a new frame
        frame = parseFrame(wsline);
        info[0]++;

        // Skip frames of different type than the one we want
        type = (frame[ETH2_TYPE_OFFSET] << BYTE_SIZE) + frame[ETH2_TYPE_OFFSET + 1];
        if (type != etherType)
        {
            continue;
        }

        // Skip frames larger than the given size
        if (frame.size() > maxFrameSize - ETH2_CRC_SIZE)
        {
            info[6]++;
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
        encodedFrame = encodings::encode8b10b(frame);

        // If the frame could not be encoded, skip it
        if (encodedFrame.size() == 0)
        {
            std::cout << "Encoding failed on frame" << info[2] << std::endl;
            info[2]++;
            continue;
        }
        info[3]++;

        // At this point we also create 2 error-free copies of the encoded frame 
        //  - the ogFrame - to be written to the output file
        //  - the xorFrame - to be XORed with the frame with errors
        bytesVec xorFrame = frame;
        bytesVec ogFrame = frame;

        // Introduce errors to the encoded frame
        // First we get random positions of bits to flip, according to the given probabilities (errors vector)
        // Then we classify the positions to the fields of the Ethernet II frame (getting map of {field: nr. of errors})
        // We use the size of the frame before encoding to get the positions in the original frame
        // Finally we flip the bits in the frame according to the positions and decode the frame back to bytes
        auto errorsPos = transerrors::getRandomPositions(errors, frame.size(), gen);
        auto errorsPosMap = transerrors::classifyPositions(errorsPos, frame.size());
        encodedFrame = transerrors::flipBits(encodedFrame, errorsPos);
        frame = encodings::decode8b10b(encodedFrame);

        // Padding the both the frame (the one with errors) and the xorFrame (currently - error free copy) 
        // to the max standard Ethernet II frame size - 1518 bytes
        // This is done by adding 0s to the end of the frame
        // First we detach the CRC32 checksum from the end of the frame (the last 4 bytes)
        bytesVec crcWithErrors(frame.begin() + frame.size() - ETH2_CRC_SIZE, frame.end());
        bytesVec crcGood(xorFrame.begin() + xorFrame.size() - ETH2_CRC_SIZE, xorFrame.end());
        bytesVec crcGood2(ogFrame.begin() + ogFrame.size() - ETH2_CRC_SIZE, ogFrame.end());
        // Then we add 0s so that the frame is 1514 (ETH2_MAX_FRAME_SIZE - ETH2_CRC_SIZE) bytes long
        frame = rightPadBytesVec(frame, maxFrameSize - ETH2_CRC_SIZE);
        xorFrame = rightPadBytesVec(xorFrame, maxFrameSize - ETH2_CRC_SIZE);
        ogFrame = rightPadBytesVec(ogFrame, maxFrameSize - ETH2_CRC_SIZE);
        // Finally we append the CRC32 checksum to the end of the frame
        frame.insert(frame.end(), crcWithErrors.begin(), crcWithErrors.end());
        xorFrame.insert(xorFrame.end(), crcGood.begin(), crcGood.end());
        ogFrame.insert(ogFrame.end(), crcGood2.begin(), crcGood2.end());

        //Pointers pointing to the file that will be used to write this frame
        std::ofstream* ofile = nullptr;
        std::ofstream* ofileXor = nullptr;
        std::ofstream* ofileErrDesc = nullptr;
        std::ofstream* ofileOg = nullptr;

        //Strings representing directories that will be used to write this frame
        std::string dataFile;
        std::string xorFile;
        std::string errDescFile;
        std::string ogFile;

        std::uniform_int_distribution<int> dist(0, allToTestRatio-1);
        int roll = dist(gen);

        //Assigning correct values to pointers and strings (assigning the frame to either test or training data)
        if(!roll)
        {
            ofile = &ofileTest;
            ofileXor = &ofileXorTest;
            ofileErrDesc = &ofileErrDescTest;
            ofileOg = &ofileOgTest;

            dataFile = dataTestFile;
            xorFile = xorTestFile;
            errDescFile = errDescTestFile;
            ogFile = ogTestFile;

            info[4]++;
        }
        else
        {
            ofile = &ofileTrain;
            ofileXor = &ofileXorTrain;
            ofileErrDesc = &ofileErrDescTrain;
            ofileOg = &ofileOgTrain;

            dataFile = dataTrainFile;
            xorFile = xorTrainFile;
            errDescFile = errDescTrainFile;
            ogFile = ogTrainFile;

            info[5]++;
        }

        // Write the new frame (with errors) to the output file
        ofile->write((char *)frame.data(), frame.size());
        if (ofile->fail())
        {
            std::cout << "Error: writing to file: " << dataFile << " failed on frame " << info[3] << std::endl;
            return 1;
        }
        if (info[3] == 1)
        {
            std::cout << "First frame size: " << frame.size() << std::endl;
        }

        if (toXOR) {
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
        }

        if (toErrDesc) {
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

        if (toOg) {
            // Write the original frame to the output file
            ofileOg->write((char *)ogFrame.data(), ogFrame.size());
            if (ofileOg->fail())
            {
                std::cout << "Error: writing to file: " << ogFile << " failed on frame " << info[3] << std::endl;
                return 1;
            }
        }

        // Report progress
        if (info[3] % 10000 == 0) {
            std::cout << "Frames processed and saved " << info[3] << std::endl;
        }
    }

    ifile.close();
    ofileTrain.close();
    ofileXorTrain.close();
    ofileErrDescTrain.close();
    ofileTest.close();
    ofileXorTest.close();
    ofileErrDescTest.close();

    std::cout << std::endl << "=== INFO ===" << std::endl;
    std::cout << "Frames read: " << info[0] << std::endl;
    std::cout << "IPv4 frames: " << info[1] << std::endl;
    std::cout << "Encoding fails: " << info[2] << std::endl;
    std::cout << "Frames encoded, processed and written: " << info[3] << std::endl;
    std::cout << "Total frames in test data: " << info[4] << std::endl;
    std::cout << "Total frames in training data: " << info[5] << std::endl;
    std::cout << "IPv4 frames longer than 1514 bytes: " << info[6] << std::endl;

    return 0;
}