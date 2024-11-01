// Author: Marek Szymański
// errDescription: Convert Wireshark capture to binary data file

#include "../include/crc32.hpp"
#include "../include/encoding.hpp"
#include "../include/transerrors.hpp"
#include "../include/frame.hpp"
#include "../include/filesaver.hpp"



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
    FileSaver fs("../data/", "capture1.txt", "f100");

    // === variables ===
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
    std::uniform_int_distribution<int> dist(0, allToTestRatio-1);

    // Open the source file and output files
    if (!fs.openFiles())
    {
        return 1;
    }
    
    while (getline(fs.source(), wsline))
    {
        // Filter off non-data lines
        if (wsline[0] != '|' || wsline[1] != '0') {
            continue;
        }

        // Remove the first 6 characters - in Wireshark format they are always "|0   " before the actuall frame
        wsline.erase(0, 6);

        // Parse a new frame
        frame = parseFrame(wsline);
        info[0]++;

        // Skip frames of different type than the one we want
        if (getEtherType(frame) != etherType) {
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

        // Decide if the frame should be assigned to the training or testing data
        bool train = dist(gen) != 0;

        // perform bitwise XOR of the frame with errors and the error-free frame
        for (int i = 0; i < frame.size(); i++)
        {
            xorFrame[i] = xorFrame[i] ^ frame[i];
        }

        bool erred = false;
        for (int i = 0; i < frame.size(); i++)
        {
            if (frame[i] != ogFrame[i])
            {
                erred = true;
                info[7]++;
                break;
            }
        }
        

        // Write the frames and desc to the output files, exits if failed
        if (!fs.write(ogFrame, frame, xorFrame, errorsPosMap, train)) {
            return 1;
        }

        // Report progress
        if (info[3] % 10000 == 0) {
            std::cout << "Frames processed and saved " << info[3] << std::endl;
        }
    }

    fs.closeFiles();

    std::cout << std::endl << "=== INFO ===" << std::endl;
    std::cout << "Frames read: " << info[0] << std::endl;
    std::cout << "IPv4 frames: " << info[1] << std::endl;
    std::cout << "Encoding fails: " << info[2] << std::endl;
    std::cout << "Frames encoded, processed and written: " << info[3] << std::endl;
    std::cout << "Total frames in test data: " << info[4] << std::endl;
    std::cout << "Total frames in training data: " << info[5] << std::endl;
    std::cout << "IPv4 frames longer than 1514 bytes: " << info[6] << std::endl;
    std::cout << "Frames with real errors: " << info[7] << std::endl;

    return 0;
}