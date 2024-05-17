// Author: Marek Szymański
// errDescription: Convert Wireshark capture to binary data file

#include "../include/crc32.hpp"
#include "../include/encoding.hpp"
#include "../include/transerrors.hpp"
#include "../include/frame.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

int main()
{
    std::string wiresharkFile = "../data/raw/capture_test.txt";        // .txt file with Wireshark capture
    std::string dataFile = "../data/prep/capture_test.dat";            // .dat binary file with encoded frames with errors
    std::string xorFile = "../data/prep/capture_test_xor.dat";         // .dat binary file with error vectors
    std::string errDescFile = "../data/prep/capture_test_errDesc.csv"; // .csv file with error classes

    std::ifstream ifile(wiresharkFile);
    std::ofstream ofile(dataFile, std::ios::binary);
    std::ofstream ofileXor(xorFile, std::ios::binary);
    std::ofstream ofileErrDesc(errDescFile);

    if (!ifile.is_open())
    {
        std::cout << "Error: could not open: " << wiresharkFile << std::endl;
        return 1;
    }

    if (!ofile.is_open())
    {
        std::cout << "Error: could not open: " << dataFile << std::endl;
        return 1;
    }

    if (!ofileXor.is_open())
    {
        std::cout << "Error: could not open: " << xorFile << std::endl;
        return 1;
    }

    if (!ofileErrDesc.is_open())
    {
        std::cout << "Error: could not open: " << errDescFile << std::endl;
        return 1;
    }

    std::string wsline;
    bytesVec frame;
    std::uint16_t type;
    int complement4 = 0;
    bytesVec crcVec;
    uint32_t crcTable[256];
    std::cout << "Started\n";
    crc32::generate_table(crcTable);
    int info[] = {0, 0, 0, 0, 0, 0, 0, 0};
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
        std::cout << "Frame " << info[0] << std::endl;
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

        // Write the new frame (with errors) to the output file
        ofile.write((char *)frame.data(), frame.size());
        if (ofile.fail())
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
        ofileXor.write((char *)xorFrame.data(), xorFrame.size());
        if (ofileXor.fail())
        {
            std::cout << "Error: writing to file: " << xorFile << " failed on xor-frame " << info[3] << std::endl;
            return 1;
        }

        // Write the error class (errorsPosMap) of the frame to the errDescription file
        for (auto it = errorsPosMap.begin(); it != errorsPosMap.end(); it++)
        {
            if (std::next(it) == errorsPosMap.end())
            {
                ofileErrDesc << it->second << "\n";
            }
            else
            {
                ofileErrDesc << it->second << ",";
            }
        }
        if (ofileErrDesc.fail())
        {
            std::cout << "Error: writing to file: " << errDescFile << " failed on the desc of frame " << info[3] << std::endl;
            return 1;
        }
    }

    ifile.close();
    ofile.close();
    ofileXor.close();
    ofileErrDesc.close();

    std::cout << "Frames read: " << info[0] << std::endl;
    std::cout << "IPv4 frames: " << info[1] << std::endl;
    std::cout << "Encoding failed: " << info[2] << std::endl;
    std::cout << "Frames encoded, processed and written: " << info[3] << std::endl;

    return 0;
}