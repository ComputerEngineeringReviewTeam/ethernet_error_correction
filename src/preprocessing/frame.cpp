// Author: Marek Szymański
// Description: 

// TODO: split into files
// TODO: clean up

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <random>
#include <unordered_set>

#include "encodingtables.h"
#include "crc32.hpp"
#include "encoding.hpp"
#include "transerrors.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;


/**
 * @brief Creates bytesVec with 6 random bytes, immulating MAC address
 * 
 * @return bytesVec 
 */
bytesVec randomMAC() {
    bytesVec mac;
    mac.reserve(6);
    for (int i = 0; i < 6; i++) {
        mac.push_back(rand() % 256);
    }
    return mac;
}

/**
 * @brief Parses frame from string to bytesVec
 * @details Frame is in the format of Wireshark capture, ie.
 *          chain of bytes written in hexadecimal, separated by '|'
 * 
 * @param line      - string representation a Ethernet frame
 * @return bytesVec - vector of bytes
 */
bytesVec parseFrame(std::string line) {
    bytesVec bytes;
    std::string token;
    std::stringstream frame(line);
    while (getline(frame, token, '|')) {
        try {
            bytes.push_back(std::stoi(token, nullptr, 16));
        } catch (std::invalid_argument& e) {
            std::cout << "Invalid argument: " << token << "|" << std::endl;
        }
    }
    return bytes;
}

/**
 * @brief Parses frames from file to vector of bytesVec
 * @details Opens the Wireshark capture file pointed to by filename and reads frames from it
 *          The file contains Ethernet frames written in the following format: |0   |32|4a|1c|09|f2|55|7d|9a|...
 *          As additional informations are added by Wireshark (timestamp etc.), the function filters only lines starting with '|0'
 *          To make parsing easier the beginning "|0   " is removed from each line containing a frame
 * 
 * @param filename                  - name of the file containing frames
 * @return std::vector<bytesVec>    - vector of frames, each frame is a vector of bytes
 */
std::vector<bytesVec> readFrames(std::string filename) {
    std::vector<bytesVec> frames;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            if (line[0] == '|' && line[1] == '0') {     // filter only lines containing frames (start with '|0')
                line.erase(0, 6);                       // remove front padding added by Wireshark
                frames.push_back(parseFrame(line));     
            }
        }
    } else {
        std::cout << "Error opening file" << std::endl;
    }

    return frames;
}

/**
 * @brief Prints bytesVec with std::cout as series of bytes in hex, separated by ' ' and followed by '::' and size in bytes
 * 
 * @param bytes - vector of bytes to be printed
 */
void printBytesVec(const bytesVec& bytes) {
    for (int i = 0; i < bytes.size(); i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i] << " ";
    }
    std::cout << ":: " << std::dec << bytes.size() << "b" << std::endl;

}

/**
 * @brief Write bytesVec as vector bytes to binary file
 * 
 * @param bytes     - vector of bytes to be written
 * @param filename  - name of the file to write to
 */
void writeBytesVec(const bytesVec& bytes, std::string filename) {
    std::ofstream file;
    file.open(filename, std::ios::binary);
    file.write((char*)bytes.data(), bytes.size());
    file.close();
}

/**
 * @brief Padds the bytesVec with given (or default = 0) padding to the given size
 * 
 * @param bytes     - vector of bytes to be padded, not modified
 * @param size      - size to which the vector should be padded
 * @param padding   - byte to be used as padding, default is 0
 * @return bytesVec - padded vector of bytes
 */
bytesVec rightPadBytesVec(const bytesVec& bytes, int size, byte padding = 0) {
    bytesVec padded = bytes;
    padded.resize(size, padding);
    return padded;
}





int main() {
    auto testFrames = readFrames("../data/raw/capture_test.txt");
    bytesVec testFrame = testFrames[2];
    printBytesVec(testFrame);
    std::cout << "\tTEST FRAME" << std::endl;
    if (testFrame.size() % 4 != 0) {
        testFrame = rightPadBytesVec(testFrame, testFrame.size() + 4 - testFrame.size() % 4);
    }
    bytesVec encoded = encodings::encodeBytesVec8b10b(testFrame);
    printBytesVec(encoded);
    std::cout << "\tENCODED FRAME" << std::endl;
    bytesVec decoded = encodings::decodeBytesVec10b8b(encoded);
    std::cout << "\n";
    printBytesVec(decoded);
    std::cout << "\tDECODED FRAME" << std::endl;
    printBytesVec(testFrame);
    std::cout << "\tTEST FRAME" << std::endl;

    std::cout << std::endl << std::endl;

    std::mt19937 gen;
    std::unordered_map<std::string, int> targets = {{"DestMAC", 1}, {"SourceMAC", 1}, {"EtherType", 1}, {"CRC", 0}, {"Data", 1}};
    bytesVec flipped = transerrors::flipBits(encoded, transerrors::getPositionsInEncodedFrame(targets, gen, testFrame.size()));
    printBytesVec(flipped);
    std::cout << "\tFLIPPED FRAME" << std::endl;
    printBytesVec(encodings::decodeBytesVec10b8b(flipped));
    std::cout << "\tDECODED FRAME" << std::endl;
    printBytesVec(testFrame);
    std::cout << "\tTEST FRAME" << std::endl;
    return 0;
}