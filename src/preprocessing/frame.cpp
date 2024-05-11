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



namespace errors {

    /**
    * @brief Negates bits at given positions in given vector of bytes
    * @details If the positions would be {0, 12, 17} then bits negated would be: 
    *          0th bit of 0th byte, 4th bit of 1st byte, 1st bit of 3rd byte and so on.
    *          Positions out of range are ignored.    
    * 
    * @param data      - vector of original bytes, not modified
    * @param positions - vector of positions to negate bits at, should be no longer than 8*data.size() (nr of bits in data)
    * @return          - vector of bytes with bits negated at given positions
    */
    bytesVec flipBits(const bytesVec&data, const std::unordered_set<int>& positions) {
        bytesVec flipped = data;
        for (int pos : positions) {
            try {
                flipped.at(pos / 8) ^= (1 << (pos % 8));
            } catch (std::out_of_range& e) {
                //std::cout << "Out of range: " << pos << std::endl;
            }
        }
        return flipped;
    }

    /**
     * @brief Get the position of random bit in destination MAC field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'destination MAC' field of a 8b/10b encoded Ethernet II frame
     *          the 'destination MAC' field is the first 6 bytes of the frame,
     *          which after encoding are the first 60 bits of the encoded frame (bytes 0 - 7 and the older half of byte 8)      
     * 
     * @param gen  - mt19937 random number generator
     * @return int - position of the bit
     */
    int getPositionsInFirstEncodedMAC(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, 59);
        return dist(gen);
    }

    /**
     * @brief Get the position of random bit in source MAC field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'source MAC' field of a 8b/10b encoded Ethernet II frame
     *          the 'source MAC' field are the bytes 7 - 12 of the frame,
     *          which after encoding are the bits 61 - 119 (indexing from 0) of the encoded frame (younger half of byte 8 and bytes 9 - 14)      
     * 
     * @param gen  - mt19937 random number generator
     * @return int - position of the bit
     */
    int getPositionsInSecondEncodedMAC(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(60, 119);
        return dist(gen);
    }

    /**
     * @brief Get the position of random bit in EtherType field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'EtherType' field of a 8b/10b encoded Ethernet II frame
     *          the 'EtherTypes' field are the bytes 13 - 14 of the frame,
     *          which after encoding are the bits 120 - 139 (indexing from 0) of the encoded frame (bytes 15 - 16 and the older half of byte 17)      
     * 
     * @param gen  - mt19937 random number generator
     * @return int - position of the bit
     */
    int getPositionsInEncodedEtherType(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(120, 139);
        return dist(gen);
    }

    /**
     * @brief Get the position of random bit in CRC field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'CRC' field of a 8b/10b encoded Ethernet II frame
     *          the 'CRC' field are the last 4 bytes (bytes size-4 - size-1) of the frame,
     *          which after encoding are the last 40 bits of the encoded frame 
     *          as such the length of the frame before encoding (which might introduce padding) should be passed as an argument
     * 
     * @param gen            - mt19937 random number generator
     * @param plainframeSize - length of the in bytes frame before encoding (which might introduce padding)
     * @return int           - position of the bit
     */
    int getPositionsInEncodedCRC(std::mt19937& gen, int plainframeSize) {
        std::uniform_int_distribution<int> dist(plainframeSize * 8 - 40, plainframeSize * 8 - 1);
        return dist(gen);
    }

    /**
     * @brief Get the position of random bit in Data field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'Data' field of a 8b/10b encoded Ethernet II frame
     *          the 'Data' field are the bytes 15 - size-5 of the frame (all bytes between the header and the last 4 bytes of CRC)
     *          which after encoding are the bits 140 - size*8 - 40 (indexing from 0) of the encoded frame 
     *          as such the length of the frame before encoding (which might introduce padding) should be passed as an argument
     * 
     * @param gen            - mt19937 random number generator
     * @param plainframeSize - length of the frame in bytes before encoding (which might introduce padding)
     * @return int           - position of the bit
     */
    int getPositionsInEncodedFrameData(std::mt19937& gen, int plainframeSize) {
        std::uniform_int_distribution<int> dist(140, plainframeSize * 8 - 40 - 1);
        return dist(gen);
    }

    /**
     * @brief Get the positions of random  bits within the fields of a 8b/10b encoded Ethernet II frame according to given map
     * @details Returns random positions of single bits within the given fields of a 8b/10b encoded Ethernet II frame   
     *          The map details how many errors should be introduced in each field
     *          The format would be: {{"DestMAC", (0-60)}, {"SourceMAC", (0-60)}, {"EtherType", (0-20)}, {"CRC", (0-*)}, {"Data", (0-40)}}
     *          possible ranges for each field given in the brackets
     *          (* - length of the frame in bytes before encoding * 10 - 140 (Ethernet II header) - 40 (CRC) ).
     *          Fields not present in the map are ignored
     *          The function returns std::unordered_set of unique positions of bits to be flipped (to be used in flipBits() function or inspected)
     * 
     * 
     * @param targets                  - map of fields and number of errors to introduce in each field
     * @param gen                      - mt19937 random number generator
     * @param plainframeSize           - length of the frame before encoding (which might introduce padding)
     * @return std::unordered_set<int> - set of positions of bits to be flipped
     */
    std::unordered_set<int> getPositionsInEncodedFrame(const std::unordered_map<std::string, int>& targets, std::mt19937& gen, int plainframeSize) {
        std::unordered_set<int> positions;
        std::string field = "";

        for (auto it = targets.begin(); it != targets.end(); it++) {
            field = it->first;
            int count = it->second;
            int uniqueErrors = 0;
            while (uniqueErrors < count) {
                int pos = 0;
                if (field == "DestMAC") {
                    pos = getPositionsInFirstEncodedMAC(gen);
                } else if (field == "SourceMAC") {
                    pos = getPositionsInSecondEncodedMAC(gen);
                } else if (field == "EtherType") {
                    pos = getPositionsInEncodedEtherType(gen);
                } else if (field == "CRC") {
                    pos = getPositionsInEncodedCRC(gen, plainframeSize);
                } else if (field == "Data") {
                    pos = getPositionsInEncodedFrameData(gen, plainframeSize);
                } else {
                    // std::cout << "Unknown field: " << field << std::endl;
                    break;
                }

                auto inserted = positions.insert(pos);
                if (inserted.second) {
                    uniqueErrors++;
                }
            }
        }
        return positions;
    }
};

int main() {
    auto testFrames = readFrames("../../data/raw/capture_test.txt");
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
    bytesVec flipped = errors::flipBits(encoded, errors::getPositionsInEncodedFrame(targets, gen, testFrame.size()));
    printBytesVec(flipped);
    std::cout << "\tFLIPPED FRAME" << std::endl;
    printBytesVec(encodings::decodeBytesVec10b8b(flipped));
    std::cout << "\tDECODED FRAME" << std::endl;
    printBytesVec(testFrame);
    std::cout << "\tTEST FRAME" << std::endl;
    return 0;
}