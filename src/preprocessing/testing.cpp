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

#include "crc32.hpp"
#include "encoding.hpp"
#include "transerrors.hpp"
#include "frame.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

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