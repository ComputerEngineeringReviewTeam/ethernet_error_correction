// Author: Marek Szymański
// Description: 

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <random>
#include <unordered_set>

#include "../include/crc32.hpp"
#include "../include/encoding.hpp"
#include "../include/transerrors.hpp"
#include "../include/frame.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

int main() {
    auto testFrames = readFrames("../data/raw/capture1.txt");
    // std::string prepFile = "../data/prep/capture_test.dat";

    // std::ofstream prepStream(prepFile, std::ios::binary);
    // if (!prepStream.is_open()) {
    //     std::cerr << "Error: could not open file " << prepFile << std::endl;
    //     return 1;
    // }
    //int written = 0;
    std::unordered_map<int, int> sizes(100);
    for (auto f : testFrames) {
        std::uint16_t type = (f[ETH2_TYPE_OFFSET] << BYTE_SIZE) + f[ETH2_TYPE_OFFSET + 1];
        if (type != ETH2_TYPE_IP4) {
            continue;
        }
        //     auto bytes = rightPadBytesVec(f, ETH2_MAX_FRAME_SIZE, 0);
        //     prepStream.write((char*)bytes.data(), bytes.size());
        //     written++;
        // }
        if (f.size() > 100)
            continue;
        if (sizes.find(f.size()) == sizes.end()) {
            sizes[f.size()] = 1;
        } else {
            sizes[f.size()]++;
        }
    }
    for (auto s : sizes) {
        std::cout << s.first << ": " << s.second << std::endl;
    }
    // prepStream.close();
    // std::cout << "Written " << written << " frames to " << prepFile << std::endl;

    // auto prepFrames = readFramesFromBinary(prepFile, ETH2_MAX_FRAME_SIZE);
    // std::cout << "Read " << prepFrames.size() << " frames from " << prepFile << std::endl;

    // std::cout << "Frame[2]: ";
    // printBytesVec(testFrames[2]);
    // std::cout << "Frame[2]: ";
    // printBytesVec(prepFrames[2]);



//    bytesVec testFrame = testFrames[2];
//    printBytesVec(testFrame);
//    std::cout << "\tTEST FRAME" << std::endl;
//    if (testFrame.size() % 4 != 0) {
//        testFrame = rightPadBytesVec(testFrame, testFrame.size() + 4 - testFrame.size() % 4);
//    }
//    bytesVec encoded = encodings::encodeBytesVec8b10b(testFrame);
//    printBytesVec(encoded);
//    std::cout << "\tENCODED FRAME" << std::endl;
//    bytesVec decoded = encodings::decodeBytesVec10b8b(encoded);
//    std::cout << "\n";
//    printBytesVec(decoded);
//    std::cout << "\tDECODED FRAME" << std::endl;
//    printBytesVec(testFrame);
//    std::cout << "\tTEST FRAME" << std::endl;
//
//    std::cout << std::endl << std::endl;
//
//    std::mt19937 gen;
//    std::unordered_map<std::string, int> targets = {{"DestMAC", 1}, {"SourceMAC", 1}, {"EtherType", 1}, {"CRC", 0}, {"Data", 1}};
//    bytesVec flipped = transerrors::flipBits(encoded, transerrors::getPositionsInEncodedFrame(targets, gen, testFrame.size()));
//    printBytesVec(flipped);
//    std::cout << "\tFLIPPED FRAME" << std::endl;
//    printBytesVec(encodings::decodeBytesVec10b8b(flipped));
//    std::cout << "\tDECODED FRAME" << std::endl;
//    printBytesVec(testFrame);
//    std::cout << "\tTEST FRAME" << std::endl;
    return 0;
}