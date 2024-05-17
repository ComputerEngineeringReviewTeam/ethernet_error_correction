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

int main()
{
    auto testFrames = readFrames("../data/raw/capture_test.txt");
    for (auto f : testFrames)
    {
        std::uint16_t type = (f[ETH2_TYPE_OFFSET] << BYTE_SIZE) + f[ETH2_TYPE_OFFSET + 1];
        if (type != ETH2_TYPE_IP4)
            continue;

        bytesVec frame = rightPadBytesVec(f, f.size() + 4 - f.size() % 4, 0);
        std::cout << "Size: " << f.size()<< "Padded size: " << frame.size() <<  std::endl; 
        bytesVec encoded = encodings::encodeBytesVec8b10b(frame);
    }
    // std::cout << "\tTEST FRAME" << std::endl;
    // if (testFrame.size() % 4 != 0)
    // {
    //     testFrame = rightPadBytesVec(testFrame, testFrame.size() + 4 - testFrame.size() % 4);
    // }
    // bytesVec encoded = encodings::encodeBytesVec8b10b(testFrame);

    // printBytesVec(decoded);
    // std::cout << "\tDECODED FRAME" << std::endl;
    // printBytesVec(testFrame);
    // std::cout << "\tTEST FRAME" << std::endl;

    return 0;
}