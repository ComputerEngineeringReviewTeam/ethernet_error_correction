#include "../include/frame.hpp"

bytesVec randomMAC() {
    bytesVec mac;
    mac.reserve(6);
    for (int i = 0; i < 6; i++) {
        mac.push_back(rand() % 256);
    }
    return mac;
}

bytesVec parseFrame(std::string line) {
    bytesVec bytes;
    std::string token;
    std::stringstream frame(line);
    while (getline(frame, token, '|')) {
        try {
            bytes.push_back(std::stoi(token, nullptr, 16));
        } catch (std::invalid_argument& e) {
            //std::cout << "Invalid argument: " << token << "|" << std::endl;
        }
    }
    return bytes;
}

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

void printBytesVec(const bytesVec& bytes) {
    for (int i = 0; i < bytes.size(); i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i] << " ";
    }
    std::cout << ":: " << std::dec << bytes.size() << "b" << std::endl;

}

void writeBytesVecToBinary(const bytesVec& bytes, std::string filename) {
    std::ofstream file;
    file.open(filename, std::ios::binary);
    file.write((char*)bytes.data(), bytes.size());
    file.close();
}

bytesVec rightPadBytesVec(const bytesVec& bytes, int size, byte padding) {
    bytesVec padded = bytes;
    padded.resize(size, padding);
    return padded;
}

bytesVec randomizeMAC(const bytesVec& frame, int mode) {
    bytesVec randomized = frame;
    if (mode == 0 || mode == 2) {
        for (int i = ETH2_DST_MAC_OFFSET; i < ETH2_DST_MAC_OFFSET + MAC_SIZE; i++) {
            randomized[i] = rand() % BYTE_MAX;
        }
    }
    if (mode == 1 || mode == 2) {
        for (int i = ETH2_SRC_MAC_OFFSET; i < ETH2_SRC_MAC_OFFSET + MAC_SIZE; i++) {
            randomized[i] = rand() % BYTE_MAX;
        }
    }
    return randomized;
}

std::vector<bytesVec> readFramesFromBinary(std::string filename, int frameSize) {
    std::vector<bytesVec> frames;
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        int framesCount = fileSize / frameSize;
        for (int i = 0; i < framesCount; i++) {
            bytesVec frame(frameSize);
            file.read((char*)frame.data(), frameSize);
            frames.push_back(frame);
        }
    } else {
        std::cout << "Error opening file" << std::endl;
    }
    return frames;
}

std::uint32_t getEtherType(const bytesVec& frame) {
    return (frame[ETH2_TYPE_OFFSET] << 8) + frame[ETH2_TYPE_OFFSET + 1];
}

// IPv4

bytesVec randomizeIPv4Addr(const bytesVec& frame, int mode) {
    bytesVec randomized = frame;
    if (mode == 1 || mode == 2) {
        for (int i = IP4_DST_ADDR_OFFSET; i < IP4_DST_ADDR_OFFSET + IP4_ADDR_SIZE; i++) {
            randomized[i] = rand() % BYTE_MAX;
        }
    }
    if (mode == 0 || mode == 2) {
        for (int i = IP4_SRC_ADDR_OFFSET; i < IP4_SRC_ADDR_OFFSET + IP4_ADDR_SIZE; i++) {
            randomized[i] = rand() % BYTE_MAX;
        }
    }
    return randomized;
}