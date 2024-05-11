#include "frame.hpp"

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
            std::cout << "Invalid argument: " << token << "|" << std::endl;
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

bytesVec rightPadBytesVec(const bytesVec& bytes, int size, byte padding = 0) {
    bytesVec padded = bytes;
    padded.resize(size, padding);
    return padded;
}