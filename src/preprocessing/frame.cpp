// Author: Marek Szymański
// Description: 

// TODO: split into files
// TOOD: add documentantion and comments
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


using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

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
            // filter only lines containing frames 
            if (line[0] == '|' && line[1] == '0') {
                line.erase(0, 6);                       // remove front padding from wireshark
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

void writeBytesVec(const bytesVec& bytes, std::string filename) {
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

namespace crc32 {
    // implementation by https://gist.github.com/timepp/1f678e200d9e0f2a043a9ec6b3690635
    // modified to use bytesVec
    // code of generate_table() unchanged
	void generate_table(uint32_t *table)
	{
		uint32_t polynomial = 0xEDB88320;
		for (uint32_t i = 0; i < 256; i++) 
		{
			uint32_t c = i;
			for (size_t j = 0; j < 8; j++) 
			{
				if (c & 1) {
					c = polynomial ^ (c >> 1);
				}
				else {
					c >>= 1;
				}
			}
			table[i] = c;
		}
	}

	uint32_t crc(uint32_t* table, const bytesVec& data)
	{
		uint32_t crc = 0xFFFFFFFF;
        for (int i = 0; i < data.size(); i++) {
            crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
		return crc ^ 0xFFFFFFFF;
	}

    bytesVec toBytesVec(uint32_t crc) {
        bytesVec bytes;
        bytes.reserve(4);
        for (int i = 3; i >= 0; i--) {
            bytes.push_back((crc >> (i * 8)) & 0xFF);
        }
        return bytes;
    }
};

namespace encodings {
    const std::uint8_t EDCBA = 0b00011111;
    const std::uint8_t HGF = 0b11100000;

    std::uint8_t encodeByte5b6b(std::uint8_t input, int RD) {
        if (RD == -1)
            return CODE_5b6b.at(input).first;
        else if (RD == 1)
            return CODE_5b6b.at(input).second;
        else
            throw std::invalid_argument("RD must be -1 or 1");
    }

    std::uint8_t encodeByte3b4b(std::uint8_t input, int RD) {
        if (RD == -1)
            return CODE_3b4b.at(input).first;
        else if (RD == 1)
            return CODE_3b4b.at(input).second;
        else
            throw std::invalid_argument("RD must be -1 or 1");
    }

    // inefficient, but for portability reasons - to not rely on compiler's built-in functions
    int count_ones(symbol10 input) {
        int count = 0;
        for (int i = 0; i < 8; i++) {
            count += (input >> i) & 1;
        }
        return count;
    }

    symbol10 encodeByte8b10b(std::uint8_t input, int RD) {
        std::uint8_t x = input & EDCBA;
        std::uint8_t y = input >> 5;
        std::uint8_t abcdei = encodeByte5b6b(x, RD);
        if (y == 7) {                                               // check for special case of D.x.P7/D.x.A7
            if ((RD == -1 && (x == 17 || x == 18 || x == 20)) ||
                (RD ==  1 && (x == 11 || x == 13 || x == 24))) {
                    y = 8;
                }
        }
        std::uint8_t fghj = encodeByte3b4b(y, RD);
        symbol10 output = 0;
        output = (abcdei << 4) | fghj;
        return output;
    }

    bytesVec encodeBytesVec8b10b(const bytesVec& data) {
        int RD = -1;
        bytesVec encoded;           // encoded data, in a vector of bytes (bytesVec)
        std::uint64_t buffer = 0;   // buffer used to hold 4 10bit symbols before they are saved to encoded bytesVec (as 5 bytes)
        int bitsInBuffer = 0;       // state of the buffer
        symbol10 symbol;   
        uint8_t buff[5] ;        

        for (int i = 0; i < data.size(); i++) {
            // encode new symbol and update buffer
            symbol = encodeByte8b10b(data[i], RD);
            buffer = (buffer << 10) | symbol;
            bitsInBuffer += 10;

            // check for running disparity and update RD if needed
            // '1' in symbol10 (uint16_t really) is counted by count_ones()
            // '0' in uint16_t would be 16 - count_ones(),
            // but we are looking for the difference in a 10bit symbol, so 
            // '0' in symbol10 is 16 - count_ones() - 6 (6 '0' of front-padding)
            // diff = '1' - '0' = count_ones() - (16 - count_ones() - 6) = 2*count_ones() - 6
            int diff = 2 * count_ones(symbol) - 6;
            if (diff < 0)
                RD = -1;
            else if (diff > 0)
                RD = 1;

            // if there are 40 bits (4 symbols 10bit each or 5 bytes) in the buffer, save them to encoded bytesVec
            if (bitsInBuffer == 40) {
                // encoded.push_back(buffer & 0xFF);
                // encoded.push_back((buffer >> 8) & 0xFF);
                // encoded.push_back((buffer >> 16) & 0xFF);
                // encoded.push_back((buffer >> 24) & 0xFF);
                // encoded.push_back((buffer >> 32) & 0xFF);
                for (int j = 0; j < 5; j++) {
                    buff[j] = buffer & 0xFF;
                    buffer = buffer >> 8;
                    bitsInBuffer -= 8;
                }
                encoded.push_back(buff[4]);
                encoded.push_back(buff[3]);
                encoded.push_back(buff[2]);
                encoded.push_back(buff[1]);
                encoded.push_back(buff[0]);
                bitsInBuffer = 0;
                buffer = 0;
            }
        }

        // if there are less than 40 bits in the buffer, save them to encoded bytesVec
        // but first, pad buffer with '0' to make it 40 bits long
        if (bitsInBuffer > 0) {
            buffer = buffer << (40 - bitsInBuffer);
            encoded.push_back(buffer & 0xFF);
            encoded.push_back((buffer >> 8) & 0xFF);
            encoded.push_back((buffer >> 16) & 0xFF);
            encoded.push_back((buffer >> 24) & 0xFF);
            encoded.push_back((buffer >> 32) & 0xFF);
        }

        return encoded;
    }

    std::uint8_t decodeSymbol10b8b(symbol10 symbol) {
        std::uint8_t abcdei = symbol >> 4;
        std::uint8_t fghj = symbol & 0b1111;
        std::uint8_t x=0, y=0;
        try {
            x = DECODE_5b6b.at(abcdei);
        } catch (std::out_of_range& e) {
            //std::cout << "x not found:" << std::hex << std::setw(2) << std::setfill('0') << (int)abcdei << ":";
        }
        try {
            y = DECODE_3b4b.at(fghj);
        } catch (std::out_of_range& e) {
            //std::cout << "y not found:" << std::hex << std::setw(2) << std::setfill('0') << (int)fghj << ":";
        }

        return (y << 5) | x;
    }

    bytesVec decodeByteVec10b8b(const bytesVec& data) {
        int RD = -1;
        bytesVec decoded;
        std::uint64_t buffer = 0;
        int bitsInBuffer = 0;
        symbol10 symbol;

        for (int i = 0; i < data.size(); i++) {
            // push bytes into buffer
            buffer = (buffer << 8) | data[i];
            bitsInBuffer += 8;
            uint8_t buff[4];

            // after there are 40 bits (5 bytes, 4 symbols 10bit each) in the buffer
            // decode each symbol and save decoded bytes to decoded bytesVec
            if (bitsInBuffer == 40) {
                //std::cout << "buffer: " << std::hex << std::setw(16) << std::setfill('0') << buffer << std::endl;
                for (int j = 0; j < 4; j++) {
                    symbol = 0;
                    symbol = buffer & 0b0000001111111111;
                    // std::cout << "symbol: " << std::hex << std::setw(4) << std::setfill('0') << symbol << ":\n";
                    buffer = buffer >> 10;
                    bitsInBuffer -= 10;
                    //decoded.push_back(decodeSymbol10b8b(symbol));
                    buff[j] = decodeSymbol10b8b(symbol);
                }
                decoded.push_back(buff[3]);
                decoded.push_back(buff[2]);
                decoded.push_back(buff[1]);
                decoded.push_back(buff[0]);
            }
        }

        // if the buffer is not empty, pad it with '0' and decode the remaining symbols
        if (bitsInBuffer > 0) {
            buffer = buffer << (40 - bitsInBuffer);
            for (int j = 0; j < 4; j++) {
                symbol = buffer & 0x0000001111111111;
                buffer = buffer >> 10;
                bitsInBuffer -= 10;
                decoded.push_back(decodeSymbol10b8b(symbol));
            }
        }

        return decoded;
    }
};

namespace errors {

    /**
    * @brief negates bits at given positions in given vector of bytes
    * 
    * @param data vector of original bytes, not modified
    * @param positions vector of positions to negate bits at, should be no longer than 8*data.size() (nr of bits in data)
    * @return vector of bytes with bits negated at given positions
    * 
    * @details if the positions be [0, 12, 17] then bits negated would be: 0th bit of 0th byte,
    *          4th bit of 1st byte, 1st bit of 3rd byte and so on
    *          positions out of range are ignored
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

    // returns random position of single bit within the 'destination MAC' field of a 8b/10b encoded Ethernet II frame
    // the 'destination MAC' field is the first 6 bytes of the frame
    int getPositionsInFirstEncodedMAC(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, 59);
        return dist(gen);
    }

    // returns random position of single bit within the 'source MAC' field of a 8b/10b encoded Ethernet II frame
    // the 'source MAC' field is the next 6 bytes of the frame (bytes 7 to 12)
    int getPositionsInSecondEncodedMAC(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(60, 119);
        return dist(gen);
    }

    // returns random position of single bit within the 'EtherType' field of a 8b/10b encoded Ethernet II frame
    // the 'EtherType' field is the next 2 bytes of the frame (bytes 13 to 14)
    int getPositionsInEncodedEtherType(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(120, 139);
        return dist(gen);
    }

    // returns random position of single bit within the 'CRC' field of a 8b/10b encoded Ethernet II frame
    // the CRC field is the last 4 bytes of the frame (bytes size-4 to size-1)
    // as such the length of the frame before encoding (which might introduce padding) should be passed as an argument
    int getPositionsInEncodedCRC(std::mt19937& gen, int plainframeSize) {
        std::uniform_int_distribution<int> dist(plainframeSize * 8 - 40, plainframeSize * 8 - 1);
        return dist(gen);
    }

    // returns random position of single bit within the 'Data' field of a 8b/10b encoded Ethernet II frame
    // the Data field is the part of the frame between the 'EtherType' and 'CRC' fields (bytes 15 to size-4)
    // as such the length of the frame before encoding (which might introduce padding) should be passed as an argument
    int getPositionsInEncodedFrameData(std::mt19937& gen, int plainframeSize) {
        std::uniform_int_distribution<int> dist(140, plainframeSize * 8 - 40 - 1);
        return dist(gen);
    }

    // returns random positions of single bits within the given fields of a 8b/10b encoded Ethernet II frame
    // takes a map of fields and number of errors to introduce in each field
    // the fields are: "DestMAC", "SourceMAC", "EtherType", "CRC", "Data"
    // the number of errors should be less than 8*fieldSize
    // the length of the frame before encoding (which might introduce padding) should be passed as an argument
    // the function returns std::unordered_set of unique positions of bits to be flipped (to be used in flipBits() function or inspected)
    std::unordered_set<int> getPositionsInEncodedFrame(const std::unordered_map<std::string, int>& targets, std::mt19937& gen, int plainframeSize) {
        std::unordered_set<int> positions;
        std::uniform_int_distribution<int> dist(0, 7);
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
                    std::cout << "Unknown field: " << field << std::endl;
                    break;
                }

                auto [it, inserted] = positions.insert(pos);
                if (inserted) {
                    uniqueErrors++;
                }
            }
        }
        return positions;
    }
};

int main() {
    auto testFrames = readFrames("capture_test.txt");
    bytesVec testFrame = testFrames[2];
    printBytesVec(testFrame);
    std::cout << "\tTEST FRAME" << std::endl;
    bytesVec encoded = encodings::encodeBytesVec8b10b(testFrame);
    printBytesVec(encoded);
    std::cout << "\tENCODED FRAME" << std::endl;
    bytesVec decoded = encodings::decodeByteVec10b8b(encoded);
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
    printBytesVec(encodings::decodeByteVec10b8b(flipped));
    std::cout << "\tDECODED FRAME" << std::endl;
    printBytesVec(testFrame);
    std::cout << "\tTEST FRAME" << std::endl;
    return 0;
}