// Author: Marek Szymański
// Description: 

// TODO: split into files
// TODO: add documentantion and comments
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

namespace crc32 {
    // implementation by https://gist.github.com/timepp/1f678e200d9e0f2a043a9ec6b3690635
    // modified to use bytesVec
    // code of generate_table() unchanged

    /**
     * @brief Generates CRC32 reference table, credit to https://github.com/timepp
     * @details Generates a reference table of 256 CRC32 values, used for CRC32 calculation
     *          Implementation by https://gist.github.com/timepp/1f678e200d9e0f2a043a9ec6b3690635
     *          Not modified
     * 
     * @param table - pointer to the table to be filed with CRC32 values
     */
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

    /**
     * @brief Calculates CRC32 of the given bytesVec, using pregenerated CRC32 reference table
     * 
     * @param table     - pointer to the pregenerated CRC32 reference table (genreted by generate_table())
     * @param data      - vector of bytes to calculate CRC32 of
     * @return uint32_t - CRC32 value of the data, as uint32_t 
     */
	uint32_t crc(uint32_t* table, const bytesVec& data)
	{
		uint32_t crc = 0xFFFFFFFF;
        for (int i = 0; i < data.size(); i++) {
            crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
		return crc ^ 0xFFFFFFFF;
	}

    /**
     * @brief Converts CRC32 value from uint32_t to bytesVec of 4 bytes
     * 
     * @param crc       - CRC32 value to be converted
     * @return bytesVec - vector of 4 bytes equal to the CRC32 value of crc 
     */
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
    /**
     * @brief Bit mask to extract the younger 5 bits from byte
     * 
     */
    const std::uint8_t EDCBA = 0b00011111;
    /**
     * @brief Bit mask to extract the older 3 bits from byte
     * 
     */
    const std::uint8_t HGF   = 0b11100000;

    /**
     * @brief 5b/6b encoding of a single symbol
     * @details Encodes 5 bits to 6 bits, using the 5b/6b encoding table, with th respect of running disparity (used in 8b/10b encoding)
     *          The table, defined in encodingtables.h, is a map of pairs of 6bit symbols, former for RD=-1 and latter for RD=1
     *          As it's not really feasible to pass around packets of 5 or 6 bits, the function takes 8bit input and returns 8bit output
     *          the "additional" bits are filled with '0's 
     *          Throws std::invalid_argument if RD is neither -1 nor 1
     * 
     * @param input         - symbol to encode - uint8_t used to transport 5 bit symbol of 5b/6b encoding, with the "additional" bits are set to 0
     * @param RD            - running disparity, -1 or 1
     * @return std::uint8_t - encoded symbol - uint8_t used to transport 6 bit symbol of 5b/6b encoding, with the "additional" bits are set to 0
     */
    std::uint8_t encodeByte5b6b(std::uint8_t input, int RD) {
        if (RD == -1)
            return CODE_5b6b.at(input).first;
        else if (RD == 1)
            return CODE_5b6b.at(input).second;
        else
            throw std::invalid_argument("RD must be -1 or 1");
    }

    /**
     * @brief 3b/4b encoding of a single symbol
     * @details Encodes 3 bits to 4 bits, using the 3b/4b encoding table, with th respect of running disparity (used in 8b/10b encoding)
     *          The table, defined in encodingtables.h, is a map of pairs of 4bit symbols, former for RD=-1 and latter for RD=1
     *          As it's not really feasible to pass around packets of 3 or 4 bits, the function takes 8bit input and returns 8bit output
     *          the "additional" bits are filled with '0's 
     *          Throws std::invalid_argument if RD is neither -1 nor 1
     * 
     * @param input         - symbol to encode - uint8_t used to transport 3 bit symbol of 3b/4b encoding, with the "additional" bits are set to 0
     * @param RD            - running disparity, -1 or 1
     * @return std::uint8_t - encoded symbol - uint8_t used to transport 4 bit symbol of 3b/4b encoding, with the "additional" bits are set to 0
     */
    std::uint8_t encodeByte3b4b(std::uint8_t input, int RD) {
        if (RD == -1)
            return CODE_3b4b.at(input).first;
        else if (RD == 1)
            return CODE_3b4b.at(input).second;
        else
            throw std::invalid_argument("RD must be -1 or 1");
    }

    /**
     * @brief Counts high bits ('1') in the given 10bit symbol
     * @details While such implementation is obviously inefficient, it is used to avoid relying on compiler's built-in functions
     * 
     * @param input - 10bit symbol to count high bits in
     * @return int  - number of high bits in the symbol
     */
    int count_ones(symbol10 input) {
        int count = 0;
        for (int i = 0; i < 8; i++) {
            count += (input >> i) & 1;
        }
        return count;
    }

    /**
     * @brief 8b/10b encoding of a single symbol
     * @details Encodes 8 bits to 10 bits, using 5b/6b and 3b/4b encodings
     *          See https://en.wikipedia.org/wiki/8b/10b_encoding for more detailed explanation of the algorithm
     *          Uses encodeByte5b6b() to encode the younger 5 bits of the input symbol into the older 6 bits of the output symbol
     *          and encodeByte3b4b() to encode the older 3 bits of the input symbol into the younger 4 bits of the output symbol
     *          In order to avoid long series of 0 or 1 the 8b/10b encoding tracks the "running disparity" (RD) 
     *          RD is the difference between the number of '1' and '0' bits in the previous encoded symbol
     *          and is used to choose the correct 5b/6b and 3b/4b encoding of the next symbol
     * 
     * @param input     - symbol to encode - uint8_t used to transport 8 bit symbol of 8b/10b encoding, with the "additional" bits are set to 0
     * @param RD        - running disparity, -1 or 1
     * @return symbol10 - uint16_t used to transport 10 bit symbol of 8b/10b encoding, with the "additional" bits are set to 0
     */
    symbol10 encodeByte8b10b(std::uint8_t input, int RD) {
        std::uint8_t x = input & EDCBA;
        std::uint8_t y = input >> 5;
        std::uint8_t abcdei = encodeByte5b6b(x, RD);
        if (y == 7) {                                               // check for special case of D.x.P7/D.x.A7
            if ((RD == -1 && (x == 17 || x == 18 || x == 20)) ||    // see https://en.wikipedia.org/wiki/8b/10b_encoding for explanation
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

    /**
     * @brief Decodes 10bit symbol to 8bit symbol
     * @details Decodes 10bit symbol to 8bit symbol, using 5b/6b and 3b/4b decodings
     *          See https://en.wikipedia.org/wiki/8b/10b_encoding for more detailed explanation of the algorithm
     *          Uses DECODE_5b6b and DECODE_3b4b tables to decode the 10bit symbol
     *          The function first extracts the younger 6 bits of the 10bit symbol and decodes them using DECODE_5b6b table
     *          Then it extracts the older 4 bits of the 10bit symbol and decodes them using DECODE_3b4b table
     *          The decoded 6 and 4 bits are then combined into 8bit symbol
     * 
     * @param symbol        - 10bit symbol to decode
     * @return std::uint8_t - 8bit decoded symbol
     */
    std::uint8_t decodeSymbol10b8b(symbol10 symbol) {
        std::uint8_t abcdei = symbol >> 4;
        std::uint8_t fghj = symbol & 0b1111;
        std::uint8_t x=0, y=0;
        try {
            x = DECODE_5b6b.at(abcdei);
        } catch (std::out_of_range& e) {
        }
        try {
            y = DECODE_3b4b.at(fghj);
        } catch (std::out_of_range& e) {
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