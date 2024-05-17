

#include "../crc32.hpp"
#include "../encoding.hpp"
#include "../transerrors.hpp"
#include "../frame.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;


int main() {
    std::string wiresharkFile = "../data/raw/capture_test.txt";
    std::string dataFile = "../data/prep/capture_test.dat";

    std::ifstream ifile(wiresharkFile);
    std::ofstream ofile(dataFile, std::ios::binary);

    if (!ifile.is_open()) {
        std::cerr << "Error: could not open file " << wiresharkFile << std::endl;
        return 1;
    }

    if (!ofile.is_open()) {
        std::cerr << "Error: could not open file " << dataFile << std::endl;
        return 1;
    }

    std::string wsline;
    bytesVec frame;
    std::uint16_t type;
    int complement4 = 0;
    bytesVec crcVec;
    uint32_t *crcTable;
    crc32::generate_table(crcTable);
    int info[] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> errors = {0.1, 0.01};
    std::mt19937 gen;
    while (getline(ifile, wsline)) {
        if (wsline[0] == '|' && wsline[1] == '0') {
            wsline.erase(0, 6);

            frame = parseFrame(wsline);
            info[0]++;

            type = (frame[ETH2_TYPE_OFFSET] << BYTE_SIZE) + frame[ETH2_TYPE_OFFSET + 1];
            if (type == ETH2_TYPE_IP4) {
                info[1]++;
                frame = randomizeMAC(frame, 2);
                frame = randomizeIPv4Addr(frame, 2);

                crcVec = crc32::toBytesVec(crc32::crc(crcTable, frame));
                frame.insert(frame.end(), crcVec.begin(), crcVec.end());

                frame = rightPadBytesVec(frame, frame.size() + (4 - (frame.size() % 4)), 0);
                frame = encodings::encodeBytesVec8b10b(frame);

                frame = transerrors::flipRandomBits(frame, errors, gen);

                frame = rightPadBytesVec(frame, ETH2_MAX_FRAME_SIZE, 0);

                ofile.write((char*)frame.data(), frame.size());
            }


        }
    }
    ifile.close();
    ofile.close();

    
}