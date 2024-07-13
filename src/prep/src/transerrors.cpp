// Author: Marek Szymański
// Description: Flipping bits in the fields of a 8b/10b encoded Ethernet II frame

#include "../include/transerrors.hpp"

namespace transerrors {

    bytesVec flipBits(const bytesVec&data, const std::unordered_set<int>& positions) {
        bytesVec flipped = data;
        for (int pos : positions) {
            try {
                flipped.at((int) (pos / 8)) ^= (1 << std::abs((pos % 8) - 8));
            } catch (std::out_of_range& e) {
                //std::cout << "Out of range: " << pos << std::endl;
            }
        }
        return flipped;
    }

    symbolVec flipBits(const symbolVec&data, const std::unordered_set<int>& positions) {
        symbolVec flipped = data;
        for (int pos : positions) {
            try {
                flipped.at((int) (pos / 10)) ^= (1 << std::abs((pos % 10) - 9));
            } catch (std::out_of_range& e) {
                //std::cout << "Out of range: " << pos << std::endl;
            }
        }
        return flipped;
    }

    int getPositionsInFirstEncodedMAC(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, 59);
        return dist(gen);
    }

    int getPositionsInSecondEncodedMAC(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(60, 119);
        return dist(gen);
    }

    int getPositionsInEncodedEtherType(std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(120, 139);
        return dist(gen);
    }

    int getPositionsInEncodedCRC(std::mt19937& gen, int plainframeSize) {
        std::uniform_int_distribution<int> dist(plainframeSize * 8 - 40, plainframeSize * 8 - 1);
        return dist(gen);
    }

    int getPositionsInEncodedFrameData(std::mt19937& gen, int plainframeSize) {
        std::uniform_int_distribution<int> dist(140, plainframeSize * 8 - 40 - 1);
        return dist(gen);
    }

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

    bytesVec flipRandomBits(const bytesVec& data, int count, std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, data.size() * 8 - 1);
        std::unordered_set<int> positions;
        int uniqueErrors = 0;
        while (uniqueErrors < count) {
            int pos = dist(gen);
            auto inserted = positions.insert(pos);
            if (inserted.second) {
                uniqueErrors++;
            }
        }
        return flipBits(data, positions);
    }

    bytesVec flipRandomBits(const bytesVec& data, const std::vector<double>& chances, std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, data.size() * 8 - 1);
        std::uniform_real_distribution<double> next(0, 1);
        std::unordered_set<int> positions;

        positions.insert(dist(gen));    // guaranteed to insert at least one position

        for (int i = 0; i < chances.size(); i++)
        {
            double chanceForNext = next(gen);
            if (chanceForNext < chances[i])
            {
                positions.insert(dist(gen));
            }
        }
        
        return flipBits(data, positions);
    }

    std::unordered_set<int> getRandomPositions(int count, int size, std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, size - 1);
        std::unordered_set<int> positions;
        int uniqueErrors = 0;
        while (uniqueErrors < count) {
            int pos = dist(gen);
            auto inserted = positions.insert(pos);
            if (inserted.second) {
                uniqueErrors++;
            }
        }
        return positions;
    }

    std::unordered_set<int> getRandomPositions(const std::vector<double>& chances, int size, std::mt19937& gen) {
        std::uniform_int_distribution<int> dist(0, size - 1);
        std::uniform_real_distribution<double> next(0, 1);
        std::unordered_set<int> positions;

        for (int i = 0; i < chances.size(); i++)
        {
            double chanceForNext = next(gen);
            if (chanceForNext < chances[i])
            {
                positions.insert(dist(gen));
            }
        }
        
        return positions;
    }

    std::unordered_map<std::string, int> classifyPositions(const std::unordered_set<int>& positions, int plainframeSize) {
        auto classified = FIELDS_TEMPLATE_MAP;

        for (int pos : positions) {
            if (pos < 60) {
                classified["DestMAC"]++;
            } else if (pos < 120) {
                classified["SourceMAC"]++;
            } else if (pos < 140) {
                classified["EtherType"]++;
            } else if (pos < plainframeSize * 8 - 40) {
                classified["Data"]++;
                if (pos < 340) {
                    classified["IPHeader"]++;
                } else {
                    classified["IPData"]++;
                }
            } else {
                classified["CRC"]++;
            }
        }

        return classified;
    }
};