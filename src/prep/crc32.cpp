#include "crc32.hpp"

namespace crc32 {
	void generate_table(uint32_t *table)
	{
		uint32_t polynomial = 0xEDB88320;
		for (uint32_t i = 0; i < 256; i++) 
		{
			uint32_t c = i;
			for (int j = 0; j < 8; j++) 
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