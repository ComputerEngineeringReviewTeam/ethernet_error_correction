// Author: Marek Szymański
// Description: Constants used in the project

#pragma once

#include <cstdint>

const int BYTE_SIZE = 8;
const int BYTE_MAX = 256;

const int ETH2_HEADER_SIZE = 14;
const int MAC_SIZE = 6;
const int ETH2_SRC_MAC_OFFSET = 6;
const int ETH2_DST_MAC_OFFSET = 0;
const int ETH2_TYPE_SIZE = 2;
const int ETH2_TYPE_OFFSET = 12;
const int ETH2_PAYLOAD_OFFSET = 14;
const int ETH2_MIN_PAYLOAD_SIZE = 46;
const int ETH2_MAX_PAYLOAD_SIZE = 1500;
const int ETH2_MIN_FRAME_SIZE = 64;
const int ETH2_MAX_FRAME_SIZE = 1518;
const int ETH2_CRC_SIZE = 4;
const int ETH2_CRC_BACK_OFFSET = 4;

const int IP4_HEADER_SIZE = 20;
const int IP4_ADDR_SIZE = 4;
const int IP4_SRC_ADDR_OFFSET = 14;
const int IP4_DST_ADDR_OFFSET = 18;

const std::uint16_t ETH2_TYPE_IP4 = 0x0800;
const std::uint16_t ETH2_TYPE_IP6 = 0x08dd;

const int ETH2_ENC_MAC_BIT_SIZE = 60;
const int ETH2_ENC_SRC_MAC_BIT_OFFSET = 60;
const int ETH2_ENC_DST_MAC_BIT_OFFSET = 0;
const int ETH2_ENC_CRC_BIT_SIZE = 20;
const int ETH2_ENC_TYPE_BIT_OFFSET = 120;
