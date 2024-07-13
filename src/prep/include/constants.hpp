// Author: Marek Szymański
// Description: Constants used in the project

#pragma once

#include <cstdint>
#include <vector>

const int BYTE_SIZE = 8;
const int BYTE_MAX = 256;

// all offsets are calculated in bits from the bit 0 of the Ethernet II frame

const int ETH2_HEADER_SIZE = 14;                // Ethernet II header size
const int ETH2_SRC_MAC_OFFSET = 6;              // Ethernet II source MAC address offset
const int ETH2_DST_MAC_OFFSET = 0;              // Ethernet II destination MAC address offset
const int ETH2_TYPE_SIZE = 2;                   // Ethernet II type size
const int ETH2_TYPE_OFFSET = 12;                // Ethernet II type offset
const int ETH2_PAYLOAD_OFFSET = 14;             // Ethernet II payload offset
const int ETH2_MIN_PAYLOAD_SIZE = 46;           // Ethernet II minimum payload size
const int ETH2_MAX_PAYLOAD_SIZE = 1500;         // Ethernet II maximum payload size
const int ETH2_MIN_FRAME_SIZE = 64;             // Ethernet II minimum frame size
const int ETH2_MAX_FRAME_SIZE = 1518;           // Ethernet II maximum frame size
const int ETH2_CRC_SIZE = 4;                    // Ethernet II CRC size
const int ETH2_CRC_BACK_OFFSET = 4;             // Ethernet II CRC back offset
const int ETH2_ENC_MAC_BIT_SIZE = 60;           // Ethernet II MAC bit size
const int ETH2_ENC_SRC_MAC_BIT_OFFSET = 60;     // Ethernet II source MAC bit offset
const int ETH2_ENC_DST_MAC_BIT_OFFSET = 0;      // Ethernet II destination MAC bit offset
const int ETH2_ENC_CRC_BIT_SIZE = 20;           // Ethernet II CRC bit size
const int ETH2_ENC_TYPE_BIT_OFFSET = 120;       // Ethernet II type bit offset

const std::uint16_t ETH2_TYPE_IP4 = 0x0800;     // IPv4 type code
const std::uint16_t ETH2_TYPE_IP6 = 0x08dd;     // IPv6 type code

const int MAC_SIZE = 6;                         // MAC address size

const int IP4_HEADER_SIZE = 20;                 // IPv4 header size
const int IP4_ADDR_SIZE = 4;                    // IPv4 address size
const int IP4_SRC_ADDR_OFFSET = 14;             // IPv4 source address offset
const int IP4_DST_ADDR_OFFSET = 18;             // IPv4 destination address offset

using byte = std::uint8_t;                      // 8-bit byte
using bytesVec = std::vector<byte>;             // vector of bytes
using symbol10 = std::uint16_t;                 // 10-bit symbol - stored on younger 10 bits of uint16_t
using symbolVec = std::vector<symbol10>;        // vector of 10-bit symbols
