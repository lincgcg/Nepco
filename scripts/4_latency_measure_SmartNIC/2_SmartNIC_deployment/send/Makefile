APP = dpdk_sender
SRCS = replay4.cpp
PKGCONF ?= pkg-config
CXX ?= g++
PORT_ID ?= 1

CXXFLAGS += -O3 -g -std=c++17 -DPORT_ID=$(PORT_ID) $(shell $(PKGCONF) --cflags libdpdk)
LDFLAGS += $(shell $(PKGCONF) --libs libdpdk) -pthread -lpcap

all: $(APP)

$(APP): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(APP)

.PHONY: all clean
