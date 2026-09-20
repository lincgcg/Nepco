CXX ?= g++
PKGCONF ?= pkg-config
TARGET = analysis
SRCS = analysis.cpp

ONNX_ROOT ?= /home/ubuntu/pre2/onnx
MODEL_PATH ?= /home/ubuntu/pre2/models/output.preprocessed.onnx
VOCAB_PATH ?= /home/ubuntu/pre2/vocab/hex_vocab.txt
NUM_CLASSES ?= 2
SEQ_LENGTH ?= 128
MAX_PACKETS_PER_FLOW ?= 5
TARGET_BYTE_LEN ?= 128

CXXFLAGS += -O3 -std=c++17 -I$(ONNX_ROOT)/include $(shell $(PKGCONF) --cflags libdpdk)
CXXFLAGS += -DMODEL_PATH=\"$(MODEL_PATH)\"
CXXFLAGS += -DVOCAB_PATH=\"$(VOCAB_PATH)\"
CXXFLAGS += -DNUM_CLASSES=$(NUM_CLASSES)
CXXFLAGS += -DSEQ_LENGTH=$(SEQ_LENGTH)
CXXFLAGS += -DMAX_PACKETS_PER_FLOW=$(MAX_PACKETS_PER_FLOW)
CXXFLAGS += -DTARGET_BYTE_LEN=$(TARGET_BYTE_LEN)

LDFLAGS += -L$(ONNX_ROOT)/lib
LDLIBS += -lonnxruntime $(shell $(PKGCONF) --libs libdpdk) -lpthread

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
