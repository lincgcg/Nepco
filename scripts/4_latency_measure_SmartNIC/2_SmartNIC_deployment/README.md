# SmartNIC Deployment

This folder contains the BF3 online evaluation for Nepco under a dynamic flow distribution.

## Build

Build the sender:

```bash
bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_build_sender.sh
```

Build the receiver:

```bash
MODEL_PATH=/absolute/path/to/output.preprocessed.onnx \
VOCAB_PATH=/absolute/path/to/hex_vocab.txt \
LABELS_NUM=2 \
bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_build_receiver.sh
```

`VOCAB_PATH` should point to Nepco's `hex_vocab.txt`. The receiver zeroes IPv4 source/destination addresses and TCP/UDP source/destination ports in the copied model input, then encodes packets as 4-hex-character tokens. The original 5-tuple is still preserved for flow lookup and hardware filter-rule installation.

## Run

Run the sender:

```bash
sudo bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_sender.sh
```

Run the receiver:

```bash
sudo bash scripts/4_latency_measure_SmartNIC/2_SmartNIC_deployment/run_receiver.sh
```

Useful environment variables:

- `ONNX_ROOT`: ONNX Runtime root on BF3, default `/home/ubuntu/pre2/onnx`.
- `MODEL_PATH`: compiled into the receiver by `run_build_receiver.sh`, default `outputs/onnx/output.preprocessed.onnx`.
- `VOCAB_PATH`: compiled into the receiver by `run_build_receiver.sh`, default `vocab/hex_vocab.txt`.
- `LABELS_NUM`: class count compiled into the receiver, default `2`.
- `MAX_PACKETS_PER_FLOW`: packets collected before inference, default `5`.
- `SENDER_PCI` and `RECEIVER_PCI`: BF3 DPDK PCI addresses.
- `OUTPUT_DIR`: receiver output directory for `log.txt` and `flow_tuples.txt`.

The sender prints per-second Mpps/Gbps. The receiver prints and logs per-second queue wait, input construction, model inference, and total analysis latency.
