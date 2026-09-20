/*
 * BF3 Inference Tool - 2-Class Classification (Binary Model)
 * Features:
 * - Flow Aggregation (5 packets)
 * - Flow Ageing (30s Timeout, Linear Scan)
 * - Thread-Safe Global Hash Table (Anti-RSS Spraying)
 * - Metrics: Per-Second Latency Breakdown (Queue Wait, Input Construction, Model Infer, Analysis Latency)
 * - Architecture: 1 Main -> 1 RX -> 1 HW Offload Core -> N-3 AI Cores
 * - Hardware Offload: Dedicated DPDK lcore for Asynchronous Drop rule offloading
 */

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <getopt.h>
#include <csignal>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>

#include <rte_common.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>
#include <rte_hash.h>
#include <rte_hash_crc.h>
#include <rte_ring.h>
#include <rte_spinlock.h>
#include <rte_flow.h>

#include <onnxruntime_cxx_api.h>
#include <arpa/inet.h>

#ifndef INFERENCE_BATCH_SIZE
#define INFERENCE_BATCH_SIZE 1
#endif
#define RX_RING_SIZE 8192
#define TX_RING_SIZE 8192
#define NB_MBUF_PER_QUEUE 131071
#define MEMPOOL_CACHE_SIZE 512
#define BURST_SIZE 128
#ifndef SEQ_LENGTH
#define SEQ_LENGTH 128
#endif

#define MAX_FLOWS_TOTAL 2000000
#define AI_RING_SIZE 262144
#define HW_OFFLOAD_RING_SIZE 262144
#define MAX_PIPELINES 16
#define MAX_RAW_BYTES 1024
#ifndef MAX_PACKETS_PER_FLOW
#define MAX_PACKETS_PER_FLOW 5
#endif
#ifndef TARGET_BYTE_LEN
#define TARGET_BYTE_LEN 128
#endif
#define FLOW_TIMEOUT_SEC 10

#ifndef NUM_CLASSES
#define NUM_CLASSES 2
#endif

#ifndef MODEL_PATH
#define MODEL_PATH "/home/ubuntu/pre2/models/output.preprocessed.onnx"
#endif

#ifndef VOCAB_PATH
#define VOCAB_PATH "/home/ubuntu/pre2/vocab/hex_vocab.txt"
#endif

static volatile bool force_quit = false;
static uint16_t port_id = 0;

static std::unique_ptr<Ort::Env> g_ort_env;

/* --- Statistics Structs --- */
struct rx_stats {
  uint64_t rx_pkts;
  uint64_t rx_bytes;
  uint64_t drops;
} __rte_cache_aligned;

struct ai_stats {
  bool active;
  int pipeline_id;
  uint64_t inf_count;
} __rte_cache_aligned;

struct rx_stats g_rx_stats[MAX_PIPELINES];
struct ai_stats g_ai_stats[RTE_MAX_LCORE];

typedef std::map<int, uint64_t> LabelMap;
static LabelMap g_subnet_labels[MAX_PIPELINES][NUM_CLASSES];

/* --- Core latency counters in TSC units --- */
static uint64_t g_lat_queue_wait_sum[MAX_PIPELINES];
static uint64_t g_lat_input_prep_sum[MAX_PIPELINES];
static uint64_t g_lat_model_infer_sum[MAX_PIPELINES];
static uint64_t g_lat_analysis_total_sum[MAX_PIPELINES];
static uint64_t g_lat_cnt[MAX_PIPELINES];

static rte_spinlock_t g_locks[MAX_PIPELINES];

static FILE *g_tuple_file = NULL;
static rte_spinlock_t g_file_lock;
static rte_spinlock_t g_flow_api_lock = RTE_SPINLOCK_INITIALIZER;

/* --- Global flow state and asynchronous offload queue --- */
static struct rte_hash *g_hash_table = NULL;
struct flow_entry *g_flow_pool = NULL;
static struct rte_ring *g_offload_ring = NULL;

struct VerifyConfig {
  uint32_t target_base = 0;
  uint32_t mask = 0;
};
VerifyConfig g_vconf;

/* --- Structs --- */
struct flow_key {
  uint32_t src_ip;
  uint32_t dst_ip;
  uint16_t src_port;
  uint16_t dst_port;
  uint8_t proto;
  uint8_t pad[3];
};

enum FlowStatus { COLLECTING = 0, PENDING, CLASSIFIED };

struct flow_entry {
  struct flow_key key;
  rte_spinlock_t flow_lock;
  uint8_t token_buf[MAX_RAW_BYTES];
  int16_t total_len;
  uint8_t packet_count;
  uint16_t pkt_lens[MAX_PACKETS_PER_FLOW];
  volatile int8_t status;
  int8_t label;
  uint64_t last_seen;
  uint64_t start_tsc;
  uint64_t rx_parsed_tsc;
  uint64_t ai_dequeued_tsc;
  uint32_t src_ip;
  struct rte_flow *hw_flow;
} __rte_cache_aligned;

struct worker_conf {
  int pipeline_id;
  uint16_t queue_id;
  struct rte_ring *ring;
};

static inline void zero_model_input_address_fields(uint8_t *buf, int len) {
  if (len < static_cast<int>(sizeof(struct rte_ipv4_hdr))) return;

  auto *ip = reinterpret_cast<struct rte_ipv4_hdr *>(buf);
  uint8_t version = ip->version_ihl >> 4;
  uint8_t ihl = (ip->version_ihl & 0x0f) * 4;
  if (version != 4 || ihl < sizeof(struct rte_ipv4_hdr) || len < ihl) return;

  uint8_t proto = ip->next_proto_id;
  ip->src_addr = 0;
  ip->dst_addr = 0;
  ip->hdr_checksum = 0;

  if (proto == IPPROTO_TCP && len >= ihl + static_cast<int>(sizeof(struct rte_tcp_hdr))) {
    auto *tcp = reinterpret_cast<struct rte_tcp_hdr *>(buf + ihl);
    tcp->src_port = 0;
    tcp->dst_port = 0;
    tcp->cksum = 0;
  } else if (proto == IPPROTO_UDP && len >= ihl + static_cast<int>(sizeof(struct rte_udp_hdr))) {
    auto *udp = reinterpret_cast<struct rte_udp_hdr *>(buf + ihl);
    udp->src_port = 0;
    udp->dst_port = 0;
    udp->dgram_cksum = 0;
  }
}

/* --- FastTokenizer --- */
struct FastTokenizer {
  int64_t* lut_4hex = nullptr;
  int64_t id_cls = 0, id_sep = 0, id_pad = 0, id_unk = 0;

  void load(const std::string& path) {
    std::unordered_map<std::string, int64_t> temp_vocab;
    std::ifstream file(path);
    if (!file.is_open()) rte_exit(EXIT_FAILURE, "Cannot open vocab: %s\n", path.c_str());
    std::string line; int64_t idx = 0;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\n') line.pop_back();
      if (!line.empty() && line.back() == '\r') line.pop_back();
      temp_vocab[line] = idx++;
    }
    if (temp_vocab.count("[CLS]")) id_cls = temp_vocab["[CLS]"];
    if (temp_vocab.count("[SEP]")) id_sep = temp_vocab["[SEP]"];
    if (temp_vocab.count("[PAD]")) id_pad = temp_vocab["[PAD]"];
    if (temp_vocab.count("[UNK]")) id_unk = temp_vocab["[UNK]"];

    lut_4hex = (int64_t*)malloc(65536 * sizeof(int64_t));
    if (!lut_4hex) rte_exit(EXIT_FAILURE, "Cannot allocate 4-hex tokenizer LUT\n");
    char hex_buf[5];
    for (int i = 0; i < 65536; i++) {
      snprintf(hex_buf, sizeof(hex_buf), "%04x", i);
      std::string s(hex_buf);
      lut_4hex[i] = temp_vocab.count(s) ? temp_vocab[s] : id_unk;
    }
  }

  int encode_flow_direct(const struct flow_entry* f, int64_t* dest_buf, int max_len) {
    if (max_len < 2) return 0;
    int pos = 0;
    int byte_offset = 0;
    dest_buf[pos++] = id_cls;
    for (int i = 0; i < f->packet_count && pos < max_len; i++) {
      int seg_len = f->pkt_lens[i];
      const uint8_t* pkt_ptr = f->token_buf + byte_offset;
      for (int j = 0; j + 1 < seg_len; j += 2) {
        if (pos >= max_len) break;
        uint16_t token = ((uint16_t)pkt_ptr[j] << 8) | pkt_ptr[j + 1];
        dest_buf[pos++] = lut_4hex[token];
      }
      byte_offset += seg_len;
    }
    if (pos < max_len) dest_buf[pos++] = id_sep;
    while (pos < max_len) dest_buf[pos++] = id_pad;
    return pos;
  }
};
static FastTokenizer g_tokenizer;

static void signal_handler(int signum) { if (signum == SIGINT || signum == SIGTERM) force_quit = true; }

void load_verify_config() {
  g_vconf.target_base = inet_addr("10.0.0.0");
  g_vconf.mask = inet_addr("255.0.0.0");
}

static inline struct rte_flow* install_drop_rule(uint16_t port, const struct flow_key *key) {
  struct rte_flow_error error;
  struct rte_flow_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.ingress = 1;
  attr.priority = 0;

  struct rte_flow_item pattern[4];
  struct rte_flow_action action[2];
  int pattern_idx = 0;

  memset(pattern, 0, sizeof(pattern));
  memset(action, 0, sizeof(action));

  pattern[pattern_idx].type = RTE_FLOW_ITEM_TYPE_ETH;
  pattern[pattern_idx].spec = NULL;
  pattern[pattern_idx].mask = NULL;
  pattern_idx++;

  struct rte_flow_item_ipv4 ip_spec;
  struct rte_flow_item_ipv4 ip_mask;
  memset(&ip_spec, 0, sizeof(ip_spec));
  memset(&ip_mask, 0, sizeof(ip_mask));
  ip_spec.hdr.src_addr = key->src_ip;
  ip_spec.hdr.dst_addr = key->dst_ip;
  ip_spec.hdr.next_proto_id = key->proto;
  ip_mask.hdr.src_addr = 0xFFFFFFFF;
  ip_mask.hdr.dst_addr = 0xFFFFFFFF;
  ip_mask.hdr.next_proto_id = 0xFF;

  pattern[pattern_idx].type = RTE_FLOW_ITEM_TYPE_IPV4;
  pattern[pattern_idx].spec = &ip_spec;
  pattern[pattern_idx].mask = &ip_mask;
  pattern_idx++;

  struct rte_flow_item_tcp tcp_spec, tcp_mask;
  struct rte_flow_item_udp udp_spec, udp_mask;
  if (key->proto == IPPROTO_TCP) {
      memset(&tcp_spec, 0, sizeof(tcp_spec));
      memset(&tcp_mask, 0, sizeof(tcp_mask));
      tcp_spec.hdr.src_port = key->src_port;
      tcp_spec.hdr.dst_port = key->dst_port;
      tcp_mask.hdr.src_port = 0xFFFF;
      tcp_mask.hdr.dst_port = 0xFFFF;
      pattern[pattern_idx].type = RTE_FLOW_ITEM_TYPE_TCP;
      pattern[pattern_idx].spec = &tcp_spec;
      pattern[pattern_idx].mask = &tcp_mask;
      pattern_idx++;
  } else if (key->proto == IPPROTO_UDP) {
      memset(&udp_spec, 0, sizeof(udp_spec));
      memset(&udp_mask, 0, sizeof(udp_mask));
      udp_spec.hdr.src_port = key->src_port;
      udp_spec.hdr.dst_port = key->dst_port;
      udp_mask.hdr.src_port = 0xFFFF;
      udp_mask.hdr.dst_port = 0xFFFF;
      pattern[pattern_idx].type = RTE_FLOW_ITEM_TYPE_UDP;
      pattern[pattern_idx].spec = &udp_spec;
      pattern[pattern_idx].mask = &udp_mask;
      pattern_idx++;
  }

  pattern[pattern_idx].type = RTE_FLOW_ITEM_TYPE_END;

  action[0].type = RTE_FLOW_ACTION_TYPE_DROP;
  action[1].type = RTE_FLOW_ACTION_TYPE_END;

  rte_spinlock_lock(&g_flow_api_lock);
  struct rte_flow *flow = rte_flow_create(port, &attr, pattern, action, &error);
  rte_spinlock_unlock(&g_flow_api_lock);

  return flow;
}

/* --- Asynchronous hardware offload worker --- */
int lcore_hw_offload_worker(void *arg) {
      (void)arg;
      void *ptrs[128];
      while (!force_quit) {
          unsigned int n = rte_ring_sc_dequeue_burst(g_offload_ring, ptrs, 128, NULL);
          if (unlikely(n == 0)) {
              rte_pause();
              continue;
          }
          for (unsigned int i = 0; i < n; i++) {
              struct flow_entry *f = (struct flow_entry *)ptrs[i];

              if (f->status == CLASSIFIED && f->key.src_ip == 0) {
                  struct rte_flow_error error;
                  rte_spinlock_lock(&g_flow_api_lock);
                  rte_flow_destroy(port_id, f->hw_flow, &error);
                  rte_spinlock_unlock(&g_flow_api_lock);
                  rte_free(f);
                  continue;
              }

              rte_spinlock_lock(&f->flow_lock);
              if (f->status == COLLECTING || f->hw_flow != NULL) {
                  rte_spinlock_unlock(&f->flow_lock);
                  continue;
              }
              struct flow_key key = f->key;
              uint64_t current_start = f->start_tsc;
              rte_spinlock_unlock(&f->flow_lock);

              struct rte_flow *hw_rule = install_drop_rule(port_id, &key);

              if (hw_rule) {
                  rte_spinlock_lock(&f->flow_lock);
                  if (f->start_tsc == current_start && f->hw_flow == NULL) {
                      f->hw_flow = hw_rule;
                  } else {
                      rte_spinlock_unlock(&f->flow_lock);
                      struct rte_flow_error error;
                      rte_spinlock_lock(&g_flow_api_lock);
                      rte_flow_destroy(port_id, hw_rule, &error);
                      rte_spinlock_unlock(&g_flow_api_lock);
                      continue;
                  }
                  rte_spinlock_unlock(&f->flow_lock);
              }
          }
      }
      return 0;
}

/* --- Asynchronous flow aging --- */
void age_global_flows() {
  uint64_t now = rte_get_tsc_cycles();
  uint64_t timeout_cycles = FLOW_TIMEOUT_SEC * rte_get_timer_hz();

  for (int i = 0; i < MAX_FLOWS_TOTAL; i++) {
    struct flow_entry *f = &g_flow_pool[i];
    if (f->last_seen != 0) {
        if (unlikely(now > f->last_seen && (now - f->last_seen) > timeout_cycles)) {
              rte_spinlock_lock(&f->flow_lock);
              uint64_t current_now = rte_get_tsc_cycles();
              if (current_now > f->last_seen && (current_now - f->last_seen) > timeout_cycles) {
                rte_hash_del_key(g_hash_table, &f->key);
                f->last_seen = 0;
                f->start_tsc = 0;
                f->status = COLLECTING;

                struct rte_flow *hw = f->hw_flow;
                f->hw_flow = NULL;
                rte_spinlock_unlock(&f->flow_lock);

                if (hw != NULL) {
                    struct flow_entry *dummy = (struct flow_entry *)rte_zmalloc(NULL, sizeof(struct flow_entry), 64);
                    if (dummy) {
                        dummy->status = CLASSIFIED;
                        dummy->key.src_ip = 0;
                        dummy->hw_flow = hw;
                        if (unlikely(rte_ring_mp_enqueue(g_offload_ring, dummy) != 0)) {
                            struct rte_flow_error error;
                            rte_spinlock_lock(&g_flow_api_lock);
                            rte_flow_destroy(port_id, hw, &error);
                            rte_spinlock_unlock(&g_flow_api_lock);
                            rte_free(dummy);
                        }
                    } else {
                        struct rte_flow_error error;
                        rte_spinlock_lock(&g_flow_api_lock);
                        rte_flow_destroy(port_id, hw, &error);
                        rte_spinlock_unlock(&g_flow_api_lock);
                    }
                }
              } else {
                rte_spinlock_unlock(&f->flow_lock);
              }
            }
     }
   }
}

/* --- AI Worker --- */
int lcore_ai_worker(void *arg) {
  struct worker_conf *conf = (struct worker_conf *)arg;
  struct rte_ring *ring = conf->ring;
  int pid = conf->pipeline_id;
  unsigned int lcore_id = rte_lcore_id();

  g_ai_stats[lcore_id].active = true;
  g_ai_stats[lcore_id].pipeline_id = pid;

  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  std::unique_ptr<Ort::Session> session;
  try {
    session.reset(new Ort::Session(*g_ort_env, MODEL_PATH, session_options));
  }
  catch (const Ort::Exception& e) { rte_exit(EXIT_FAILURE, "ONNX Init Failed: %s\n", e.what()); }

  int64_t* batch_src = (int64_t*)rte_malloc(NULL, INFERENCE_BATCH_SIZE * SEQ_LENGTH * sizeof(int64_t), 64);
  int64_t* batch_seg = (int64_t*)rte_malloc(NULL, INFERENCE_BATCH_SIZE * SEQ_LENGTH * sizeof(int64_t), 64);
  void* batch_ptrs[INFERENCE_BATCH_SIZE];

  const char* input_names[] = {"src", "seg"};
  const char* output_names[] = {"logits"};

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::RunOptions run_options{nullptr};

  memset(batch_src, 0, INFERENCE_BATCH_SIZE * SEQ_LENGTH * sizeof(int64_t));
  memset(batch_seg, 0, INFERENCE_BATCH_SIZE * SEQ_LENGTH * sizeof(int64_t));

  {
    int64_t warmup_dims[] = {(int64_t)INFERENCE_BATCH_SIZE, SEQ_LENGTH};
    std::vector<Ort::Value> warmup_inputs;
    warmup_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, batch_src, INFERENCE_BATCH_SIZE * SEQ_LENGTH, warmup_dims, 2));
    warmup_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, batch_seg, INFERENCE_BATCH_SIZE * SEQ_LENGTH, warmup_dims, 2));
    try {
      (void)session->Run(run_options, input_names, warmup_inputs.data(), 2, output_names, 1);
    } catch (...) {}
  }

  unsigned int local_count = 0;
  uint64_t last_flush_tsc = rte_get_tsc_cycles();
  uint64_t flush_timeout = rte_get_timer_hz() / 1000;

  while (!force_quit) {
    unsigned int req = INFERENCE_BATCH_SIZE - local_count;
    unsigned int n = rte_ring_mc_dequeue_burst(ring, &batch_ptrs[local_count], req, NULL);
    uint64_t deq_tsc = rte_get_tsc_cycles();
    for (unsigned int i = local_count; i < local_count + n; i++) {
        ((struct flow_entry *)batch_ptrs[i])->ai_dequeued_tsc = deq_tsc;
    }
    local_count += n;

    uint64_t current_tsc = rte_get_tsc_cycles();

    if (local_count == 0 || (local_count < INFERENCE_BATCH_SIZE && (current_tsc - last_flush_tsc) < flush_timeout)) {
      rte_pause();
      continue;
    }

    unsigned int batch_n = local_count;

    // Metric point 1: input construction start.
    uint64_t tsc_prep_start = rte_get_tsc_cycles();
    for (unsigned int i = 0; i < batch_n; i++) {
      struct flow_entry *f = (struct flow_entry *)batch_ptrs[i];
      g_tokenizer.encode_flow_direct(f, &batch_src[i * SEQ_LENGTH], SEQ_LENGTH);
      int64_t* src = &batch_src[i * SEQ_LENGTH];
      int64_t* seg = &batch_seg[i * SEQ_LENGTH];
      for (int j = 0; j < SEQ_LENGTH; j++) seg[j] = (src[j] == g_tokenizer.id_pad) ? 0 : 1;
    }

    int64_t dims[] = {(int64_t)batch_n, SEQ_LENGTH};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, batch_src, batch_n*SEQ_LENGTH, dims, 2));
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, batch_seg, batch_n*SEQ_LENGTH, dims, 2));

    try {
      // Metric point 2: model inference start.
      uint64_t tsc_infer_start = rte_get_tsc_cycles();
      auto outputs = session->Run(run_options, input_names, inputs.data(), 2, output_names, 1);
      // Metric point 3: model inference end.
      uint64_t tsc_infer_end = rte_get_tsc_cycles();

      float* logits = outputs[0].GetTensorMutableData<float>();
      size_t n_labels = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

      rte_spinlock_lock(&g_locks[pid]);
      for (unsigned int i = 0; i < batch_n; i++) {
        struct flow_entry *f = (struct flow_entry *)batch_ptrs[i];
        float* l = logits + (i * n_labels);
        int pred = 0; float max_v = l[0];
        for(size_t j=1; j<n_labels; j++) if(l[j]>max_v) { max_v=l[j]; pred=(int)j; }

        f->label = pred;
        f->status = CLASSIFIED;

        uint32_t ip_h = rte_be_to_cpu_32(f->src_ip);
        int gt_class = (ip_h >> 16) & 0xFF;
        if (gt_class < 0 || gt_class >= NUM_CLASSES) gt_class = 0;

        g_subnet_labels[pid][gt_class][pred]++;

        if (f->start_tsc > 0 && tsc_infer_end > f->start_tsc) {
          uint64_t q_wait       = (f->ai_dequeued_tsc > f->rx_parsed_tsc) ? (f->ai_dequeued_tsc - f->rx_parsed_tsc) : 0;
          uint64_t prep         = (tsc_infer_start > tsc_prep_start) ? (tsc_infer_start - tsc_prep_start) : 0;
          uint64_t infer        = (tsc_infer_end > tsc_infer_start) ? (tsc_infer_end - tsc_infer_start) : 0;
          uint64_t analysis_lat = (tsc_infer_end > f->start_tsc) ? (tsc_infer_end - f->start_tsc) : 0;

          g_lat_queue_wait_sum[pid]     += q_wait;
          g_lat_input_prep_sum[pid]     += prep;
          g_lat_model_infer_sum[pid]    += infer;
          g_lat_analysis_total_sum[pid] += analysis_lat;
          g_lat_cnt[pid]++;
        }
      }
      rte_spinlock_unlock(&g_locks[pid]);

      if (g_tuple_file) {
        rte_spinlock_lock(&g_file_lock);
        for (unsigned int i = 0; i < batch_n; i++) {
          struct flow_entry *f = (struct flow_entry *)batch_ptrs[i];
          uint32_t sip = rte_be_to_cpu_32(f->key.src_ip);
          uint32_t dip = rte_be_to_cpu_32(f->key.dst_ip);

          fprintf(g_tuple_file, "%u.%u.%u.%u %u.%u.%u.%u %u %u %u %lu %lu %lu %lu %lu\n",
               (sip >> 24) & 0xFF, (sip >> 16) & 0xFF, (sip >> 8) & 0xFF, sip & 0xFF,
               (dip >> 24) & 0xFF, (dip >> 16) & 0xFF, (dip >> 8) & 0xFF, dip & 0xFF,
               rte_be_to_cpu_16(f->key.src_port),
               rte_be_to_cpu_16(f->key.dst_port),
               f->key.proto,
               f->start_tsc,
               f->rx_parsed_tsc,
               f->ai_dequeued_tsc,
               tsc_prep_start,
               tsc_infer_end);
        }
        rte_spinlock_unlock(&g_file_lock);
      }

      g_ai_stats[lcore_id].inf_count += batch_n;

    } catch (const Ort::Exception& e) {
        fprintf(stderr, "\n[Fatal] AI Worker %d ONNX Runtime error: %s\n", pid, e.what());
    } catch (const std::exception& e) {
        fprintf(stderr, "\n[Fatal] AI Worker %d C++ exception: %s\n", pid, e.what());
    } catch (...) {
        fprintf(stderr, "\n[Fatal] AI Worker %d unknown error.\n", pid);
    }

    local_count = 0;
    last_flush_tsc = rte_get_tsc_cycles();
  }
  rte_free(batch_src); rte_free(batch_seg);
  return 0;
}

/* --- RX Worker --- */
int lcore_rx_worker(void *arg) {
  struct worker_conf *conf = (struct worker_conf *)arg;
  struct rte_mbuf *pkts[BURST_SIZE];

  while (!force_quit) {
    uint16_t nb_rx = rte_eth_rx_burst(port_id, conf->queue_id, pkts, BURST_SIZE);
    if (unlikely(nb_rx == 0)) continue;

    uint64_t now = rte_get_tsc_cycles();
    uint64_t bytes = 0;

    for (int i = 0; i < nb_rx; i++) {
      struct rte_mbuf *m = pkts[i];
      bytes += rte_pktmbuf_pkt_len(m);
      struct rte_ether_hdr *eth = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);

      if (eth->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
        struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(eth + 1);

        struct flow_key key;
        memset(&key, 0, sizeof(key));
        key.src_ip = ip->src_addr;
        key.dst_ip = ip->dst_addr;
        key.proto = ip->next_proto_id;

        if (ip->next_proto_id == IPPROTO_TCP) {
          struct rte_tcp_hdr *tcp = (struct rte_tcp_hdr *)((uint8_t*)ip + ((ip->version_ihl & 0x0f) * 4));
          key.src_port = tcp->src_port; key.dst_port = tcp->dst_port;
        } else if (ip->next_proto_id == IPPROTO_UDP) {
          struct rte_udp_hdr *udp = (struct rte_udp_hdr *)((uint8_t*)ip + ((ip->version_ihl & 0x0f) * 4));
          key.src_port = udp->src_port; key.dst_port = udp->dst_port;
        }

        int ret = rte_hash_lookup(g_hash_table, &key);
        int slot = (ret >= 0) ? ret : rte_hash_add_key(g_hash_table, &key);

        if (likely(slot >= 0)) {
          struct flow_entry *f = &g_flow_pool[slot];

          rte_spinlock_lock(&f->flow_lock);

          if (unlikely(f->start_tsc == 0)) {
            f->packet_count = 0;
            f->total_len = 0;
            f->status = COLLECTING;
            f->src_ip = key.src_ip;
            f->key = key;
            f->start_tsc = now;
            f->hw_flow = NULL;
          }

          f->last_seen = now;
          if (f->status == COLLECTING) {
            if (f->packet_count < MAX_PACKETS_PER_FLOW) {
              int available = rte_pktmbuf_pkt_len(m) - sizeof(struct rte_ether_hdr);
              int copy_len = std::min(available, TARGET_BYTE_LEN);

              if (copy_len > 0) {
                uint8_t model_input[TARGET_BYTE_LEN];
                rte_memcpy(model_input, (uint8_t*)ip, copy_len);
                zero_model_input_address_fields(model_input, copy_len);
                rte_memcpy(&f->token_buf[f->total_len], model_input, copy_len);
                f->pkt_lens[f->packet_count++] = copy_len;
                f->total_len += copy_len;
              }

              if (f->packet_count >= MAX_PACKETS_PER_FLOW) {
                f->status = PENDING;
                f->rx_parsed_tsc = rte_get_tsc_cycles();

                if (rte_ring_sp_enqueue(conf->ring, f) != 0) g_rx_stats[conf->pipeline_id].drops++;
                rte_ring_mp_enqueue(g_offload_ring, f);
              }
            }
          }
          f->last_seen = rte_get_tsc_cycles();
          rte_spinlock_unlock(&f->flow_lock);
        }
      }
      rte_pktmbuf_free(m);
    }

    g_rx_stats[conf->pipeline_id].rx_pkts += nb_rx;
    g_rx_stats[conf->pipeline_id].rx_bytes += bytes;
  }
  return 0;
}

/* --- Main --- */
int main(int argc, char *argv[]) {
  int ret = rte_eal_init(argc, argv);
  if (ret < 0) rte_exit(EXIT_FAILURE, "EAL Init\n");
  argc -= ret; argv += ret;
  int opt; while ((opt = getopt(argc, argv, "p:")) != -1) if (opt == 'p') port_id = atoi(optarg);

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  rte_spinlock_init(&g_file_lock);
  g_tuple_file = fopen("flow_tuples.txt", "w");
  if (!g_tuple_file) {
    printf("Warning: Cannot open flow_tuples.txt for writing.\n");
  }

  std::ofstream log_file("log.txt", std::ios::out | std::ios::app);
  if (!log_file.is_open()) {
    printf("Warning: Cannot open log.txt for writing.\n");
  }

  load_verify_config();
  printf("Using ONNX model: %s\n", MODEL_PATH);
  printf("Using Nepco vocab: %s\n", VOCAB_PATH);
  printf("MAX_PACKETS_PER_FLOW: %d\n", MAX_PACKETS_PER_FLOW);
  g_tokenizer.load(VOCAB_PATH);

  g_ort_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "BF3_AI"));

  int num_rx = 1;
  int num_hw_offload = 1;
  int num_pipelines = num_rx;

  int num_workers = (int)rte_lcore_count() - 1;
  int num_ai = num_workers - num_rx - num_hw_offload;

  if (num_workers < num_rx + num_hw_offload || num_ai < 0) {
    rte_exit(EXIT_FAILURE, "Need at least 4 lcores (1 Main, 1 RX, 1 HW Offload, rest AI)\n");
  }

  struct rte_hash_parameters h_params = {
    .name = "GLOBAL_FLOW_HASH",
    .entries = MAX_FLOWS_TOTAL,
    .key_len = sizeof(struct flow_key),
    .hash_func = rte_hash_crc,
    .socket_id = (int)rte_socket_id(),
    .extra_flag = RTE_HASH_EXTRA_FLAGS_RW_CONCURRENCY_LF | RTE_HASH_EXTRA_FLAGS_MULTI_WRITER_ADD
  };
  g_hash_table = rte_hash_create(&h_params);
  if (!g_hash_table) rte_exit(EXIT_FAILURE, "Global Hash Create Failed\n");

  g_flow_pool = (struct flow_entry *)rte_zmalloc("GLOBAL_FP", MAX_FLOWS_TOTAL * sizeof(struct flow_entry), 64);
  for(int i = 0; i < MAX_FLOWS_TOTAL; i++) {
    rte_spinlock_init(&g_flow_pool[i].flow_lock);
  }

  g_offload_ring = rte_ring_create("HW_OFFLOAD_RING", HW_OFFLOAD_RING_SIZE, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
  if (!g_offload_ring) rte_exit(EXIT_FAILURE, "Offload Ring Create Failed\n");

  struct rte_eth_conf port_conf = {};
  port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;
  port_conf.rxmode.offloads = RTE_ETH_RX_OFFLOAD_RSS_HASH;
  port_conf.rx_adv_conf.rss_conf.rss_hf = RTE_ETH_RSS_IP | RTE_ETH_RSS_TCP | RTE_ETH_RSS_UDP;

  rte_eth_dev_configure(port_id, num_pipelines, num_pipelines, &port_conf);
  struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NB_MBUF_PER_QUEUE * num_pipelines, MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

  for (int q = 0; q < num_pipelines; q++) {
    rte_eth_rx_queue_setup(port_id, q, RX_RING_SIZE, rte_socket_id(), NULL, mbuf_pool);
    rte_eth_tx_queue_setup(port_id, q, TX_RING_SIZE, rte_socket_id(), NULL);
    rte_spinlock_init(&g_locks[q]);
  }

  struct worker_conf *configs = (struct worker_conf *)rte_zmalloc("Configs", RTE_MAX_LCORE * sizeof(struct worker_conf), 64);

  rte_eth_dev_start(port_id);
  rte_eth_promiscuous_enable(port_id);

  unsigned int lcore_id;
  int rx_launched = 0;
  int hw_offload_launched = 0;
  int ai_launched = 0;
  unsigned int main_lcore = rte_get_main_lcore();

  for (int i = 0; i < num_pipelines; i++) {
    char rng_n[32]; sprintf(rng_n, "R_%d", i);
    rte_ring_create(rng_n, AI_RING_SIZE, rte_socket_id(), RING_F_SP_ENQ);
  }

  RTE_LCORE_FOREACH_WORKER(lcore_id) {
    if (lcore_id == main_lcore) continue;

    if (rx_launched < num_rx) {
      int pipeline_id = rx_launched;
      char rng_n[32]; sprintf(rng_n, "R_%d", pipeline_id);
      configs[lcore_id].pipeline_id = pipeline_id;
      configs[lcore_id].queue_id = pipeline_id;
      configs[lcore_id].ring = rte_ring_lookup(rng_n);
      rte_eal_remote_launch(lcore_rx_worker, &configs[lcore_id], lcore_id);
      rx_launched++;
    } else if (hw_offload_launched < num_hw_offload) {
      rte_eal_remote_launch(lcore_hw_offload_worker, NULL, lcore_id);
      hw_offload_launched++;
    } else if (ai_launched < num_ai) {
      int pipeline_id = ai_launched % num_rx;
      char rng_n[32]; sprintf(rng_n, "R_%d", pipeline_id);
      configs[lcore_id].pipeline_id = pipeline_id;
      configs[lcore_id].ring = rte_ring_lookup(rng_n);
      rte_eal_remote_launch(lcore_ai_worker, &configs[lcore_id], lcore_id);
      ai_launched++;
    }
  }

/* --- Monitor and metrics loop --- */
  uint64_t last_age_check = rte_get_tsc_cycles();
  uint64_t hz = rte_get_timer_hz();

  while (!force_quit) {
    uint64_t now = rte_get_tsc_cycles();

    if (now - last_age_check > hz) {
      age_global_flows();
      last_age_check = now;
    }

    sleep(1);

    auto t_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ts_ss;
    ts_ss << std::put_time(std::localtime(&t_now), "%Y-%m-%d %H:%M:%S");
    std::string time_str = ts_ss.str();

    uint64_t q_wait_sum = 0, prep_sum = 0, infer_sum = 0, analysis_total_sum = 0, cnt = 0;

    for(int i=0; i<num_pipelines; i++) {
      rte_spinlock_lock(&g_locks[i]);
      q_wait_sum         += g_lat_queue_wait_sum[i];
      prep_sum           += g_lat_input_prep_sum[i];
      infer_sum          += g_lat_model_infer_sum[i];
      analysis_total_sum += g_lat_analysis_total_sum[i];
      cnt                += g_lat_cnt[i];

      g_lat_queue_wait_sum[i]     = 0;
      g_lat_input_prep_sum[i]     = 0;
      g_lat_model_infer_sum[i]    = 0;
      g_lat_analysis_total_sum[i] = 0;
      g_lat_cnt[i]                = 0;
      rte_spinlock_unlock(&g_locks[i]);
    }

    double avg_q_wait = 0.0, avg_prep = 0.0, avg_infer = 0.0, avg_analysis_total = 0.0;
    double pct_q_wait = 0.0, pct_prep = 0.0, pct_infer = 0.0;

    if (cnt > 0) {
      double us_factor = 1000000.0 / hz;
      avg_q_wait         = ((double)q_wait_sum / cnt) * us_factor;
      avg_prep           = ((double)prep_sum / cnt) * us_factor;
      avg_infer          = ((double)infer_sum / cnt) * us_factor;
      avg_analysis_total = ((double)analysis_total_sum / cnt) * us_factor;

      if (avg_analysis_total > 0.0001) {
          pct_q_wait = (avg_q_wait / avg_analysis_total) * 100.0;
          pct_prep   = (avg_prep / avg_analysis_total) * 100.0;
          pct_infer  = (avg_infer / avg_analysis_total) * 100.0;
      }
    }

    std::stringstream out_ss;
    out_ss << "================================================================================\n";
    out_ss << " [" << time_str << "] BF3 Real-time Latency Breakdown (Per-Second Window)\n";
    out_ss << "================================================================================\n";
    out_ss << " Metric Name             | Avg Latency (us) | % of Total | Description\n";
    out_ss << "-------------------------+------------------+------------+----------------------\n";

    char buf[512];
    snprintf(buf, sizeof(buf),
             " 1. Queue Wait           | %16.2f | %9.1f%% | RX Ring -> AI Dequeue\n"
             " 2. Input Construction   | %16.2f | %9.1f%% | Tokenization & Tensor Prep\n"
             " 3. Model Infer          | %16.2f | %9.1f%% | ONNX Session Run\n"
             "-------------------------+------------------+------------+----------------------\n"
             " * 4. Analysis Latency  | %16.2f |     100.0%% | Start -> AI Final Output\n",
             avg_q_wait, pct_q_wait,
             avg_prep, pct_prep,
             avg_infer, pct_infer,
             avg_analysis_total);
    out_ss << buf;
    out_ss << "================================================================================\n";

    std::string report = out_ss.str();

    printf("\033[2J\033[H%s", report.c_str());
    fflush(stdout);

    if (log_file.is_open()) {
        log_file << report << "\n";
        log_file.flush();
    }
  }

  rte_eal_mp_wait_lcore();

  printf("DPDK Timer HZ: %lu\n", rte_get_timer_hz());
  struct rte_flow_error flow_error;
  rte_flow_flush(port_id, &flow_error);

  if (g_tuple_file) {
    fclose(g_tuple_file);
  }

  if (log_file.is_open()) {
      log_file.close();
  }
  return 0;
}
