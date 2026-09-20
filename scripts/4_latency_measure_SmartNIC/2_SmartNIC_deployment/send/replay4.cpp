#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <csignal>
#include <unistd.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <cmath>
#include <random>

extern "C" {
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_atomic.h>
#include <rte_ring.h>
#include <rte_mempool.h>
}

/* --- Configuration --- */
struct ReplayConfig {
    uint32_t base_ip = 0;
};

/* --- Dynamic flow state --- */
struct FlowState {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t pkt_size;

    uint32_t pkts_total;
    uint32_t pkts_left;
    uint32_t pkts_sent;

    uint64_t start_tsc;
    uint64_t next_send_tsc;

    uint64_t base_gap_tsc;
    uint64_t current_gap_tsc;
    double burst_amp;
    double jitter_amp;
    double pulse_period_s;
    double phase1;
    double phase2;

    uint16_t pkts_since_update;
};

struct WorkerCtx {
    uint16_t qid;
    uint16_t worker_idx;
    uint16_t worker_cnt;
    struct rte_ring *rx_ring;
};

#ifndef PORT_ID
#define PORT_ID 1
#endif
#define BURST_SIZE 64
#define MAX_TX_WORKERS 7
#define FLOW_POOL_SIZE 262143
#define RING_SIZE 65536

ReplayConfig g_conf;
struct rte_mempool *mbuf_pool = nullptr;
struct rte_mempool *flow_pool = nullptr;

static rte_atomic64_t tx_pkts_total;
static rte_atomic64_t tx_flows_total;
static volatile bool force_quit = false;
static uint16_t g_active_workers = 0;

static const uint8_t bf3_mac[6] = {0xd8, 0x94, 0x24, 0x57, 0x9e, 0x4b};
static struct rte_ether_addr g_sender_mac;

static void signal_handler(int) { force_quit = true; }

void load_config() {
    g_conf.base_ip = ntohl(inet_addr("10.0.0.0"));
}

/* --- Update the dynamic packet gap inside each worker. --- */
static inline void update_dynamic_gap(FlowState &fs, uint64_t now, uint64_t tsc_hz, std::mt19937 &rng) {
    double age_s = (double)(now - fs.start_tsc) / (double)tsc_hz;
    if (age_s < 0) age_s = 0;

    double x = age_s / fs.pulse_period_s;

    double periodic = 1.0 +
        fs.burst_amp * sin(2 * M_PI * x + fs.phase1) +
        0.55 * fs.burst_amp * sin(2 * M_PI * 0.43 * x + fs.phase2) +
        0.25 * fs.burst_amp * sin(2 * M_PI * 1.8 * x + 0.5 * fs.phase1);

    std::normal_distribution<double> dist_norm(0.0, fs.jitter_amp);
    double jitter = 1.0 + dist_norm(rng);

    double gap_scale = periodic * jitter;
    if (gap_scale < 0.30) gap_scale = 0.30;
    if (gap_scale > 3.00) gap_scale = 3.00;

    fs.current_gap_tsc = (uint64_t)(fs.base_gap_tsc * gap_scale);
}

/* --- TX worker data plane --- */
int tx_worker(void *arg) {
    WorkerCtx *ctx = reinterpret_cast<WorkerCtx *>(arg);
    const uint16_t qid = ctx->qid;
    const uint64_t tsc_hz = rte_get_tsc_hz();

    std::mt19937 rng(rte_rdtsc() + ctx->worker_idx);

    const uint64_t const_gap_tsc = (tsc_hz * 10000ULL) / 1000000ULL;
    const uint32_t gap_after_pkts = 5;

    std::vector<FlowState> active_flows;
    active_flows.reserve(16384);

    struct rte_mbuf *burst[BURST_SIZE];
    void *new_flows[128];
    size_t flow_idx = 0;

    while (!force_quit) {
        uint64_t now = rte_get_tsc_cycles();

        unsigned int nb_recv = rte_ring_dequeue_burst(ctx->rx_ring, new_flows, 128, nullptr);
        if (nb_recv > 0) {
            for (unsigned int i = 0; i < nb_recv; i++) {
                FlowState *fs_ptr = static_cast<FlowState *>(new_flows[i]);
                active_flows.push_back(*fs_ptr);
                rte_atomic64_inc(&tx_flows_total);
            }
            rte_mempool_put_bulk(flow_pool, new_flows, nb_recv);
        }

        if (active_flows.empty()) {
            rte_pause();
            continue;
        }

        uint16_t nb_alloc = 0;
        uint32_t scan_budget = (uint32_t)active_flows.size();

        while (nb_alloc < BURST_SIZE && !active_flows.empty() && scan_budget--) {
            if (flow_idx >= active_flows.size()) flow_idx = 0;
            FlowState &fs = active_flows[flow_idx];

            if (now < fs.start_tsc || (fs.next_send_tsc != 0 && now < fs.next_send_tsc)) {
                flow_idx++;
                continue;
            }

            struct rte_mbuf *m = rte_pktmbuf_alloc(mbuf_pool);
            if (unlikely(m == nullptr)) break;

            char *ptr = static_cast<char *>(rte_pktmbuf_append(m, fs.pkt_size));
            if (unlikely(ptr == nullptr)) { rte_pktmbuf_free(m); break; }

            auto *eth = reinterpret_cast<rte_ether_hdr *>(ptr);
            rte_memcpy(eth->dst_addr.addr_bytes, bf3_mac, RTE_ETHER_ADDR_LEN);
            rte_memcpy(eth->src_addr.addr_bytes, g_sender_mac.addr_bytes, RTE_ETHER_ADDR_LEN);
            eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

            auto *ip = reinterpret_cast<rte_ipv4_hdr *>(eth + 1);
            ip->version_ihl = 0x45; ip->type_of_service = 0;
            ip->total_length = rte_cpu_to_be_16(fs.pkt_size - sizeof(rte_ether_hdr));
            ip->time_to_live = 64; ip->next_proto_id = IPPROTO_UDP;
            ip->src_addr = fs.src_ip; ip->dst_addr = fs.dst_ip;
            ip->hdr_checksum = 0;
            ip->hdr_checksum = rte_ipv4_cksum(ip);
            auto *udp = reinterpret_cast<rte_udp_hdr *>(ip + 1);
            udp->src_port = fs.src_port; udp->dst_port = fs.dst_port;
            udp->dgram_len = rte_cpu_to_be_16(fs.pkt_size - sizeof(rte_ether_hdr) - sizeof(rte_ipv4_hdr));
            udp->dgram_cksum = 0;

            m->l2_len = sizeof(rte_ether_hdr); m->l3_len = sizeof(rte_ipv4_hdr); m->l4_len = sizeof(rte_udp_hdr);

            burst[nb_alloc++] = m;

            fs.pkts_left--;
            fs.pkts_sent++;

            if (fs.pkts_sent == gap_after_pkts && fs.pkts_left > 0) {
                fs.next_send_tsc = now + const_gap_tsc;
            } else if (fs.pkts_left > 0) {
                if (++fs.pkts_since_update >= 16) {
                    update_dynamic_gap(fs, now, tsc_hz, rng);
                    fs.pkts_since_update = 0;
                }
                fs.next_send_tsc = now + fs.current_gap_tsc;
            } else {
                fs.next_send_tsc = 0;
            }

            if (fs.pkts_left == 0) {
                active_flows[flow_idx] = active_flows.back();
                active_flows.pop_back();
                if (flow_idx >= active_flows.size() && !active_flows.empty()) flow_idx = 0;
            } else {
                flow_idx++;
            }
        }

        if (nb_alloc > 0) {
            uint16_t sent = rte_eth_tx_burst(PORT_ID, qid, burst, nb_alloc);
            rte_atomic64_add(&tx_pkts_total, sent);
            for (uint16_t i = sent; i < nb_alloc; i++) rte_pktmbuf_free(burst[i]);
        }
    }
    return 0;
}

/* --- Generate a synthetic dynamic flow profile. --- */
void generate_python_flow(FlowState *fs, uint64_t flow_id, uint64_t sec_start_tsc, uint64_t tsc_hz, std::mt19937 &rng) {
    uint32_t ip_offset = (uint32_t)(flow_id & 0x00FFFFFFULL);
    fs->src_ip   = rte_cpu_to_be_32(g_conf.base_ip + ip_offset);
    fs->dst_ip   = rte_cpu_to_be_32(RTE_IPV4(192, 168, 1, 1));
    fs->src_port = rte_cpu_to_be_16((uint16_t)(1024 + (flow_id % 50000)));
    fs->dst_port = rte_cpu_to_be_16(80);

    std::uniform_real_distribution<double> dist_01(0.0, 1.0);
    double r = dist_01(rng);
    double duration_s, base_gap_us;

    if (r < 0.58) { // short
        duration_s = 0.10 + dist_01(rng) * 0.90;
        base_gap_us = 120 + dist_01(rng) * 1280;
        fs->pkt_size = 300 + (uint16_t)(dist_01(rng) * 900);
        fs->burst_amp = 0.08 + dist_01(rng) * 0.22;
    } else if (r < 0.88) { // medium
        duration_s = 1.00 + dist_01(rng) * 7.00;
        base_gap_us = 50 + dist_01(rng) * 450;
        fs->pkt_size = 500 + (uint16_t)(dist_01(rng) * 950);
        fs->burst_amp = 0.05 + dist_01(rng) * 0.17;
    } else { // long
        duration_s = 8.00 + dist_01(rng) * 17.00;
        base_gap_us = 20 + dist_01(rng) * 140;
        fs->pkt_size = 700 + (uint16_t)(dist_01(rng) * 800);
        fs->burst_amp = 0.02 + dist_01(rng) * 0.10;
    }

    fs->jitter_amp = 0.03 + dist_01(rng) * 0.13;
    fs->pulse_period_s = 0.04 + dist_01(rng) * 0.86;
    fs->phase1 = dist_01(rng) * 2 * M_PI;
    fs->phase2 = dist_01(rng) * 2 * M_PI;

    fs->base_gap_tsc = (uint64_t)((base_gap_us / 1000000.0) * tsc_hz);
    fs->current_gap_tsc = fs->base_gap_tsc;

    double pps = 1000000.0 / base_gap_us;
    fs->pkts_total = (uint32_t)(duration_s * pps);
    if (fs->pkts_total == 0) fs->pkts_total = 1;

    fs->pkts_left = fs->pkts_total;
    fs->pkts_sent = 0;
    fs->pkts_since_update = 0;
    fs->next_send_tsc = 0;

    fs->start_tsc = sec_start_tsc + (uint64_t)(dist_01(rng) * tsc_hz);
}

int main(int argc, char **argv) {
    signal(SIGINT, signal_handler); signal(SIGTERM, signal_handler);

    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");

    rte_atomic64_init(&tx_pkts_total);
    rte_atomic64_init(&tx_flows_total);
    load_config();

    uint16_t worker_lcores = (uint16_t)(rte_lcore_count() - 1);
    g_active_workers = std::min<uint16_t>(MAX_TX_WORKERS, worker_lcores);

    // Mbuf pool
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", 262143, 512, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    // Flow pool
    flow_pool = rte_mempool_create("FLOW_POOL", FLOW_POOL_SIZE, sizeof(FlowState), 256, 0, nullptr, nullptr, nullptr, nullptr, rte_socket_id(), 0);

    rte_eth_conf eth_conf{};
    rte_eth_dev_configure(PORT_ID, 0, g_active_workers, &eth_conf);

    for (uint16_t q = 0; q < g_active_workers; q++) {
        rte_eth_tx_queue_setup(PORT_ID, q, 2048, rte_socket_id(), nullptr);
    }
    rte_eth_dev_start(PORT_ID);
    rte_eth_macaddr_get(PORT_ID, &g_sender_mac);

    WorkerCtx worker_ctx[MAX_TX_WORKERS]{};
    unsigned int lcore_id;
    uint16_t launched = 0;

    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (launched >= g_active_workers) break;
        worker_ctx[launched].qid = launched;
        worker_ctx[launched].worker_idx = launched;

        char ring_name[32];
        snprintf(ring_name, sizeof(ring_name), "TX_RING_%u", launched);
        worker_ctx[launched].rx_ring = rte_ring_create(ring_name, RING_SIZE, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);

        rte_eal_remote_launch(tx_worker, &worker_ctx[launched], lcore_id);
        launched++;
    }

    // --- Main control loop ---
    std::mt19937 rng_main(rte_rdtsc());
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);

    struct rte_eth_stats last_nic_stats{};
    rte_eth_stats_get(PORT_ID, &last_nic_stats);
    uint64_t sim_sec = 0;
    uint64_t last_tsc = rte_get_tsc_cycles();
    uint64_t global_flow_id = 0;
    const uint64_t tsc_hz = rte_get_tsc_hz();

    while (!force_quit) {
        uint64_t prev_pkts  = rte_atomic64_read(&tx_pkts_total);
        uint64_t prev_flows = rte_atomic64_read(&tx_flows_total);

        sleep(1);

        uint64_t now_tsc = rte_get_tsc_cycles();
        double dt = (double)(now_tsc - last_tsc) / tsc_hz;
        last_tsc = now_tsc;

        uint64_t cur_pkts  = rte_atomic64_read(&tx_pkts_total);
        uint64_t cur_flows = rte_atomic64_read(&tx_flows_total);

        struct rte_eth_stats nic_stats{};
        rte_eth_stats_get(PORT_ID, &nic_stats);

        double current_mpps = dt > 0 ? ((double)(cur_pkts - prev_pkts) / 1000000.0 / dt) : 0;
        double current_gbps = dt > 0 ? ((double)(nic_stats.obytes - last_nic_stats.obytes) * 8.0 / 1000000000.0 / dt) : 0;

        double t = (double)(sim_sec % 60);
        double slow = 8.0 + 4.8 * sin(2 * M_PI * t / 42.0 - 1.0);
        double mid1 = 3.8 * sin(2 * M_PI * t / 17.0 + 0.7);
        double mid2 = 2.7 * sin(2 * M_PI * t / 11.0 - 1.2);
        double fast1 = 1.2 * sin(2 * M_PI * t / 5.2 + 1.1);
        double fast2 = 0.7 * sin(2 * M_PI * t / 3.1 - 0.3);
        double amp_env = 0.82 + 0.28 * sin(2 * M_PI * t / 26.0 + 0.2);

        double target_mpps = slow + amp_env * (mid1 + mid2) + fast1 + fast2;
        target_mpps = std::max(0.35, std::min(23.5, target_mpps));

        double gap_mpps = target_mpps - current_mpps;
        int background = (int)(60 + target_mpps * 48);
        background = std::max(30, std::min(2800, background));

        uint32_t target_flows = 0;
        if (gap_mpps <= 0) {
            target_flows = (uint32_t)(background * (0.25 + dist_01(rng_main) * 0.55));
        } else {
            int need = (int)std::ceil((gap_mpps / 0.0042) * 0.88);
            target_flows = (uint32_t)((background + need) * (0.88 + dist_01(rng_main) * 0.24));
        }
        target_flows = std::min((uint32_t)2800, target_flows);

        for (uint32_t i = 0; i < target_flows; ++i) {
            void *obj = nullptr;
            if (rte_mempool_get(flow_pool, &obj) == 0) {
                FlowState *fs = static_cast<FlowState *>(obj);
                generate_python_flow(fs, global_flow_id++, now_tsc, tsc_hz, rng_main);

                uint16_t w_idx = i % g_active_workers;
                if (rte_ring_enqueue(worker_ctx[w_idx].rx_ring, obj) < 0) {
                    rte_mempool_put(flow_pool, obj);
                }
            }
        }

        printf("[Stats] Sec:%3lu | AppRate: %7.3f Mpps | %7.3f Gbps \n",
               sim_sec + 1, current_mpps, current_gbps);

        last_nic_stats = nic_stats;
        sim_sec++;
    }

    rte_eal_mp_wait_lcore();
    return 0;
}
