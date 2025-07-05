/*
 * AINKA eBPF Tracepoint Programs
 * 
 * This file contains eBPF programs for monitoring system events
 * through tracepoints, providing safe and efficient data collection
 * for the AI-Native kernel assistant.
 * 
 * Copyright (C) 2024 AINKA Community
 * Licensed under GPLv2
 */

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/net.h>
#include <linux/skbuff.h>
#include <linux/trace_events.h>
#include <linux/version.h>

/* BPF license */
char _license[] SEC("license") = "GPL";

/* Data structures for event collection */
struct ainka_event {
    u64 timestamp;
    u32 event_type;
    u32 pid;
    u32 cpu;
    u64 data[4];
};

/* Event types */
#define AINKA_EVENT_SCHED_SWITCH    1
#define AINKA_EVENT_IO_COMPLETE     2
#define AINKA_EVENT_NETWORK_RX      3
#define AINKA_EVENT_NETWORK_TX      4
#define AINKA_EVENT_SYSCALL_ENTER   5
#define AINKA_EVENT_SYSCALL_EXIT    6
#define AINKA_EVENT_MEMORY_ALLOC    7
#define AINKA_EVENT_MEMORY_FREE     8

/* BPF maps for data collection */
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(u32));
    __uint(max_entries, 1024);
} ainka_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u64));
    __uint(max_entries, 10000);
} ainka_stats SEC(".maps");

/* CPU scheduling tracepoint */
SEC("tracepoint/sched/sched_switch")
int trace_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    u32 pid = ctx->next_pid;
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_SCHED_SWITCH;
    event.pid = pid;
    event.cpu = cpu;
    event.data[0] = ctx->prev_pid;
    event.data[1] = ctx->next_pid;
    event.data[2] = ctx->prev_state;
    event.data[3] = ctx->next_comm[0]; // First char of comm
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    /* Update statistics */
    u32 key = AINKA_EVENT_SCHED_SWITCH;
    u64 *count = bpf_map_lookup_elem(&ainka_stats, &key);
    if (count) {
        (*count)++;
    } else {
        u64 new_count = 1;
        bpf_map_update_elem(&ainka_stats, &key, &new_count, BPF_ANY);
    }
    
    return 0;
}

/* Block I/O completion tracepoint */
SEC("tracepoint/block/block_rq_complete")
int trace_block_complete(struct trace_event_raw_block_rq_complete *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_IO_COMPLETE;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = ctx->dev;
    event.data[1] = ctx->sector;
    event.data[2] = ctx->nr_sector;
    event.data[3] = ctx->errors;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    /* Update statistics */
    u32 key = AINKA_EVENT_IO_COMPLETE;
    u64 *count = bpf_map_lookup_elem(&ainka_stats, &key);
    if (count) {
        (*count)++;
    } else {
        u64 new_count = 1;
        bpf_map_update_elem(&ainka_stats, &key, &new_count, BPF_ANY);
    }
    
    return 0;
}

/* Network receive tracepoint */
SEC("tracepoint/net/netif_receive_skb")
int trace_net_rx(struct trace_event_raw_netif_receive_skb *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    struct sk_buff *skb = ctx->skbaddr;
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_NETWORK_RX;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = skb->len;
    event.data[1] = skb->protocol;
    event.data[2] = skb->pkt_type;
    event.data[3] = 0;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    /* Update statistics */
    u32 key = AINKA_EVENT_NETWORK_RX;
    u64 *count = bpf_map_lookup_elem(&ainka_stats, &key);
    if (count) {
        (*count)++;
    } else {
        u64 new_count = 1;
        bpf_map_update_elem(&ainka_stats, &key, &new_count, BPF_ANY);
    }
    
    return 0;
}

/* Network transmit tracepoint */
SEC("tracepoint/net/netif_xmit")
int trace_net_tx(struct trace_event_raw_netif_xmit *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    struct sk_buff *skb = ctx->skbaddr;
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_NETWORK_TX;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = skb->len;
    event.data[1] = skb->protocol;
    event.data[2] = ctx->rc;
    event.data[3] = 0;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    /* Update statistics */
    u32 key = AINKA_EVENT_NETWORK_TX;
    u64 *count = bpf_map_lookup_elem(&ainka_stats, &key);
    if (count) {
        (*count)++;
    } else {
        u64 new_count = 1;
        bpf_map_update_elem(&ainka_stats, &key, &new_count, BPF_ANY);
    }
    
    return 0;
}

/* Memory allocation tracepoint */
SEC("tracepoint/kmem/kmalloc")
int trace_memory_alloc(struct trace_event_raw_kmalloc *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_MEMORY_ALLOC;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = ctx->bytes_req;
    event.data[1] = ctx->bytes_alloc;
    event.data[2] = ctx->gfp_flags;
    event.data[3] = 0;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    /* Update statistics */
    u32 key = AINKA_EVENT_MEMORY_ALLOC;
    u64 *count = bpf_map_lookup_elem(&ainka_stats, &key);
    if (count) {
        (*count)++;
    } else {
        u64 new_count = 1;
        bpf_map_update_elem(&ainka_stats, &key, &new_count, BPF_ANY);
    }
    
    return 0;
}

/* Memory free tracepoint */
SEC("tracepoint/kmem/kfree")
int trace_memory_free(struct trace_event_raw_kfree *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_MEMORY_FREE;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = ctx->call_site;
    event.data[1] = 0;
    event.data[2] = 0;
    event.data[3] = 0;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    /* Update statistics */
    u32 key = AINKA_EVENT_MEMORY_FREE;
    u64 *count = bpf_map_lookup_elem(&ainka_stats, &key);
    if (count) {
        (*count)++;
    } else {
        u64 new_count = 1;
        bpf_map_update_elem(&ainka_stats, &key, &new_count, BPF_ANY);
    }
    
    return 0;
}

/* System call entry kprobe */
SEC("kprobe/do_sys_openat2")
int kprobe_sys_openat2(struct pt_regs *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_SYSCALL_ENTER;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = 257; // __NR_openat2
    event.data[1] = 0;
    event.data[2] = 0;
    event.data[3] = 0;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    return 0;
}

/* System call exit kretprobe */
SEC("kretprobe/do_sys_openat2")
int kretprobe_sys_openat2(struct pt_regs *ctx)
{
    struct ainka_event event = {};
    u32 cpu = bpf_get_smp_processor_id();
    long ret = PT_REGS_RC(ctx);
    
    event.timestamp = bpf_ktime_get_ns();
    event.event_type = AINKA_EVENT_SYSCALL_EXIT;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.cpu = cpu;
    event.data[0] = 257; // __NR_openat2
    event.data[1] = ret;
    event.data[2] = 0;
    event.data[3] = 0;
    
    /* Send event to userspace */
    bpf_perf_event_output(ctx, &ainka_events, BPF_F_CURRENT_CPU, &event, sizeof(event));
    
    return 0;
} 