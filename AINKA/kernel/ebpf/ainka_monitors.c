/*
 * AINKA eBPF Programs
 * Safe kernel-space monitoring and data collection for AI decision making
 */

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/net.h>

// =============================================================================
// SYSCALL_TRACER.C - Monitor system calls for pattern detection
// =============================================================================

struct syscall_event {
    __u32 pid;
    __u32 tgid;
    __u32 syscall_nr;
    __u64 timestamp;
    __u64 duration;
    __s32 retval;
    char comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} syscall_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);
    __type(value, __u64);
} syscall_start_time SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 512);
    __type(key, __u32);
    __type(value, __u64);
} syscall_counts SEC(".maps");

SEC("tracepoint/raw_syscalls/sys_enter")
int trace_syscall_enter(struct trace_event_raw_sys_enter *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u64 ts = bpf_ktime_get_ns();
    
    // Record start time for duration calculation
    bpf_map_update_elem(&syscall_start_time, &pid, &ts, BPF_ANY);
    
    // Count syscalls per type
    __u32 syscall_nr = ctx->id;
    __u64 *count = bpf_map_lookup_elem(&syscall_counts, &syscall_nr);
    if (count) {
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&syscall_counts, &syscall_nr, &initial, BPF_ANY);
    }
    
    return 0;
}

SEC("tracepoint/raw_syscalls/sys_exit")
int trace_syscall_exit(struct trace_event_raw_sys_exit *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    
    __u64 *start_ts = bpf_map_lookup_elem(&syscall_start_time, &pid);
    if (!start_ts) {
        return 0;
    }
    
    __u64 end_ts = bpf_ktime_get_ns();
    __u64 duration = end_ts - *start_ts;
    
    // Only report long-running or frequently called syscalls
    if (duration > 100000 || ctx->id == 0 || ctx->id == 1 || ctx->id == 3) { // read/write/close
        struct syscall_event *event = bpf_ringbuf_reserve(&syscall_events, sizeof(*event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->syscall_nr = ctx->id;
            event->timestamp = *start_ts;
            event->duration = duration;
            event->retval = ctx->ret;
            bpf_get_current_comm(event->comm, sizeof(event->comm));
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    bpf_map_delete_elem(&syscall_start_time, &pid);
    return 0;
}

// =============================================================================
// MEMORY_MONITOR.C - Monitor memory allocation and pressure
// =============================================================================

struct memory_event {
    __u32 pid;
    __u32 tgid;
    __u64 size;
    __u64 address;
    __u64 timestamp;
    __u8 operation; // 0=alloc, 1=free, 2=oom
    char comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} memory_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32); // PID
    __type(value, __u64); // total allocated bytes
} process_memory SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} total_allocations SEC(".maps");

SEC("kprobe/__kmalloc")
int trace_kmalloc(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    
    size_t size = PT_REGS_PARM1(ctx);
    
    // Track large allocations
    if (size > 4096) { // > 4KB
        struct memory_event *event = bpf_ringbuf_reserve(&memory_events, sizeof(*event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->size = size;
            event->address = 0; // Will be filled on return
            event->timestamp = bpf_ktime_get_ns();
            event->operation = 0; // alloc
            bpf_get_current_comm(event->comm, sizeof(event->comm));
            
            bpf_ringbuf_submit(event, 0);
        }
        
        // Update per-process memory tracking
        __u64 *total = bpf_map_lookup_elem(&process_memory, &pid);
        if (total) {
            __sync_fetch_and_add(total, size);
        } else {
            bpf_map_update_elem(&process_memory, &pid, &size, BPF_ANY);
        }
        
        // Update global allocation counter
        __u32 key = 0;
        __u64 *global_total = bpf_map_lookup_elem(&total_allocations, &key);
        if (global_total) {
            __sync_fetch_and_add(global_total, size);
        } else {
            bpf_map_update_elem(&total_allocations, &key, &size, BPF_ANY);
        }
    }
    
    return 0;
}

SEC("kprobe/kfree")
int trace_kfree(struct pt_regs *ctx)
{
    // Note: kfree doesn't provide size, so we can't track exact deallocations
    // In a real implementation, we'd need to track allocations in a map
    return 0;
}

SEC("tracepoint/oom/oom_score_adj_update")
int trace_oom_score_update(void *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    
    struct memory_event *event = bpf_ringbuf_reserve(&memory_events, sizeof(*event), 0);
    if (event) {
        event->pid = pid;
        event->tgid = tgid;
        event->size = 0;
        event->address = 0;
        event->timestamp = bpf_ktime_get_ns();
        event->operation = 2; // oom event
        bpf_get_current_comm(event->comm, sizeof(event->comm));
        
        bpf_ringbuf_submit(event, 0);
    }
    
    return 0;
}

// =============================================================================
// PERFORMANCE_MONITOR.C - Monitor system performance metrics
// =============================================================================

struct perf_event {
    __u64 timestamp;
    __u32 cpu;
    __u64 cycles;
    __u64 instructions;
    __u64 cache_misses;
    __u64 branch_misses;
    __u32 load_avg_1m;
    __u32 nr_running;
    __u32 nr_uninterruptible;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 64 * 1024);
} perf_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 8); // Performance counter types
    __type(key, __u32);
    __type(value, __u64);
} perf_counters SEC(".maps");

SEC("perf_event")
int collect_perf_data(struct bpf_perf_event_data *ctx)
{
    __u32 cpu = bpf_get_smp_processor_id();
    __u64 ts = bpf_ktime_get_ns();
    
    // Sample performance data periodically
    static __u64 last_sample_time = 0;
    if (ts - last_sample_time < 1000000000) { // 1 second interval
        return 0;
    }
    last_sample_time = ts;
    
    struct perf_event *event = bpf_ringbuf_reserve(&perf_events, sizeof(*event), 0);
    if (event) {
        event->timestamp = ts;
        event->cpu = cpu;
        
        // Read hardware performance counters
        event->cycles = bpf_perf_event_read(&perf_counters, 0);
        event->instructions = bpf_perf_event_read(&perf_counters, 1);
        event->cache_misses = bpf_perf_event_read(&perf_counters, 2);
        event->branch_misses = bpf_perf_event_read(&perf_counters, 3);
        
        // Get system load information (simplified)
        event->load_avg_1m = 0; // Would need to read from /proc/loadavg equivalent
        event->nr_running = 0;
        event->nr_uninterruptible = 0;
        
        bpf_ringbuf_submit(event, 0);
    }
    
    return 0;
}

// =============================================================================
// SECURITY_MONITOR.C - Monitor security-relevant events
// =============================================================================

struct security_event {
    __u32 pid;
    __u32 tgid;
    __u32 uid;
    __u32 gid;
    __u64 timestamp;
    __u8 event_type; // 0=capability_check, 1=setuid, 2=execve, 3=ptrace
    __u8 result; // 0=denied, 1=allowed
    __u32 capability; // for capability checks
    char comm[16];
    char target_comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 128 * 1024);
} security_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 1000);
    __type(key, __u32); // UID
    __type(value, __u64); // suspicious activity count
} suspicious_activity SEC(".maps");

SEC("lsm/capable")
int monitor_capability_check(struct bpf_capable_args *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    __u32 uid = bpf_get_current_uid_gid();
    __u32 gid = bpf_get_current_uid_gid() >> 32;
    
    // Monitor privileged capability checks
    int cap = ctx->cap;
    if (cap == CAP_SYS_ADMIN || cap == CAP_DAC_OVERRIDE || cap == CAP_SETUID) {
        struct security_event *event = bpf_ringbuf_reserve(&security_events, sizeof(*event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->uid = uid;
            event->gid = gid;
            event->timestamp = bpf_ktime_get_ns();
            event->event_type = 0; // capability_check
            event->result = 1; // allowed (if we reach here)
            event->capability = cap;
            bpf_get_current_comm(event->comm, sizeof(event->comm));
            __builtin_memset(event->target_comm, 0, sizeof(event->target_comm));
            
            bpf_ringbuf_submit(event, 0);
        }
        
        // Track suspicious activity
        __u64 *activity_count = bpf_map_lookup_elem(&suspicious_activity, &uid);
        if (activity_count) {
            __sync_fetch_and_add(activity_count, 1);
        } else {
            __u64 initial = 1;
            bpf_map_update_elem(&suspicious_activity, &uid, &initial, BPF_ANY);
        }
    }
    
    return 0;
}

SEC("tracepoint/syscalls/sys_enter_execve")
int monitor_execve(struct trace_event_raw_sys_enter *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    __u32 uid = bpf_get_current_uid_gid();
    __u32 gid = bpf_get_current_uid_gid() >> 32;
    
    struct security_event *event = bpf_ringbuf_reserve(&security_events, sizeof(*event), 0);
    if (event) {
        event->pid = pid;
        event->tgid = tgid;
        event->uid = uid;
        event->gid = gid;
        event->timestamp = bpf_ktime_get_ns();
        event->event_type = 2; // execve
        event->result = 1; // will be executed
        event->capability = 0;
        bpf_get_current_comm(event->comm, sizeof(event->comm));
        __builtin_memset(event->target_comm, 0, sizeof(event->target_comm));
        
        bpf_ringbuf_submit(event, 0);
    }
    
    return 0;
}

// =============================================================================
// NETWORK_MONITOR.C - Monitor network traffic and patterns
// =============================================================================

struct network_event {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 protocol;
    __u32 bytes;
    __u64 timestamp;
    __u8 direction; // 0=ingress, 1=egress
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} network_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 10000);
    __type(key, __u64); // flow hash
    __type(value, __u64); // byte count
} flow_stats SEC(".maps");

static __always_inline __u64 flow_hash(__u32 saddr, __u32 daddr, __u16 sport, __u16 dport, __u8 proto)
{
    return ((__u64)saddr << 32) | daddr | ((__u64)sport << 16) | dport | ((__u64)proto << 48);
}

SEC("tc/ingress")
int monitor_ingress(struct __sk_buff *skb)
{
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;
    
    if (data + sizeof(struct ethhdr) > data_end)
        return TC_ACT_OK;
        
    struct ethhdr *eth = data;
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return TC_ACT_OK;
        
    struct iphdr *ip = data + sizeof(struct ethhdr);
    if ((void *)(ip + 1) > data_end)
        return TC_ACT_OK;
        
    __u16 sport = 0, dport = 0;
    if (ip->protocol == IPPROTO_TCP) {
        struct tcphdr *tcp = (void *)ip + (ip->ihl * 4);
        if ((void *)(tcp + 1) > data_end)
            return TC_ACT_OK;
        sport = bpf_ntohs(tcp->source);
        dport = bpf_ntohs(tcp->dest);
    } else if (ip->protocol == IPPROTO_UDP) {
        struct udphdr *udp = (void *)ip + (ip->ihl * 4);
        if ((void *)(udp + 1) > data_end)
            return TC_ACT_OK;
        sport = bpf_ntohs(udp->source);
        dport = bpf_ntohs(udp->dest);
    }
    
    __u64 hash = flow_hash(ip->saddr, ip->daddr, sport, dport, ip->protocol);
    __u32 bytes = skb->len;
    
    // Update flow statistics
    __u64 *flow_bytes = bpf_map_lookup_elem(&flow_stats, &hash);
    if (flow_bytes) {
        __sync_fetch_and_add(flow_bytes, bytes);
    } else {
        __u64 initial = bytes;
        bpf_map_update_elem(&flow_stats, &hash, &initial, BPF_ANY);
    }
    
    // Report significant flows
    if (bytes > 1500 || (flow_bytes && *flow_bytes > 1024 * 1024)) {
        struct network_event *event = bpf_ringbuf_reserve(&network_events, sizeof(*event), 0);
        if (event) {
            event->src_ip = ip->saddr;
            event->dst_ip = ip->daddr;
            event->src_port = sport;
            event->dst_port = dport;
            event->protocol = ip->protocol;
            event->bytes = bytes;
            event->timestamp = bpf_ktime_get_ns();
            event->direction = 0; // ingress
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    return TC_ACT_OK;
}

// =============================================================================
// IO_TRACER.C - Monitor I/O operations and patterns
// =============================================================================

struct io_event {
    __u32 pid;
    __u32 tgid;
    __u64 inode;
    __u64 offset;
    __u32 bytes;
    __u64 latency;
    __u64 timestamp;
    __u8 operation; // 0=read, 1=write
    char filename[64];
    char comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} io_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);
    __type(value, __u64);
} io_start_time SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 1000);
    __type(key, __u64); // inode
    __type(value, __u64); // total bytes
} file_io_stats SEC(".maps");

SEC("kprobe/vfs_read")
int trace_vfs_read_enter(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u64 ts = bpf_ktime_get_ns();
    
    bpf_map_update_elem(&io_start_time, &pid, &ts, BPF_ANY);
    return 0;
}

SEC("kretprobe/vfs_read")
int trace_vfs_read_exit(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    
    __u64 *start_ts = bpf_map_lookup_elem(&io_start_time, &pid);
    if (!start_ts)
        return 0;
        
    __u64 end_ts = bpf_ktime_get_ns();
    __u64 latency = end_ts - *start_ts;
    __s64 bytes_read = PT_REGS_RC(ctx);
    
    if (bytes_read > 0 && (latency > 1000000 || bytes_read > 4096)) { // > 1ms or > 4KB
        struct io_event *event = bpf_ringbuf_reserve(&io_events, sizeof(*event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->bytes = bytes_read;
            event->latency = latency;
            event->timestamp = *start_ts;
            event->operation = 0; // read
            bpf_get_current_comm(event->comm, sizeof(event->comm));
            
            // Try to get filename (simplified)
            event->inode = 0;
            event->offset = 0;
            __builtin_memset(event->filename, 0, sizeof(event->filename));
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    bpf_map_delete_elem(&io_start_time, &pid);
    return 0;
}

SEC("kprobe/vfs_write")
int trace_vfs_write_enter(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u64 ts = bpf_ktime_get_ns();
    
    bpf_map_update_elem(&io_start_time, &pid, &ts, BPF_ANY);
    return 0;
}

SEC("kretprobe/vfs_write")
int trace_vfs_write_exit(struct pt_regs *ctx)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid;
    __u32 tgid = pid_tgid >> 32;
    
    __u64 *start_ts = bpf_map_lookup_elem(&io_start_time, &pid);
    if (!start_ts)
        return 0;
        
    __u64 end_ts = bpf_ktime_get_ns();
    __u64 latency = end_ts - *start_ts;
    __s64 bytes_written = PT_REGS_RC(ctx);
    
    if (bytes_written > 0 && (latency > 1000000 || bytes_written > 4096)) { // > 1ms or > 4KB
        struct io_event *event = bpf_ringbuf_reserve(&io_events, sizeof(*event), 0);
        if (event) {
            event->pid = pid;
            event->tgid = tgid;
            event->bytes = bytes_written;
            event->latency = latency;
            event->timestamp = *start_ts;
            event->operation = 1; // write
            bpf_get_current_comm(event->comm, sizeof(event->comm));
            
            event->inode = 0;
            event->offset = 0;
            __builtin_memset(event->filename, 0, sizeof(event->filename));
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    bpf_map_delete_elem(&io_start_time, &pid);
    return 0;
}

// =============================================================================
// SCHEDULER_HOOK.C - Monitor CPU scheduling and context switches
// =============================================================================

struct sched_event {
    __u32 prev_pid;
    __u32 next_pid;
    __u64 timestamp;
    __u64 runtime;
    __u8 prev_state;
    __u8 cpu;
    char prev_comm[16];
    char next_comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} sched_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32); // PID
    __type(value, __u64); // start time
} task_start_time SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} context_switch_count SEC(".maps");

SEC("tracepoint/sched/sched_switch")
int trace_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    __u64 ts = bpf_ktime_get_ns();
    __u32 cpu = bpf_get_smp_processor_id();
    
    // Calculate runtime for previous task
    __u64 *start_ts = bpf_map_lookup_elem(&task_start_time, &ctx->prev_pid);
    __u64 runtime = 0;
    if (start_ts) {
        runtime = ts - *start_ts;
    }
    
    // Record start time for next task
    bpf_map_update_elem(&task_start_time, &ctx->next_pid, &ts, BPF_ANY);
    
    // Count context switches
    __u32 key = 0;
    __u64 *count = bpf_map_lookup_elem(&context_switch_count, &key);
    if (count) {
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&context_switch_count, &key, &initial, BPF_ANY);
    }
    
    // Report significant events (long running tasks or frequent switches)
    if (runtime > 10000000 || ctx->prev_state != TASK_RUNNING) { // > 10ms or blocked
        struct sched_event *event = bpf_ringbuf_reserve(&sched_events, sizeof(*event), 0);
        if (event) {
            event->prev_pid = ctx->prev_pid;
            event->next_pid = ctx->next_pid;
            event->timestamp = ts;
            event->runtime = runtime;
            event->prev_state = ctx->prev_state;
            event->cpu = cpu;
            
            bpf_probe_read_kernel_str(event->prev_comm, sizeof(event->prev_comm), ctx->prev_comm);
            bpf_probe_read_kernel_str(event->next_comm, sizeof(event->next_comm), ctx->next_comm);
            
            bpf_ringbuf_submit(event, 0);
        }
    }
    
    return 0;
}

// =============================================================================
// SHARED UTILITIES AND HELPER FUNCTIONS
// =============================================================================

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} ainka_config SEC(".maps");

// Configuration flags
#define AINKA_CONFIG_ENABLE_SYSCALL_TRACE   (1ULL << 0)
#define AINKA_CONFIG_ENABLE_NETWORK_TRACE   (1ULL << 1)
#define AINKA_CONFIG_ENABLE_IO_TRACE        (1ULL << 2)
#define AINKA_CONFIG_ENABLE_SCHED_TRACE     (1ULL << 3)
#define AINKA_CONFIG_ENABLE_MEMORY_TRACE    (1ULL << 4)
#define AINKA_CONFIG_ENABLE_PERF_TRACE      (1ULL << 5)
#define AINKA_CONFIG_ENABLE_SECURITY_TRACE  (1ULL << 6)
#define AINKA_CONFIG_DEBUG_MODE             (1ULL << 63)

static __always_inline int ainka_is_enabled(__u64 flag)
{
    __u32 key = 0;
    __u64 *config = bpf_map_lookup_elem(&ainka_config, &key);
    if (!config)
        return 1; // Default to enabled
    return (*config & flag) != 0;
}

static __always_inline void ainka_debug_print(const char *fmt, ...)
{
    if (ainka_is_enabled(AINKA_CONFIG_DEBUG_MODE)) {
        char msg[128];
        __builtin_va_list args;
        __builtin_va_start(args, fmt);
        bpf_trace_vprintk(fmt, sizeof(fmt), args);
        __builtin_va_end(args);
    }
}

// Rate limiting for event generation
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 1000);
    __type(key, __u64); // event hash
    __type(value, __u64); // last timestamp
} rate_limit_map SEC(".maps");

static __always_inline int ainka_rate_limit(__u64 event_hash, __u64 min_interval_ns)
{
    __u64 now = bpf_ktime_get_ns();
    __u64 *last_time = bpf_map_lookup_elem(&rate_limit_map, &event_hash);
    
    if (last_time && (now - *last_time) < min_interval_ns) {
        return 1; // Rate limited
    }
    
    bpf_map_update_elem(&rate_limit_map, &event_hash, &now, BPF_ANY);
    return 0; // Not rate limited
}

// Hash function for event deduplication
static __always_inline __u64 ainka_hash_event(__u32 pid, __u32 event_type, __u64 context)
{
    return ((__u64)pid << 32) | event_type | (context & 0xFFFF);
}

// =============================================================================
// LOADER AND MANAGEMENT FUNCTIONS
// =============================================================================

SEC("tracepoint/bpf/bpf_prog_load")
int ainka_prog_load_monitor(void *ctx)
{
    // Monitor when new BPF programs are loaded
    // This helps track if other monitoring tools are interfering
    ainka_debug_print("BPF program loaded\n");
    return 0;
}

// Auto-attach function for easier deployment
SEC("raw_tracepoint/sys_enter")
int ainka_auto_attach(struct bpf_raw_tracepoint_args *ctx)
{
    // This function serves as an entry point for automatic attachment
    // The actual attachment logic would be handled by the userspace loader
    return 0;
}

char _license[] SEC("license") = "GPL";
__u32 _version SEC("version") = LINUX_VERSION_CODE; 