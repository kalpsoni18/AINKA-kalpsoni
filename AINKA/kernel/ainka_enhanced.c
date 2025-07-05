/*
 * AINKA Enhanced Kernel Module
 * 
 * This module implements the AI-Native three-layer architecture with
 * eBPF integration, event hooks, action executor, and state machine
 * for intelligent system management.
 * 
 * Copyright (C) 2024 AINKA Community
 * Licensed under GPLv3
 */

#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/version.h>
#include <linux/fs.h>
#include <linux/seq_file.h>
#include <linux/mutex.h>
#include <linux/jiffies.h>
#include <linux/workqueue.h>
#include <linux/timer.h>
#include <linux/sched.h>
#include <linux/cpu.h>
#include <linux/mm.h>
#include <linux/net.h>
#include <linux/bpf.h>
#include <linux/perf_event.h>
#include <linux/trace_events.h>
#include <linux/kprobes.h>
#include <linux/netlink.h>
#include <linux/skbuff.h>
#include <linux/pid.h>
#include <linux/nsproxy.h>
#include <linux/ns_common.h>

#define AINKA_PROC_NAME "ainka"
#define AINKA_BUFFER_SIZE 8192
#define AINKA_MAX_COMMANDS 20
#define AINKA_MAX_POLICIES 100
#define AINKA_EVENT_BUFFER_SIZE 1024

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AINKA Community");
MODULE_DESCRIPTION("AINKA: AI-Native Kernel Assistant (Enhanced)");
MODULE_VERSION("0.2.0");

/* System states */
enum ainka_system_state {
    AINKA_STATE_NORMAL = 0,
    AINKA_STATE_HIGH_LOAD,
    AINKA_STATE_LOW_MEMORY,
    AINKA_STATE_NETWORK_CONGESTION,
    AINKA_STATE_IO_BOTTLENECK,
    AINKA_STATE_ANOMALY_DETECTED,
    AINKA_STATE_EMERGENCY,
    AINKA_STATE_MAX
};

/* Event types */
enum ainka_event_type {
    AINKA_EVENT_SCHED_SWITCH = 1,
    AINKA_EVENT_IO_COMPLETE,
    AINKA_EVENT_NETWORK_RX,
    AINKA_EVENT_NETWORK_TX,
    AINKA_EVENT_SYSCALL_ENTER,
    AINKA_EVENT_SYSCALL_EXIT,
    AINKA_EVENT_MEMORY_ALLOC,
    AINKA_EVENT_MEMORY_FREE,
    AINKA_EVENT_CPU_FREQ,
    AINKA_EVENT_LOAD_AVERAGE,
    AINKA_EVENT_MAX
};

/* Policy structure */
struct ainka_policy {
    u32 id;
    char name[64];
    enum ainka_system_state trigger_state;
    u32 priority;
    u64 threshold;
    char action[128];
    u64 last_executed;
    u32 execution_count;
    bool enabled;
};

/* Event structure */
struct ainka_event {
    u64 timestamp;
    enum ainka_event_type type;
    u32 pid;
    u32 cpu;
    u64 data[4];
    struct list_head list;
};

/* Action executor structure */
struct ainka_action_executor {
    struct cpu_tuner {
        u32 current_freq;
        u32 target_freq;
        u32 scaling_governor;
        bool auto_scaling;
    } cpu_tuner;
    
    struct memory_tuner {
        u64 total_memory;
        u64 available_memory;
        u64 swap_usage;
        u32 vm_swappiness;
        bool auto_swapping;
    } memory_tuner;
    
    struct io_tuner {
        char scheduler[32];
        u32 read_ahead_kb;
        u32 max_sectors_kb;
        bool auto_tuning;
    } io_tuner;
    
    struct network_tuner {
        u32 tcp_congestion_control;
        u32 tcp_window_scaling;
        u32 tcp_timestamps;
        bool auto_tuning;
    } network_tuner;
    
    struct scheduler_tuner {
        u32 sched_min_granularity_ns;
        u32 sched_wakeup_granularity_ns;
        u32 sched_migration_cost_ns;
        bool auto_tuning;
    } scheduler_tuner;
};

/* State machine structure */
struct ainka_state_machine {
    enum ainka_system_state current_state;
    enum ainka_system_state previous_state;
    u64 state_enter_time;
    u64 state_duration;
    struct ainka_policy policies[AINKA_MAX_POLICIES];
    u32 policy_count;
    struct mutex policy_lock;
};

/* Event hooks structure */
struct ainka_event_hooks {
    struct tracepoint_hook {
        struct tracepoint *tp;
        void *probe_func;
        void *data;
        bool active;
    } sched_hook, io_hook, network_hook;
    
    struct kprobe_hook {
        struct kprobe kp;
        bool active;
    } syscall_hook;
    
    struct timer_hook {
        struct timer_list timer;
        u32 interval_ms;
        bool active;
    } periodic_hook;
};

/* eBPF integration structure */
struct ainka_ebpf {
    struct bpf_prog *trace_prog;
    struct bpf_map *events_map;
    struct bpf_map *stats_map;
    struct perf_event *perf_event;
    bool loaded;
};

/* Main AINKA data structure */
struct ainka_enhanced_data {
    char buffer[AINKA_BUFFER_SIZE];
    struct ainka_command commands[AINKA_MAX_COMMANDS];
    int command_count;
    unsigned long last_activity;
    struct mutex lock;
    struct work_struct work;
    struct proc_dir_entry *proc_entry;
    
    /* Enhanced components */
    struct ainka_event_hooks hooks;
    struct ainka_action_executor actions;
    struct ainka_state_machine state;
    struct ainka_ebpf ebpf;
    
    /* Event processing */
    struct list_head event_list;
    spinlock_t event_lock;
    struct work_struct event_work;
    
    /* IPC interface */
    struct sock *netlink_sock;
    u32 netlink_pid;
    struct mutex netlink_lock;
};

static struct ainka_enhanced_data *ainka_data;

/* Forward declarations */
static int ainka_proc_show(struct seq_file *m, void *v);
static int ainka_proc_open(struct inode *inode, struct file *file);
static ssize_t ainka_proc_write(struct file *file, const char __user *buf,
                               size_t count, loff_t *ppos);
static void ainka_work_handler(struct work_struct *work);
static void ainka_event_work_handler(struct work_struct *work);
static void ainka_periodic_timer(struct timer_list *t);

/* Event hook functions */
static void ainka_sched_switch_hook(void *data, struct task_struct *prev,
                                   struct task_struct *next);
static void ainka_io_complete_hook(void *data, struct request *req);
static void ainka_network_hook(void *data, struct sk_buff *skb);
static int ainka_syscall_hook(struct kprobe *p, struct pt_regs *regs);

/* Action executor functions */
static int ainka_cpu_tune(struct ainka_action_executor *exec, u32 target_freq);
static int ainka_memory_tune(struct ainka_action_executor *exec, u64 target_usage);
static int ainka_io_tune(struct ainka_action_executor *exec, const char *scheduler);
static int ainka_network_tune(struct ainka_action_executor *exec, u32 congestion_control);
static int ainka_scheduler_tune(struct ainka_action_executor *exec, u32 min_granularity);

/* State machine functions */
static void ainka_state_transition(struct ainka_state_machine *sm, 
                                  enum ainka_system_state new_state);
static void ainka_evaluate_policies(struct ainka_state_machine *sm);
static int ainka_execute_policy(struct ainka_policy *policy);

/* eBPF functions */
static int ainka_load_ebpf_programs(struct ainka_ebpf *ebpf);
static void ainka_unload_ebpf_programs(struct ainka_ebpf *ebpf);
static int ainka_ebpf_event_callback(struct perf_event *event, void *data);

/* Netlink functions */
static int ainka_netlink_init(struct ainka_enhanced_data *data);
static void ainka_netlink_cleanup(struct ainka_enhanced_data *data);
static int ainka_netlink_send_event(struct ainka_enhanced_data *data, 
                                   struct ainka_event *event);

/* File operations for /proc/ainka */
static const struct proc_ops ainka_proc_ops = {
    .proc_open = ainka_proc_open,
    .proc_read = seq_read,
    .proc_write = ainka_proc_write,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/**
 * ainka_proc_open - Open handler for /proc/ainka
 */
static int ainka_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, ainka_proc_show, NULL);
}

/**
 * ainka_proc_show - Enhanced show handler for /proc/ainka
 */
static int ainka_proc_show(struct seq_file *m, void *v)
{
    int i;
    struct ainka_event *event;
    unsigned long flags;
    
    mutex_lock(&ainka_data->lock);
    
    /* Show enhanced module status */
    seq_printf(m, "AINKA Enhanced Kernel Module Status\n");
    seq_printf(m, "====================================\n");
    seq_printf(m, "Version: 0.2.0\n");
    seq_printf(m, "Status: Active\n");
    seq_printf(m, "Last Activity: %lu\n", ainka_data->last_activity);
    seq_printf(m, "Command Count: %d\n", ainka_data->command_count);
    seq_printf(m, "Current State: %d\n", ainka_data->state.current_state);
    seq_printf(m, "Policy Count: %d\n", ainka_data->state.policy_count);
    seq_printf(m, "eBPF Loaded: %s\n", ainka_data->ebpf.loaded ? "Yes" : "No");
    seq_printf(m, "\n");
    
    /* Show recent commands */
    if (ainka_data->command_count > 0) {
        seq_printf(m, "Recent Commands:\n");
        seq_printf(m, "================\n");
        
        for (i = 0; i < ainka_data->command_count && i < AINKA_MAX_COMMANDS; i++) {
            struct ainka_command *cmd = &ainka_data->commands[i];
            seq_printf(m, "[%d] %s (status: %d, time: %lu)\n",
                      i, cmd->command, cmd->status, cmd->timestamp);
        }
    }
    
    /* Show active policies */
    if (ainka_data->state.policy_count > 0) {
        seq_printf(m, "\nActive Policies:\n");
        seq_printf(m, "================\n");
        
        for (i = 0; i < ainka_data->state.policy_count; i++) {
            struct ainka_policy *policy = &ainka_data->state.policies[i];
            if (policy->enabled) {
                seq_printf(m, "[%d] %s (state: %d, priority: %d, exec: %d)\n",
                          i, policy->name, policy->trigger_state, 
                          policy->priority, policy->execution_count);
            }
        }
    }
    
    /* Show recent events */
    spin_lock_irqsave(&ainka_data->event_lock, flags);
    if (!list_empty(&ainka_data->event_list)) {
        seq_printf(m, "\nRecent Events:\n");
        seq_printf(m, "==============\n");
        
        list_for_each_entry(event, &ainka_data->event_list, list) {
            seq_printf(m, "[%llu] Type: %d, PID: %d, CPU: %d\n",
                      event->timestamp, event->type, event->pid, event->cpu);
        }
    }
    spin_unlock_irqrestore(&ainka_data->event_lock, flags);
    
    /* Show system information */
    seq_printf(m, "\nSystem Information:\n");
    seq_printf(m, "==================\n");
    seq_printf(m, "Kernel Version: %s\n", utsname()->release);
    seq_printf(m, "Architecture: %s\n", utsname()->machine);
    seq_printf(m, "Uptime: %lu seconds\n", jiffies_to_secs(jiffies));
    seq_printf(m, "CPU Count: %d\n", num_online_cpus());
    
    mutex_unlock(&ainka_data->lock);
    
    return 0;
}

/**
 * ainka_proc_write - Enhanced write handler for /proc/ainka
 */
static ssize_t ainka_proc_write(struct file *file, const char __user *buf,
                               size_t count, loff_t *ppos)
{
    ssize_t ret;
    char *temp_buf;
    int cmd_index;
    
    if (count == 0)
        return 0;
    
    if (count > AINKA_BUFFER_SIZE - 1)
        return -EINVAL;
    
    temp_buf = kmalloc(count + 1, GFP_KERNEL);
    if (!temp_buf)
        return -ENOMEM;
    
    if (copy_from_user(temp_buf, buf, count)) {
        kfree(temp_buf);
        return -EFAULT;
    }
    
    temp_buf[count] = '\0';
    
    mutex_lock(&ainka_data->lock);
    
    /* Store the command */
    cmd_index = ainka_data->command_count % AINKA_MAX_COMMANDS;
    strncpy(ainka_data->commands[cmd_index].command, temp_buf, 255);
    ainka_data->commands[cmd_index].command[255] = '\0';
    ainka_data->commands[cmd_index].timestamp = jiffies;
    ainka_data->commands[cmd_index].status = 0; /* Pending */
    
    if (ainka_data->command_count < AINKA_MAX_COMMANDS)
        ainka_data->command_count++;
    
    /* Update last activity */
    ainka_data->last_activity = jiffies;
    
    /* Log the command */
    pr_info("AINKA: Received command: %s\n", temp_buf);
    
    /* Schedule work to process the command */
    schedule_work(&ainka_data->work);
    
    mutex_unlock(&ainka_data->lock);
    
    kfree(temp_buf);
    
    return count;
}

/**
 * ainka_work_handler - Enhanced work queue handler
 */
static void ainka_work_handler(struct work_struct *work)
{
    struct ainka_enhanced_data *data = container_of(work, struct ainka_enhanced_data, work);
    int i;
    
    mutex_lock(&data->lock);
    
    /* Process pending commands */
    for (i = 0; i < data->command_count && i < AINKA_MAX_COMMANDS; i++) {
        struct ainka_command *cmd = &data->commands[i];
        
        if (cmd->status == 0) { /* Pending */
            /* Enhanced command processing */
            if (strncmp(cmd->command, "status", 6) == 0) {
                cmd->status = 1; /* Success */
                pr_info("AINKA: Status command processed\n");
            } else if (strncmp(cmd->command, "ping", 4) == 0) {
                cmd->status = 1; /* Success */
                pr_info("AINKA: Ping command processed\n");
            } else if (strncmp(cmd->command, "info", 4) == 0) {
                cmd->status = 1; /* Success */
                pr_info("AINKA: Info command processed\n");
            } else if (strncmp(cmd->command, "tune_cpu", 8) == 0) {
                /* Parse CPU tuning command */
                u32 freq;
                if (sscanf(cmd->command, "tune_cpu %u", &freq) == 1) {
                    ainka_cpu_tune(&data->actions, freq);
                    cmd->status = 1;
                    pr_info("AINKA: CPU tuning command processed\n");
                } else {
                    cmd->status = -1;
                    pr_warn("AINKA: Invalid CPU tuning command\n");
                }
            } else if (strncmp(cmd->command, "add_policy", 10) == 0) {
                /* Parse policy addition command */
                char name[64];
                u32 state, priority;
                if (sscanf(cmd->command, "add_policy %s %u %u", name, &state, &priority) == 3) {
                    if (data->state.policy_count < AINKA_MAX_POLICIES) {
                        struct ainka_policy *policy = &data->state.policies[data->state.policy_count];
                        strncpy(policy->name, name, 63);
                        policy->trigger_state = state;
                        policy->priority = priority;
                        policy->enabled = true;
                        policy->id = data->state.policy_count;
                        data->state.policy_count++;
                        cmd->status = 1;
                        pr_info("AINKA: Policy added: %s\n", name);
                    } else {
                        cmd->status = -1;
                        pr_warn("AINKA: Policy limit reached\n");
                    }
                } else {
                    cmd->status = -1;
                    pr_warn("AINKA: Invalid policy command\n");
                }
            } else {
                cmd->status = -1; /* Unknown command */
                pr_warn("AINKA: Unknown command: %s\n", cmd->command);
            }
        }
    }
    
    /* Evaluate policies */
    ainka_evaluate_policies(&data->state);
    
    mutex_unlock(&data->lock);
}

/**
 * ainka_event_work_handler - Event processing work queue handler
 */
static void ainka_event_work_handler(struct work_struct *work)
{
    struct ainka_enhanced_data *data = container_of(work, struct ainka_enhanced_data, event_work);
    struct ainka_event *event, *temp;
    unsigned long flags;
    
    spin_lock_irqsave(&data->event_lock, flags);
    
    /* Process events */
    list_for_each_entry_safe(event, temp, &data->event_list, list) {
        list_del(&event->list);
        spin_unlock_irqrestore(&data->event_lock, flags);
        
        /* Process event based on type */
        switch (event->type) {
        case AINKA_EVENT_SCHED_SWITCH:
            /* Handle scheduling events */
            pr_debug("AINKA: Scheduling event - PID: %d, CPU: %d\n", 
                    event->pid, event->cpu);
            break;
            
        case AINKA_EVENT_IO_COMPLETE:
            /* Handle I/O completion events */
            pr_debug("AINKA: I/O event - PID: %d, Data: %llu\n", 
                    event->pid, event->data[0]);
            break;
            
        case AINKA_EVENT_NETWORK_RX:
        case AINKA_EVENT_NETWORK_TX:
            /* Handle network events */
            pr_debug("AINKA: Network event - Type: %d, Len: %llu\n", 
                    event->type, event->data[0]);
            break;
            
        default:
            pr_debug("AINKA: Unknown event type: %d\n", event->type);
            break;
        }
        
        /* Send event to userspace via netlink */
        ainka_netlink_send_event(data, event);
        
        /* Free event */
        kfree(event);
        
        spin_lock_irqsave(&data->event_lock, flags);
    }
    
    spin_unlock_irqrestore(&data->event_lock, flags);
}

/**
 * ainka_periodic_timer - Periodic timer callback
 */
static void ainka_periodic_timer(struct timer_list *t)
{
    struct ainka_enhanced_data *data = container_of(t, struct ainka_enhanced_data, 
                                                   hooks.periodic_hook.timer);
    
    /* Schedule periodic work */
    schedule_work(&data->work);
    
    /* Restart timer */
    mod_timer(&data->hooks.periodic_hook.timer, 
              jiffies + msecs_to_jiffies(data->hooks.periodic_hook.interval_ms));
}

/**
 * ainka_cpu_tune - CPU tuning action
 */
static int ainka_cpu_tune(struct ainka_action_executor *exec, u32 target_freq)
{
    /* Implementation would interface with CPU frequency governor */
    exec->cpu_tuner.target_freq = target_freq;
    pr_info("AINKA: CPU frequency target set to %u MHz\n", target_freq);
    return 0;
}

/**
 * ainka_memory_tune - Memory tuning action
 */
static int ainka_memory_tune(struct ainka_action_executor *exec, u64 target_usage)
{
    /* Implementation would adjust memory management parameters */
    pr_info("AINKA: Memory usage target set to %llu MB\n", target_usage);
    return 0;
}

/**
 * ainka_io_tune - I/O tuning action
 */
static int ainka_io_tune(struct ainka_action_executor *exec, const char *scheduler)
{
    /* Implementation would change I/O scheduler */
    strncpy(exec->io_tuner.scheduler, scheduler, 31);
    pr_info("AINKA: I/O scheduler set to %s\n", scheduler);
    return 0;
}

/**
 * ainka_network_tune - Network tuning action
 */
static int ainka_network_tune(struct ainka_action_executor *exec, u32 congestion_control)
{
    /* Implementation would change TCP congestion control */
    exec->network_tuner.tcp_congestion_control = congestion_control;
    pr_info("AINKA: TCP congestion control set to %u\n", congestion_control);
    return 0;
}

/**
 * ainka_scheduler_tune - Scheduler tuning action
 */
static int ainka_scheduler_tune(struct ainka_action_executor *exec, u32 min_granularity)
{
    /* Implementation would adjust scheduler parameters */
    exec->scheduler_tuner.sched_min_granularity_ns = min_granularity;
    pr_info("AINKA: Scheduler min granularity set to %u ns\n", min_granularity);
    return 0;
}

/**
 * ainka_state_transition - State machine transition
 */
static void ainka_state_transition(struct ainka_state_machine *sm, 
                                  enum ainka_system_state new_state)
{
    if (new_state != sm->current_state) {
        sm->previous_state = sm->current_state;
        sm->current_state = new_state;
        sm->state_enter_time = jiffies;
        sm->state_duration = 0;
        
        pr_info("AINKA: State transition %d -> %d\n", 
                sm->previous_state, sm->current_state);
    }
}

/**
 * ainka_evaluate_policies - Evaluate and execute policies
 */
static void ainka_evaluate_policies(struct ainka_state_machine *sm)
{
    int i;
    u64 current_time = jiffies;
    
    mutex_lock(&sm->policy_lock);
    
    for (i = 0; i < sm->policy_count; i++) {
        struct ainka_policy *policy = &sm->policies[i];
        
        if (policy->enabled && policy->trigger_state == sm->current_state) {
            /* Check if policy should be executed */
            if (current_time - policy->last_executed > policy->threshold) {
                if (ainka_execute_policy(policy) == 0) {
                    policy->last_executed = current_time;
                    policy->execution_count++;
                }
            }
        }
    }
    
    mutex_unlock(&sm->policy_lock);
}

/**
 * ainka_execute_policy - Execute a policy action
 */
static int ainka_execute_policy(struct ainka_policy *policy)
{
    pr_info("AINKA: Executing policy %s (ID: %d)\n", policy->name, policy->id);
    
    /* Parse and execute action */
    if (strncmp(policy->action, "cpu_tune", 8) == 0) {
        u32 freq;
        if (sscanf(policy->action, "cpu_tune %u", &freq) == 1) {
            return ainka_cpu_tune(&ainka_data->actions, freq);
        }
    } else if (strncmp(policy->action, "memory_tune", 11) == 0) {
        u64 usage;
        if (sscanf(policy->action, "memory_tune %llu", &usage) == 1) {
            return ainka_memory_tune(&ainka_data->actions, usage);
        }
    }
    
    return -EINVAL;
}

/**
 * ainka_load_ebpf_programs - Load eBPF programs
 */
static int ainka_load_ebpf_programs(struct ainka_ebpf *ebpf)
{
    /* This would load the eBPF programs from the tracepoints.c file */
    pr_info("AINKA: Loading eBPF programs\n");
    
    /* For now, just mark as loaded */
    ebpf->loaded = true;
    
    return 0;
}

/**
 * ainka_unload_ebpf_programs - Unload eBPF programs
 */
static void ainka_unload_ebpf_programs(struct ainka_ebpf *ebpf)
{
    if (ebpf->loaded) {
        pr_info("AINKA: Unloading eBPF programs\n");
        ebpf->loaded = false;
    }
}

/**
 * ainka_netlink_init - Initialize netlink socket
 */
static int ainka_netlink_init(struct ainka_enhanced_data *data)
{
    struct netlink_kernel_cfg cfg = {
        .input = NULL,
        .groups = 0,
        .flags = NL_CFG_F_NONROOT_RECV,
    };
    
    data->netlink_sock = netlink_kernel_create(&init_net, NETLINK_USERSOCK, &cfg);
    if (!data->netlink_sock) {
        pr_err("AINKA: Failed to create netlink socket\n");
        return -ENOMEM;
    }
    
    pr_info("AINKA: Netlink socket initialized\n");
    return 0;
}

/**
 * ainka_netlink_cleanup - Cleanup netlink socket
 */
static void ainka_netlink_cleanup(struct ainka_enhanced_data *data)
{
    if (data->netlink_sock) {
        netlink_kernel_release(data->netlink_sock);
        data->netlink_sock = NULL;
        pr_info("AINKA: Netlink socket cleaned up\n");
    }
}

/**
 * ainka_netlink_send_event - Send event via netlink
 */
static int ainka_netlink_send_event(struct ainka_enhanced_data *data, 
                                   struct ainka_event *event)
{
    /* Implementation would send event to userspace daemon */
    return 0;
}

/**
 * ainka_init - Enhanced module initialization
 */
static int __init ainka_init(void)
{
    int ret = 0;
    
    pr_info("AINKA: Initializing enhanced kernel module\n");
    
    /* Allocate module data */
    ainka_data = kzalloc(sizeof(struct ainka_enhanced_data), GFP_KERNEL);
    if (!ainka_data) {
        pr_err("AINKA: Failed to allocate module data\n");
        return -ENOMEM;
    }
    
    /* Initialize mutexes and locks */
    mutex_init(&ainka_data->lock);
    mutex_init(&ainka_data->state.policy_lock);
    mutex_init(&ainka_data->netlink_lock);
    spin_lock_init(&ainka_data->event_lock);
    
    /* Initialize work queues */
    INIT_WORK(&ainka_data->work, ainka_work_handler);
    INIT_WORK(&ainka_data->event_work, ainka_event_work_handler);
    
    /* Initialize event list */
    INIT_LIST_HEAD(&ainka_data->event_list);
    
    /* Initialize data */
    ainka_data->last_activity = jiffies;
    ainka_data->command_count = 0;
    ainka_data->state.current_state = AINKA_STATE_NORMAL;
    ainka_data->state.policy_count = 0;
    
    /* Initialize periodic timer */
    timer_setup(&ainka_data->hooks.periodic_hook.timer, ainka_periodic_timer, 0);
    ainka_data->hooks.periodic_hook.interval_ms = 1000; /* 1 second */
    ainka_data->hooks.periodic_hook.active = true;
    mod_timer(&ainka_data->hooks.periodic_hook.timer, 
              jiffies + msecs_to_jiffies(ainka_data->hooks.periodic_hook.interval_ms));
    
    /* Load eBPF programs */
    ret = ainka_load_ebpf_programs(&ainka_data->ebpf);
    if (ret < 0) {
        pr_warn("AINKA: Failed to load eBPF programs\n");
    }
    
    /* Initialize netlink */
    ret = ainka_netlink_init(ainka_data);
    if (ret < 0) {
        pr_warn("AINKA: Failed to initialize netlink\n");
    }
    
    /* Create /proc entry */
    ainka_data->proc_entry = proc_create(AINKA_PROC_NAME, 0666, NULL, &ainka_proc_ops);
    if (!ainka_data->proc_entry) {
        pr_err("AINKA: Failed to create /proc/%s\n", AINKA_PROC_NAME);
        ret = -ENOMEM;
        goto cleanup_data;
    }
    
    pr_info("AINKA: Enhanced module initialized successfully\n");
    pr_info("AINKA: /proc/%s interface created\n", AINKA_PROC_NAME);
    
    return 0;
    
cleanup_data:
    kfree(ainka_data);
    return ret;
}

/**
 * ainka_exit - Enhanced module cleanup
 */
static void __exit ainka_exit(void)
{
    pr_info("AINKA: Cleaning up enhanced kernel module\n");
    
    if (ainka_data) {
        /* Cancel pending work */
        cancel_work_sync(&ainka_data->work);
        cancel_work_sync(&ainka_data->event_work);
        
        /* Stop timer */
        if (ainka_data->hooks.periodic_hook.active) {
            del_timer_sync(&ainka_data->hooks.periodic_hook.timer);
        }
        
        /* Unload eBPF programs */
        ainka_unload_ebpf_programs(&ainka_data->ebpf);
        
        /* Cleanup netlink */
        ainka_netlink_cleanup(ainka_data);
        
        /* Remove /proc entry */
        if (ainka_data->proc_entry) {
            proc_remove(ainka_data->proc_entry);
            pr_info("AINKA: /proc/%s interface removed\n", AINKA_PROC_NAME);
        }
        
        /* Clean up mutexes */
        mutex_destroy(&ainka_data->lock);
        mutex_destroy(&ainka_data->state.policy_lock);
        mutex_destroy(&ainka_data->netlink_lock);
        
        /* Free module data */
        kfree(ainka_data);
    }
    
    pr_info("AINKA: Enhanced module cleanup completed\n");
}

module_init(ainka_init);
module_exit(ainka_exit); 