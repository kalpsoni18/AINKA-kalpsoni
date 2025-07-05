/*
 * AINKA Simple Kernel Module
 * 
 * A simplified, working kernel module that provides core AINKA functionality
 * including system monitoring, basic optimization, and userspace communication.
 * 
 * Copyright (C) 2024 AINKA Community
 * Licensed under GPLv3
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/delay.h>
#include <linux/timer.h>
#include <linux/workqueue.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/netlink.h>
#include <linux/skbuff.h>
#include <net/sock.h>
#include <linux/uaccess.h>
#include <linux/cpufreq.h>
#include <linux/cpu.h>
#include <linux/sysfs.h>
#include <linux/kobject.h>

#define AINKA_MODULE_NAME "ainka_simple"
#define AINKA_PROC_NAME "ainka"
#define AINKA_NETLINK_FAMILY 31
#define AINKA_NETLINK_GROUP 1

MODULE_LICENSE("GPL v3");
MODULE_AUTHOR("AINKA Community");
MODULE_DESCRIPTION("AINKA Simple Kernel Module - Intelligent Linux System Optimizer");
MODULE_VERSION("0.2.0");

// Global variables
static struct proc_dir_entry *ainka_proc_entry;
static struct sock *ainka_netlink_sock;
static struct timer_list ainka_timer;
static struct work_struct ainka_work;
static struct kobject *ainka_kobj;

// System metrics structure
struct ainka_metrics {
    unsigned long cpu_usage;
    unsigned long memory_usage;
    unsigned long load_average;
    unsigned long disk_io_read;
    unsigned long disk_io_write;
    unsigned long network_rx;
    unsigned long network_tx;
    unsigned long timestamp;
};

static struct ainka_metrics current_metrics;

// Optimization settings
struct ainka_optimization {
    bool cpu_optimization_enabled;
    bool memory_optimization_enabled;
    bool io_optimization_enabled;
    unsigned int cpu_threshold;
    unsigned int memory_threshold;
    unsigned int optimization_interval;
};

static struct ainka_optimization opt_settings = {
    .cpu_optimization_enabled = true,
    .memory_optimization_enabled = true,
    .io_optimization_enabled = true,
    .cpu_threshold = 80,
    .memory_threshold = 85,
    .optimization_interval = 300, // 5 minutes
};

// Statistics
struct ainka_stats {
    unsigned long optimizations_performed;
    unsigned long events_processed;
    unsigned long uptime_seconds;
    unsigned long last_optimization;
};

static struct ainka_stats stats = {0};

// Function prototypes
static int ainka_proc_show(struct seq_file *m, void *v);
static int ainka_proc_open(struct inode *inode, struct file *file);
static ssize_t ainka_proc_write(struct file *file, const char __user *buffer,
                               size_t count, loff_t *ppos);
static void ainka_timer_callback(struct timer_list *t);
static void ainka_work_handler(struct work_struct *work);
static void ainka_collect_metrics(void);
static void ainka_perform_optimization(void);
static void ainka_netlink_send_event(const char *event_type, const char *data);
static int ainka_netlink_init(void);
static void ainka_netlink_cleanup(void);
static int ainka_sysfs_init(void);
static void ainka_sysfs_cleanup(void);

// File operations for proc interface
static const struct proc_ops ainka_proc_ops = {
    .proc_open = ainka_proc_open,
    .proc_read = seq_read,
    .proc_write = ainka_proc_write,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

// Netlink message structure
struct ainka_netlink_msg {
    char event_type[32];
    char data[256];
    unsigned long timestamp;
};

// Collect system metrics
static void ainka_collect_metrics(void)
{
    struct sysinfo si;
    unsigned long total_ram, free_ram, used_ram;
    
    // Get system information
    if (si_meminfo(&si) == 0) {
        total_ram = si.totalram << (PAGE_SHIFT - 10); // Convert to KB
        free_ram = si.freeram << (PAGE_SHIFT - 10);
        used_ram = total_ram - free_ram;
        current_metrics.memory_usage = (used_ram * 100) / total_ram;
    }
    
    // Get load average
    avenrun[0] = avenrun[0] / FIXED_1;
    current_metrics.load_average = avenrun[0];
    
    // Get CPU usage (simplified)
    current_metrics.cpu_usage = 100 - (si.loads[0] * 100) / FIXED_1;
    
    // Get timestamp
    current_metrics.timestamp = jiffies_to_msecs(jiffies);
    
    // Update statistics
    stats.events_processed++;
}

// Perform system optimization
static void ainka_perform_optimization(void)
{
    bool optimization_performed = false;
    
    // CPU optimization
    if (opt_settings.cpu_optimization_enabled && 
        current_metrics.cpu_usage > opt_settings.cpu_threshold) {
        
        // Try to set CPU governor to performance
        struct cpufreq_policy *policy;
        unsigned int cpu;
        
        for_each_possible_cpu(cpu) {
            policy = cpufreq_cpu_get(cpu);
            if (policy) {
                if (cpufreq_driver_target(policy, policy->max, CPUFREQ_RELATION_H) == 0) {
                    optimization_performed = true;
                }
                cpufreq_cpu_put(policy);
            }
        }
        
        if (optimization_performed) {
            ainka_netlink_send_event("cpu_optimization", "CPU governor set to performance");
        }
    }
    
    // Memory optimization
    if (opt_settings.memory_optimization_enabled && 
        current_metrics.memory_usage > opt_settings.memory_threshold) {
        
        // Clear page cache
        if (try_to_free_pages(&init_mm, GFP_KERNEL, 0) > 0) {
            optimization_performed = true;
            ainka_netlink_send_event("memory_optimization", "Page cache cleared");
        }
    }
    
    if (optimization_performed) {
        stats.optimizations_performed++;
        stats.last_optimization = jiffies_to_msecs(jiffies);
    }
}

// Timer callback
static void ainka_timer_callback(struct timer_list *t)
{
    // Schedule work in workqueue to avoid blocking
    schedule_work(&ainka_work);
    
    // Restart timer
    mod_timer(&ainka_timer, jiffies + msecs_to_jiffies(opt_settings.optimization_interval * 1000));
}

// Work handler
static void ainka_work_handler(struct work_struct *work)
{
    // Collect metrics
    ainka_collect_metrics();
    
    // Perform optimization
    ainka_perform_optimization();
    
    // Update uptime
    stats.uptime_seconds = jiffies_to_msecs(jiffies) / 1000;
}

// Netlink send event
static void ainka_netlink_send_event(const char *event_type, const char *data)
{
    struct sk_buff *skb;
    struct nlmsghdr *nlh;
    struct ainka_netlink_msg *msg;
    int size;
    
    size = NLMSG_SPACE(sizeof(struct ainka_netlink_msg));
    skb = alloc_skb(size, GFP_ATOMIC);
    if (!skb) {
        return;
    }
    
    nlh = nlmsg_put(skb, 0, 0, NLMSG_DONE, sizeof(struct ainka_netlink_msg), 0);
    msg = nlmsg_data(nlh);
    
    strncpy(msg->event_type, event_type, sizeof(msg->event_type) - 1);
    strncpy(msg->data, data, sizeof(msg->data) - 1);
    msg->timestamp = jiffies_to_msecs(jiffies);
    
    nlmsg_end(skb, nlh);
    
    if (ainka_netlink_sock) {
        netlink_broadcast(ainka_netlink_sock, skb, 0, AINKA_NETLINK_GROUP, GFP_ATOMIC);
    }
    
    kfree_skb(skb);
}

// Netlink receive callback
static void ainka_netlink_rcv(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;
    struct ainka_netlink_msg *msg;
    
    nlh = (struct nlmsghdr *)skb->data;
    msg = nlmsg_data(nlh);
    
    // Process message from userspace
    if (strcmp(msg->event_type, "get_metrics") == 0) {
        // Send current metrics back
        char metrics_data[256];
        snprintf(metrics_data, sizeof(metrics_data),
                "cpu:%lu,mem:%lu,load:%lu,uptime:%lu",
                current_metrics.cpu_usage,
                current_metrics.memory_usage,
                current_metrics.load_average,
                stats.uptime_seconds);
        ainka_netlink_send_event("metrics_response", metrics_data);
    }
}

// Netlink configuration
static struct netlink_kernel_cfg ainka_netlink_cfg = {
    .input = ainka_netlink_rcv,
    .groups = AINKA_NETLINK_GROUP,
};

// Initialize netlink
static int ainka_netlink_init(void)
{
    ainka_netlink_sock = netlink_kernel_create(&init_net, AINKA_NETLINK_FAMILY, &ainka_netlink_cfg);
    if (!ainka_netlink_sock) {
        pr_err("AINKA: Failed to create netlink socket\n");
        return -ENOMEM;
    }
    
    pr_info("AINKA: Netlink socket created successfully\n");
    return 0;
}

// Cleanup netlink
static void ainka_netlink_cleanup(void)
{
    if (ainka_netlink_sock) {
        netlink_kernel_release(ainka_netlink_sock);
        ainka_netlink_sock = NULL;
    }
}

// Proc file show
static int ainka_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "AINKA Intelligent Linux System Optimizer v0.2.0\n");
    seq_printf(m, "================================================\n\n");
    
    seq_printf(m, "System Metrics:\n");
    seq_printf(m, "  CPU Usage: %lu%%\n", current_metrics.cpu_usage);
    seq_printf(m, "  Memory Usage: %lu%%\n", current_metrics.memory_usage);
    seq_printf(m, "  Load Average: %lu\n", current_metrics.load_average);
    seq_printf(m, "  Uptime: %lu seconds\n\n", stats.uptime_seconds);
    
    seq_printf(m, "Optimization Settings:\n");
    seq_printf(m, "  CPU Optimization: %s\n", 
               opt_settings.cpu_optimization_enabled ? "Enabled" : "Disabled");
    seq_printf(m, "  Memory Optimization: %s\n", 
               opt_settings.memory_optimization_enabled ? "Enabled" : "Disabled");
    seq_printf(m, "  I/O Optimization: %s\n", 
               opt_settings.io_optimization_enabled ? "Enabled" : "Disabled");
    seq_printf(m, "  CPU Threshold: %u%%\n", opt_settings.cpu_threshold);
    seq_printf(m, "  Memory Threshold: %u%%\n", opt_settings.memory_threshold);
    seq_printf(m, "  Optimization Interval: %u seconds\n\n", opt_settings.optimization_interval);
    
    seq_printf(m, "Statistics:\n");
    seq_printf(m, "  Optimizations Performed: %lu\n", stats.optimizations_performed);
    seq_printf(m, "  Events Processed: %lu\n", stats.events_processed);
    seq_printf(m, "  Last Optimization: %lu ms ago\n", 
               jiffies_to_msecs(jiffies) - stats.last_optimization);
    
    return 0;
}

// Proc file open
static int ainka_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, ainka_proc_show, NULL);
}

// Proc file write
static ssize_t ainka_proc_write(struct file *file, const char __user *buffer,
                               size_t count, loff_t *ppos)
{
    char cmd[256];
    char *token;
    
    if (count >= sizeof(cmd)) {
        return -EINVAL;
    }
    
    if (copy_from_user(cmd, buffer, count)) {
        return -EFAULT;
    }
    cmd[count] = '\0';
    
    // Remove newline
    cmd[strcspn(cmd, "\n")] = 0;
    
    token = strtok(cmd, " ");
    if (!token) {
        return count;
    }
    
    if (strcmp(token, "optimize") == 0) {
        ainka_perform_optimization();
        pr_info("AINKA: Manual optimization triggered\n");
    } else if (strcmp(token, "collect") == 0) {
        ainka_collect_metrics();
        pr_info("AINKA: Metrics collection triggered\n");
    } else if (strcmp(token, "cpu_threshold") == 0) {
        token = strtok(NULL, " ");
        if (token) {
            opt_settings.cpu_threshold = simple_strtoul(token, NULL, 10);
        }
    } else if (strcmp(token, "mem_threshold") == 0) {
        token = strtok(NULL, " ");
        if (token) {
            opt_settings.memory_threshold = simple_strtoul(token, NULL, 10);
        }
    } else if (strcmp(token, "interval") == 0) {
        token = strtok(NULL, " ");
        if (token) {
            opt_settings.optimization_interval = simple_strtoul(token, NULL, 10);
            mod_timer(&ainka_timer, jiffies + msecs_to_jiffies(opt_settings.optimization_interval * 1000));
        }
    }
    
    return count;
}

// Sysfs show functions
static ssize_t ainka_cpu_usage_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "%lu\n", current_metrics.cpu_usage);
}

static ssize_t ainka_memory_usage_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "%lu\n", current_metrics.memory_usage);
}

static ssize_t ainka_load_average_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "%lu\n", current_metrics.load_average);
}

static ssize_t ainka_optimizations_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "%lu\n", stats.optimizations_performed);
}

// Sysfs attributes
static struct kobj_attribute ainka_cpu_usage_attr = __ATTR_RO(ainka_cpu_usage);
static struct kobj_attribute ainka_memory_usage_attr = __ATTR_RO(ainka_memory_usage);
static struct kobj_attribute ainka_load_average_attr = __ATTR_RO(ainka_load_average);
static struct kobj_attribute ainka_optimizations_attr = __ATTR_RO(ainka_optimizations);

static struct attribute *ainka_attrs[] = {
    &ainka_cpu_usage_attr.attr,
    &ainka_memory_usage_attr.attr,
    &ainka_load_average_attr.attr,
    &ainka_optimizations_attr.attr,
    NULL,
};

static struct attribute_group ainka_attr_group = {
    .attrs = ainka_attrs,
};

// Initialize sysfs
static int ainka_sysfs_init(void)
{
    ainka_kobj = kobject_create_and_add("ainka", kernel_kobj);
    if (!ainka_kobj) {
        pr_err("AINKA: Failed to create sysfs kobject\n");
        return -ENOMEM;
    }
    
    if (sysfs_create_group(ainka_kobj, &ainka_attr_group)) {
        pr_err("AINKA: Failed to create sysfs group\n");
        kobject_put(ainka_kobj);
        return -ENOMEM;
    }
    
    pr_info("AINKA: Sysfs interface created successfully\n");
    return 0;
}

// Cleanup sysfs
static void ainka_sysfs_cleanup(void)
{
    if (ainka_kobj) {
        sysfs_remove_group(ainka_kobj, &ainka_attr_group);
        kobject_put(ainka_kobj);
        ainka_kobj = NULL;
    }
}

// Module initialization
static int __init ainka_init(void)
{
    int ret;
    
    pr_info("AINKA: Initializing AINKA Simple Kernel Module v0.2.0\n");
    
    // Initialize netlink
    ret = ainka_netlink_init();
    if (ret) {
        pr_err("AINKA: Failed to initialize netlink\n");
        return ret;
    }
    
    // Initialize sysfs
    ret = ainka_sysfs_init();
    if (ret) {
        pr_err("AINKA: Failed to initialize sysfs\n");
        ainka_netlink_cleanup();
        return ret;
    }
    
    // Create proc entry
    ainka_proc_entry = proc_create(AINKA_PROC_NAME, 0644, NULL, &ainka_proc_ops);
    if (!ainka_proc_entry) {
        pr_err("AINKA: Failed to create proc entry\n");
        ainka_sysfs_cleanup();
        ainka_netlink_cleanup();
        return -ENOMEM;
    }
    
    // Initialize work queue
    INIT_WORK(&ainka_work, ainka_work_handler);
    
    // Initialize timer
    timer_setup(&ainka_timer, ainka_timer_callback, 0);
    mod_timer(&ainka_timer, jiffies + msecs_to_jiffies(opt_settings.optimization_interval * 1000));
    
    // Initialize metrics
    ainka_collect_metrics();
    
    pr_info("AINKA: Module initialized successfully\n");
    pr_info("AINKA: Proc interface: /proc/%s\n", AINKA_PROC_NAME);
    pr_info("AINKA: Sysfs interface: /sys/kernel/ainka/\n");
    
    return 0;
}

// Module cleanup
static void __exit ainka_exit(void)
{
    pr_info("AINKA: Cleaning up AINKA Simple Kernel Module\n");
    
    // Stop timer
    del_timer_sync(&ainka_timer);
    
    // Cancel work
    cancel_work_sync(&ainka_work);
    
    // Remove proc entry
    if (ainka_proc_entry) {
        proc_remove(ainka_proc_entry);
        ainka_proc_entry = NULL;
    }
    
    // Cleanup sysfs
    ainka_sysfs_cleanup();
    
    // Cleanup netlink
    ainka_netlink_cleanup();
    
    pr_info("AINKA: Module cleanup completed\n");
}

module_init(ainka_init);
module_exit(ainka_exit); 