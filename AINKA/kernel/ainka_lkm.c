/*
 * AINKA Kernel Module
 * 
 * This module provides kernel-level AI assistant functionality by exposing
 * a /proc interface for communication with userspace daemon.
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

#define AINKA_PROC_NAME "ainka"
#define AINKA_BUFFER_SIZE 4096
#define AINKA_MAX_COMMANDS 10

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AINKA Community");
MODULE_DESCRIPTION("AINKA: Kernel-level AI Assistant");
MODULE_VERSION("0.1.0");

/* AINKA command structure */
struct ainka_command {
    char command[256];
    unsigned long timestamp;
    int status;
};

/* AINKA module data structure */
struct ainka_data {
    char buffer[AINKA_BUFFER_SIZE];
    struct ainka_command commands[AINKA_MAX_COMMANDS];
    int command_count;
    unsigned long last_activity;
    struct mutex lock;
    struct work_struct work;
    struct proc_dir_entry *proc_entry;
};

static struct ainka_data *ainka_data;

/* Forward declarations */
static int ainka_proc_show(struct seq_file *m, void *v);
static int ainka_proc_open(struct inode *inode, struct file *file);
static ssize_t ainka_proc_write(struct file *file, const char __user *buf,
                               size_t count, loff_t *ppos);
static void ainka_work_handler(struct work_struct *work);

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
 * @inode: inode structure
 * @file: file structure
 *
 * Returns: 0 on success, negative error code on failure
 */
static int ainka_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, ainka_proc_show, NULL);
}

/**
 * ainka_proc_show - Show handler for /proc/ainka
 * @m: seq_file structure
 * @v: void pointer
 *
 * Returns: 0 on success, negative error code on failure
 */
static int ainka_proc_show(struct seq_file *m, void *v)
{
    int i;
    unsigned long flags;
    
    mutex_lock(&ainka_data->lock);
    
    /* Show module status */
    seq_printf(m, "AINKA Kernel Module Status\n");
    seq_printf(m, "==========================\n");
    seq_printf(m, "Version: 0.1.0\n");
    seq_printf(m, "Status: Active\n");
    seq_printf(m, "Last Activity: %lu\n", ainka_data->last_activity);
    seq_printf(m, "Command Count: %d\n", ainka_data->command_count);
    seq_printf(m, "Buffer Size: %d\n", AINKA_BUFFER_SIZE);
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
    
    /* Show system information */
    seq_printf(m, "\nSystem Information:\n");
    seq_printf(m, "==================\n");
    seq_printf(m, "Kernel Version: %s\n", utsname()->release);
    seq_printf(m, "Architecture: %s\n", utsname()->machine);
    seq_printf(m, "Uptime: %lu seconds\n", jiffies_to_secs(jiffies));
    
    mutex_unlock(&ainka_data->lock);
    
    return 0;
}

/**
 * ainka_proc_write - Write handler for /proc/ainka
 * @file: file structure
 * @buf: user buffer
 * @count: number of bytes to write
 * @ppos: position pointer
 *
 * Returns: number of bytes written on success, negative error code on failure
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
 * ainka_work_handler - Work queue handler for processing commands
 * @work: work structure
 */
static void ainka_work_handler(struct work_struct *work)
{
    struct ainka_data *data = container_of(work, struct ainka_data, work);
    int i;
    
    mutex_lock(&data->lock);
    
    /* Process pending commands */
    for (i = 0; i < data->command_count && i < AINKA_MAX_COMMANDS; i++) {
        struct ainka_command *cmd = &data->commands[i];
        
        if (cmd->status == 0) { /* Pending */
            /* Simple command processing - can be extended */
            if (strncmp(cmd->command, "status", 6) == 0) {
                cmd->status = 1; /* Success */
                pr_info("AINKA: Status command processed\n");
            } else if (strncmp(cmd->command, "ping", 4) == 0) {
                cmd->status = 1; /* Success */
                pr_info("AINKA: Ping command processed\n");
            } else if (strncmp(cmd->command, "info", 4) == 0) {
                cmd->status = 1; /* Success */
                pr_info("AINKA: Info command processed\n");
            } else {
                cmd->status = -1; /* Unknown command */
                pr_warn("AINKA: Unknown command: %s\n", cmd->command);
            }
        }
    }
    
    mutex_unlock(&data->lock);
}

/**
 * ainka_init - Module initialization function
 *
 * Returns: 0 on success, negative error code on failure
 */
static int __init ainka_init(void)
{
    int ret = 0;
    
    pr_info("AINKA: Initializing kernel module\n");
    
    /* Allocate module data */
    ainka_data = kzalloc(sizeof(struct ainka_data), GFP_KERNEL);
    if (!ainka_data) {
        pr_err("AINKA: Failed to allocate module data\n");
        return -ENOMEM;
    }
    
    /* Initialize mutex */
    mutex_init(&ainka_data->lock);
    
    /* Initialize work queue */
    INIT_WORK(&ainka_data->work, ainka_work_handler);
    
    /* Initialize data */
    ainka_data->last_activity = jiffies;
    ainka_data->command_count = 0;
    
    /* Create /proc entry */
    ainka_data->proc_entry = proc_create(AINKA_PROC_NAME, 0666, NULL, &ainka_proc_ops);
    if (!ainka_data->proc_entry) {
        pr_err("AINKA: Failed to create /proc/%s\n", AINKA_PROC_NAME);
        ret = -ENOMEM;
        goto cleanup_data;
    }
    
    pr_info("AINKA: Module initialized successfully\n");
    pr_info("AINKA: /proc/%s interface created\n", AINKA_PROC_NAME);
    
    return 0;
    
cleanup_data:
    kfree(ainka_data);
    return ret;
}

/**
 * ainka_exit - Module cleanup function
 */
static void __exit ainka_exit(void)
{
    pr_info("AINKA: Cleaning up kernel module\n");
    
    if (ainka_data) {
        /* Cancel pending work */
        cancel_work_sync(&ainka_data->work);
        
        /* Remove /proc entry */
        if (ainka_data->proc_entry) {
            proc_remove(ainka_data->proc_entry);
            pr_info("AINKA: /proc/%s interface removed\n", AINKA_PROC_NAME);
        }
        
        /* Clean up mutex */
        mutex_destroy(&ainka_data->lock);
        
        /* Free module data */
        kfree(ainka_data);
    }
    
    pr_info("AINKA: Module cleanup completed\n");
}

module_init(ainka_init);
module_exit(ainka_exit); 