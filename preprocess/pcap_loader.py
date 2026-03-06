import os
import glob
import random
import numpy as np
from scapy.all import rdpcap, PcapReader, IP, TCP, UDP
from tqdm import tqdm

import torch

def process_single_pcap(pcap_path, label):
    """
    Reads a single pcap file and returns a list of flow dictionaries.
    Returns: [{'packets': [...], 'label': label}, ...]
    """
    flows = {}
    
    try:
        # Use PcapReader for memory efficiency with large files
        with PcapReader(pcap_path) as packets:
            for pkt in packets:
                if IP in pkt and (TCP in pkt or UDP in pkt):
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    proto = pkt[IP].proto
                    
                    if TCP in pkt:
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                    else:
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                        
                    # Simplified 5-tuple key
                    key = (src_ip, dst_ip, src_port, dst_port, proto)
                    rev_key = (dst_ip, src_ip, dst_port, src_port, proto)
                    
                    pkt_info = {
                        'timestamp': float(pkt.time),
                        'length': len(pkt),
                        'src': src_ip
                    }
                    
                    if key in flows:
                        flows[key].append(pkt_info)
                    elif rev_key in flows:
                        # For reverse flow, we treat it as part of the same flow
                        flows[rev_key].append(pkt_info)
                    else:
                        flows[key] = [pkt_info]
                        
    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")
        return []

    processed_flows = []
    
    for key, flow_pkts in flows.items():
        if len(flow_pkts) < 5: # Filter short flows
            continue
            
        # Sort by timestamp
        flow_pkts.sort(key=lambda x: x['timestamp'])
        start_time = flow_pkts[0]['timestamp']
        src_ip_init = key[0] # The IP that started the flow (or first seen)
        
        structured_packets = []
        for pkt in flow_pkts:
            # Normalize timestamp
            ts = pkt['timestamp'] - start_time
            # Direction: 0 (fwd), 1 (bwd)
            # Compare packet source IP with flow initiator IP
            direction = 0 if pkt['src'] == src_ip_init else 1
            
            structured_packets.append({
                'timestamp': ts,
                'length': pkt['length'],
                'direction': direction
            })
            
        processed_flows.append({
            'packets': structured_packets,
            'label': label
        })
        
    return processed_flows

def load_encrypted_traffic_dataset(data_dir, max_flows=None, split='train'):
    """
    Loads encrypted traffic dataset.
    If processed .pt files exist, load them.
    Otherwise, load from pcap, split, save, and return.
    
    Args:
        split (str): 'train', 'val', or 'test'
    """
    processed_files = {
        'train': os.path.join(data_dir, 'processed_train.pt'),
        'val': os.path.join(data_dir, 'processed_val.pt'),
        'test': os.path.join(data_dir, 'processed_test.pt'),
        'label_map': os.path.join(data_dir, 'label_map.pt')
    }
    
    # Check if files exist
    if os.path.exists(processed_files['train']) and \
       os.path.exists(processed_files['val']) and \
       os.path.exists(processed_files['test']) and \
       os.path.exists(processed_files['label_map']):
           
        print(f"Loading cached {split} data from {processed_files[split]}...")
        data = torch.load(processed_files[split])
        label_map = torch.load(processed_files['label_map'])
        return data, label_map

    print("Cached data not found. Processing from PCAP files...")
    
    all_flows = []
    label_map = {}
    label_counter = 0
    
    # List categories
    categories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found categories: {categories}")
    
    flow_buffer = []

    for category in categories:
        label_map[label_counter] = category
        cat_dir = os.path.join(data_dir, category)
        # Recursive glob in case of subfolders
        pcap_files = glob.glob(os.path.join(cat_dir, "**/*.pcap"), recursive=True)
        
        print(f"Processing category '{category}' with {len(pcap_files)} pcaps...")
        
        for pcap_path in tqdm(pcap_files, desc=f"Loading {category}"):
            flows = process_single_pcap(pcap_path, label_counter)
            flow_buffer.extend(flows)
            
        label_counter += 1
        
    print(f"Total flows extracted: {len(flow_buffer)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(flow_buffer)
    
    # Split
    total = len(flow_buffer)
    test_split = int(total * 0.8) # 80% Train+Val, 20% Test
    
    train_val_buf = flow_buffer[:test_split]
    test_buf = flow_buffer[test_split:]
    
    # Split Train into Train/Val (e.g., 80% Train, 20% Val OF THE TRAINING SET)
    train_total = len(train_val_buf)
    val_split = int(train_total * 0.8)
    
    train_buf = train_val_buf[:val_split]
    val_buf = train_val_buf[val_split:]
    
    print(f"Split sizes -> Train: {len(train_buf)}, Val: {len(val_buf)}, Test: {len(test_buf)}")
    
    # Save
    print("Saving processed datasets...")
    torch.save(train_buf, processed_files['train'])
    torch.save(val_buf, processed_files['val'])
    torch.save(test_buf, processed_files['test'])
    torch.save(label_map, processed_files['label_map'])
    
    if split == 'train':
        return train_buf, label_map
    elif split == 'val':
        return val_buf, label_map
    else:
        return test_buf, label_map

