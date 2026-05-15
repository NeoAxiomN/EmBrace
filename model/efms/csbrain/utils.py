import yaml
import os


def load_dataset_configs(config_path="config.yaml"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
             
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_csbrain_config(dataset_name, configs):
    if dataset_name not in configs:
        raise ValueError(f"Dataset {dataset_name} not in config list")
    
    cfg = configs[dataset_name]
    regions = cfg.get("regions")
    ch_names = cfg.get("ch_names")
    topology = cfg.get("topology")
    
    keep_indices = []
    filtered_regions = []
    for i, r_id in enumerate(regions):
        if r_id != -1:
            keep_indices.append(i)
            filtered_regions.append(r_id)

    indexed_channels = []
    for i, (r_id, name) in enumerate(zip(filtered_regions, [ch_names[idx] for idx in keep_indices])):
        indexed_channels.append({'rel_idx': i, 'region': r_id, 'name': name.upper()})

    sorted_indices = []
    for r_id in sorted(set(filtered_regions)):
        group = [item for item in indexed_channels if item['region'] == r_id]
        
        if str(r_id) not in topology:
            raise KeyError(f"数据集中脑区 {r_id} 缺少 topology 定义")
            
        topo_order = [t.upper() for t in topology[str(r_id)]]
        
        try:
            sorted_group = sorted(
                group, 
                key=lambda x: topo_order.index(x['name'])
            )
        except ValueError as e:
            # 找到具体是哪个电极没在 YAML 里写
            missing_names = [x['name'] for x in group if x['name'] not in topo_order]
            raise ValueError(f"脑区 {r_id} 的拓扑定义缺失电极: {missing_names}") from e
            
        sorted_indices.extend([item['rel_idx'] for item in sorted_group])
            
    return filtered_regions, sorted_indices, keep_indices
