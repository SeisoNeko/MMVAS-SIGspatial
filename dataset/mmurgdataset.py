import os
import torch
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from dataset.preprocesser import main as run_preprocessing

class CellDataset(Dataset):
    """Dataset for hexagonal cells, loading tabular features and satellite images."""

    def __init__(self, npz_data, sat_img_dir, transform=None):
        """Initializes the CellDataset.

        Args:
            npz_data (dict): Pre-processed tabular data containing feature arrays.
            sat_img_dir (str): Directory path containing the satellite images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.sat_img_dir = sat_img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.poi_feat = torch.tensor(npz_data['poi_feat'], dtype=torch.float32)
        self.lu_feat = torch.tensor(npz_data['lu_feat'], dtype=torch.float32)
        self.gn_feat = torch.tensor(npz_data['gn_feat'], dtype=torch.float32)
        self.farbac_feat = torch.tensor(npz_data['farbac_feat'], dtype=torch.float32)
        
        self.num_cells = self.poi_feat.shape[0]

    def __len__(self):
        """Returns the total number of cells in the dataset.

        Returns:
            int: Total number of cells.
        """
        return self.num_cells

    def __getitem__(self, idx):
        """Retrieves a single cell sample and its corresponding satellite image.

        Args:
            idx (int): Index of the cell to retrieve.

        Returns:
            dict: A dictionary containing the cell index, features, and image tensor.
        """
        img_path = os.path.join(self.sat_img_dir, f"{idx}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            sat_img = self.transform(img)
        else:
            sat_img = torch.zeros((3, 224, 224))

        return {
            "cell_idx": idx,
            "poi_feat": self.poi_feat[idx],
            "lu_feat": self.lu_feat[idx],
            "gn_feat": self.gn_feat[idx],
            "farbac_feat": self.farbac_feat[idx],
            "sat_image": sat_img
        }

class RegionDataset(Dataset):
    """Dataset for target regions, loading text sequences and street-view images."""

    def __init__(self, npz_data, sv_img_dir, transform=None):
        """Initializes the RegionDataset.

        Args:
            npz_data (dict): Pre-processed data containing region mappings and sequences.
            sv_img_dir (str): Directory path containing street-view images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.sv_img_dir = sv_img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.raw_tax_sequences = torch.tensor(npz_data['raw_tax_sequences'], dtype=torch.float32)
        self.grid_to_region_mapping = npz_data['grid_to_region_mapping']
        self.grid_labels = torch.tensor(npz_data['grid_labels'], dtype=torch.float32)
        self.num_regions = self.raw_tax_sequences.shape[0]

        self.text_input_ids = torch.tensor(npz_data.get('text_input_ids', np.zeros((self.num_regions, 512))), dtype=torch.long)
        self.text_attention_mask = torch.tensor(npz_data.get('text_attention_mask', np.zeros((self.num_regions, 512))), dtype=torch.long)
        
        self.seq_mean = self.raw_tax_sequences.mean(dim=(0, 1), keepdim=True)
        self.seq_std = self.raw_tax_sequences.std(dim=(0, 1), keepdim=True) + 1e-8
        self.raw_tax_sequences = (self.raw_tax_sequences - self.seq_mean) / self.seq_std
        
        self.label_mean = self.grid_labels.mean()
        self.label_std = self.grid_labels.std() + 1e-8
        self.grid_labels = (self.grid_labels - self.label_mean) / self.label_std

    def __len__(self):
        """Returns the total number of regions in the dataset.

        Returns:
            int: Total number of regions.
        """
        return self.num_regions

    def __getitem__(self, idx):
        """Retrieves a single region sample and its corresponding street-view images.

        Args:
            idx (int): Index of the region to retrieve.

        Returns:
            dict: A dictionary containing the region data, images, and nested grid values.
        """
        sv_images = []
        img_dir = os.path.join(self.sv_img_dir, f"grid_{idx}")
        if not os.path.exists(img_dir):
            sv_images = [torch.zeros((3, 224, 224)) for _ in range(5)]
        else:
            for image in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, image)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    sv_images.append(self.transform(img))
                else:
                    sv_images.append(torch.zeros((3, 224, 224)))
            
        sv_images_tensor = torch.stack(sv_images)

        grids_in_region = np.where(self.grid_to_region_mapping == idx)[0]
        grid_indices = torch.tensor(grids_in_region, dtype=torch.long)
        grid_values = self.grid_labels[grids_in_region]
        region_avg = grid_values.mean() if len(grid_values) > 0 else torch.tensor(0.0)

        return {
            "region_idx": idx,
            "raw_tax_sequence": self.raw_tax_sequences[idx],
            "text_input_ids": self.text_input_ids[idx],
            "text_attention_mask": self.text_attention_mask[idx],
            "street_view_images": sv_images_tensor,
            "grid_indices_in_region": grid_indices,
            "grid_land_values": grid_values,
            "region_avg_value": region_avg
        }

def collate_fn(batch):
    """Custom collate function to handle variable-length grid predictions in batches.

    Args:
        batch (list): A list of data dictionaries returned by RegionDataset.__getitem__.

    Returns:
        dict: A dictionary of batched tensors, flattening variable-length arrays.
    """
    region_idx = torch.tensor([b["region_idx"] for b in batch], dtype=torch.long)
    raw_tax_sequence = torch.stack([b["raw_tax_sequence"] for b in batch])
    text_input_ids = torch.stack([b["text_input_ids"] for b in batch])
    text_attention_mask = torch.stack([b["text_attention_mask"] for b in batch])
    street_view_images = torch.stack([b["street_view_images"] for b in batch])
    region_avg_value = torch.stack([b["region_avg_value"] for b in batch])
    
    grid_indices_in_region = torch.cat([b["grid_indices_in_region"] for b in batch])
    grid_land_values = torch.cat([b["grid_land_values"] for b in batch])

    grid_to_batch_idx = []
    for i, b in enumerate(batch):
        grid_to_batch_idx.extend([i] * len(b["grid_indices_in_region"]))
    grid_to_batch_idx = torch.tensor(grid_to_batch_idx, dtype=torch.long)
    
    return {
        "region_idx": region_idx,
        "raw_tax_sequence": raw_tax_sequence,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "street_view_images": street_view_images,
        "grid_indices_in_region": grid_indices_in_region,
        "grid_land_values": grid_land_values,
        "region_avg_value": region_avg_value,
        "grid_to_batch_idx": grid_to_batch_idx
    }

class MMURGDataContainer:
    """Data Container for managing spatial datasets, auto-caching, and lazy loading."""

    def __init__(self, config):
        """Initializes the MMURGDataContainer, handling caching and dataset creation.

        Args:
            config (dict): Configuration dictionary containing dataset parameters and paths.
        """
        npz_path = config["paths"]["processed_npz"]
        force_preprocess = config["pipeline"]["refresh_data"]
        grid_type = config["dataset"]["grid_type"]
        raw_data_dir = config["paths"]["raw_data_dir"]
        sat_dir = raw_data_dir + config["paths"]["sat_images"]
        sv_dir = raw_data_dir + config["paths"]["sv_images"]
        cells_gdf_path = raw_data_dir + config["paths"]["grid_100m"]
        regions_gdf_path = raw_data_dir + config["paths"]["region"]
        
        if force_preprocess or not os.path.exists(npz_path):
            print(f"Auto-preprocessing triggered. Processing raw spatial data...")
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            
            dataset_dict = run_preprocessing(
                grid_type=config["dataset"]["grid_type"],
                knn_k=config["dataset"]["knn_k"],
                poi_min_occurrences=config["dataset"]["poi_min_occurrences"],
                tax_seq_len=config["dataset"]["tax_seq_len"],
                tax_raw_dim=config["dataset"]["tax_raw_dim"],
            )
            
            np.savez_compressed(npz_path, **dataset_dict)
            print(f"Preprocessing complete and cached at {npz_path}.")
            
            npz_data = dataset_dict 
        else:
            print(f"Loading cached tabular dataset from {npz_path}...")
            npz_data = dict(np.load(npz_path, allow_pickle=True))
            
        self.cell_dataset = CellDataset(npz_data, sat_dir)
        self.region_dataset = RegionDataset(npz_data, sv_dir)
        
        self.grid_to_region_mapping = torch.tensor(npz_data['grid_to_region_mapping'], dtype=torch.long)
        self.grid_labels = torch.tensor(npz_data['grid_labels'], dtype=torch.float32)

        self.cells_gdf = gpd.read_file(cells_gdf_path, layer="grid" if grid_type == "1km" else "clipped")
        if self.cells_gdf.crs is None or self.cells_gdf.crs.to_epsg() != 3826:
            self.cells_gdf = self.cells_gdf.to_crs(epsg=3826)
        if 'id' in self.cells_gdf.columns:
            self.cells_gdf = self.cells_gdf.sort_values('id').reset_index(drop=True)
        else:
            self.cells_gdf = self.cells_gdf.reset_index(drop=True)
        self.cells_gdf['node_id'] = self.cells_gdf.index
        
        self.regions_gdf = gpd.read_file(regions_gdf_path)
        if self.regions_gdf.crs is None or self.regions_gdf.crs.to_epsg() != 3826:
            self.regions_gdf = self.regions_gdf.to_crs(epsg=3826)
        self.regions_gdf['region_idx'] = range(len(self.regions_gdf))

        self.poi_edge_index = torch.tensor(npz_data['poi_edge_index'], dtype=torch.long)
        self.poi_edge_weights = torch.tensor(npz_data['poi_edge_weights'], dtype=torch.float32)
        
        self.lu_edge_index = torch.tensor(npz_data['lu_edge_index'], dtype=torch.long)
        self.lu_edge_weights = torch.tensor(npz_data['lu_edge_weights'], dtype=torch.float32)
        
        self.gn_edge_index = torch.tensor(npz_data['gn_edge_index'], dtype=torch.long)
        self.farbac_edge_index = torch.tensor(npz_data['farbac_edge_index'], dtype=torch.long)
        self.farbac_edge_weights = torch.tensor(npz_data['farbac_edge_weights'], dtype=torch.float32)