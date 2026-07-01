# MMVAS-Net: Multi-Modal Urban Region Graph Network

**MMVAS-Net** is a spatial geographic deep learning architecture designed to predict continuous street-level land values and tax rates at a micro-grid level (100m hexes). The model operates on a dual-pathway system, explicitly separating macroscopic urban baseline values from microscopic commercial and structural outliers.

## Architecture & Data Pipeline

The core philosophy of MMVAS-Net is the residual prediction equation: 

$$Prediction = Region\_Base + Grid\_Residual$$

To achieve this without suffering from graph over-smoothing or gradient hijacking, the pipeline is split into four distinct phases:

### Phase 0: Offline Feature Extraction
To prevent VRAM bottlenecks, static heavy features are cached offline.
* **Text Embeddings:** LLM embeddings (BERT-base-chinese) are extracted for regional semantics.
* **Satellite Imagery:** Extracted via a frozen ResNet50.

### Phase 1: GridLearner (Self-Supervised GNN)
* Learns representations for individual 100m micro-grids by integrating 4 distinct topologies: **Physical Neighbors, POI, Land Use, and FAR/BAC**.
* Operates on a highly dense $k=18$ graph, utilizing `k_hop_subgraph` extraction with ID relabeling to maintain localized gradient tracing and prevent city-wide CUDA memory explosions.

### Phase 2: AdaRegionGen
* A deterministic geometric projection algorithm that clusters the micro-grids into logical, macro-level urban regions.

### Phase 3: Downstream Task Model (Dual-Pathway)
This phase integrates the final predictions using two parallel modules:
1. **RegionEnricher (Macro-Pathway):** Processes live Street View Images (SVI) using a frozen ResNet50. Images are processed using an automated micro-batching loop (`chunk_size=8`) to prevent cuDNN workspace fragmentation.
2. **LocalGridPredictor (Micro-Pathway):** Protects unique micro-grid identities (like high-value commercial POIs) by utilizing **Feature Concatenation** . It explicitly staples the grid's local features to its parent region's baseline, passing the combined vector through an MLP to calculate the specific residual. 



## Project Structure

```text
├── config.yaml                 # Centralized hyperparameters and path configurations
├── train.py                    # Main training script (Phases 1-4 execution)
├── data/                       # All the raw data you should place
├── dataset/
│   ├── mmvasdataset.py         # DataContainer, graph definitions, and collate functions
│   ├── preprocesser.py         # The process code to convert raw data into npz file if you use refresh_data in config.yaml
├── models/
│   ├── GridLearner.py          # Phase 1: Multi-view GNN
│   ├── AdaRegionGen.py         # Phase 2: Region clustering
│   ├── DownStreamTask.py       # Phase 3: Core dual-pathway model integration
│   ├── RegionEnricher.py       # Macro SVI extraction (ResNet50)
│   ├── LocalGridPredictor.py   # Micro residual prediction (Concatenation + MLP)
│   └── loss.py                 # FusionSimilarityLoss and CustomMSELoss
├── tuning_history.db           # SQLite database for Optuna hyperparameter states
├── result/                     # Output directory for test_predictions.csv and .gpkg
└── utils/                      # Some helping function code
    ├── result_analyze.py       # Analyze the prediction for the test set in the last training
    └── inference.py            # Inference the whole Tainan city grids, not sure if it can use now.
```


## Installation & Environment Setup
It is highly recommended to use a Linux environment with uv for fast Python dependency resolution.

```bash
# 1. Create and activate the virtual environment
uv sync
source .venv/bin/activate

# 2. Install PyTorch with CUDA support if your torch can't use Cuda(Adjust URL based on your specific driver) 
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 3. Re-Install PyTorch Geometric and its dependencies if you do step 2.
uv pip install torch_geometric
```

## Usage

### 1. Configuration
Ensure your ```config.yaml``` is correctly mapped to your local dataset directories (especially the Street View Images and Satellite directories). Standard hyperparameters are already optimized for a $k=18$ dense urban graph (e.g., Tainan City):
* ```lambda```: 15.0 (Forces the optimizer to respect the micro-grid residuals).
* ```lr```: 0.0003 (Stabilizes the dual-pathway training).
* ```batch_size```: 16 (Regions per batch).

### 2. Training the Model
Run the primary training pipeline. The script will automatically detect existing Phase 0/1 caches and skip to the active training phases.
```bash
python train.py
```
If you find youself GPU driver kernel panic, use
```bash
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python train.py
```
to track the error

### 3. Visualizing Results
Upon successful completion of Phase 4, the model will evaluate the test set and output two files in the ```/result``` directory:
* ```test_predictions.csv```: Raw predictive metrics and node mappings.
* ```test_predictions.gpkg```: A spatial GeoPackage ready to be directly imported into QGIS for geographic visualization of the grid residuals against the region baselines.


### How to prepare data
1. put ```cache/``` and ```processed_dataset.npz```into ```dataset/```
2. unzip ```100m_street.zip```, you should find a directory name ```100m_street_images/```
3. put that directory into ```data/```
4. strat train

### How to ablation
1. Go to ```models/GridLearner.py```
```
Z_stacked = torch.stack([z_poi, z_lu, z_gn, z_farbac, z_sat], dim=1)
```
Remove the wanted feature and set
```
self.da_fusion = DAFusion(embed_dim=embed_dim, num_views=5)
```
to
```
self.da_fusion = DAFusion(embed_dim=embed_dim, num_views=4)
```
2. Go to ```config.yaml``` and set ```p1_skip``` to ```false```
3. Run ```python train.py```
4. After one times of train, you will get ```weights/phase1_embeddings.pt```, that is you grid learner result
5. Go to ```config.yaml``` and set ```p1_skip``` to ```true```
6. run with different seed in ```config.yaml```