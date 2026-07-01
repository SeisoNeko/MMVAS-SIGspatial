import numpy as np
import pandas as pd
import geopandas as gpd
import os

DATA_DIR = "./data/old/POIs"
FIRST_POI_DATA_DIR = DATA_DIR + "/first_phase_poi_data"
GOOGLE_POI_DATA_DIR = DATA_DIR + "/google_map"
OPENSTREETMAP_POI_DATA_DIR = DATA_DIR + "/OSM"

OUTPUT_CSV = "./dataset/processed_poi_data.csv"

def load_first_poi_data():
    """Loads and processes the first phase Point of Interest (POI) dataset.

    Iterates through the designated directory, reads CSV files containing 
    POI coordinates, standardizes column names, and extracts categorization 
    metadata from the filenames.

    Returns:
        pd.DataFrame: A concatenated dataframe containing latitude, longitude, 
            POI type, and category for all successfully processed records.
    """
    print("Loading and processing first POI dataset...")
    df_list = [] 
    for filename in os.listdir(FIRST_POI_DATA_DIR):
        if filename.endswith(".csv"):
            clean_name = filename.replace(".csv", "")
            
            parts = clean_name.split("_")
            poi_type = parts[0]
            category_name = parts[1] if len(parts) > 1 else "Unknown"

            filepath = os.path.join(FIRST_POI_DATA_DIR, filename)
            try:
                fileContent = pd.read_csv(filepath, encoding="utf-8")
            except UnicodeDecodeError:
                fileContent = pd.read_csv(filepath, encoding="Big5")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            cols_to_keep = ["lat", "lng"]
            if "xcoord" in fileContent.columns and "ycoord" in fileContent.columns:
                fileContent = fileContent.rename(columns={"xcoord": "lng", "ycoord": "lat"})
                cols_to_keep = ["lat", "lng"]

            try:
                temp_df = fileContent[cols_to_keep].copy().replace('"', '').replace("'", "")
            except KeyError:
                print(f"Error: Required columns not found in {filename}")
                continue
            
            temp_df = temp_df.dropna(subset=cols_to_keep)

            temp_df["poi_type"] = poi_type
            temp_df["category"] = category_name

            df_list.append(temp_df)
            
    if not df_list:
        print("Warning: No CSV files found!")
        return pd.DataFrame()
        
    final_poi_df = pd.concat(df_list, ignore_index=True)
    
    return final_poi_df

def load_google_poi_data():
    """Loads and processes the Google Maps POI dataset.

    Reads coordinate data from CSV files and infers the POI type based 
    on the filename structure. 

    Returns:
        pd.DataFrame: A concatenated dataframe containing the compiled Google 
            Maps POI records.
    """
    print("Loading and processing Google Maps POI dataset...")
    df_list = []
    for filename in os.listdir(GOOGLE_POI_DATA_DIR):
        if filename.endswith(".csv"):
            clean_name = filename.replace(".csv", "")
            poitype = " ".join(clean_name.split("_")[1:])
            filepath = os.path.join(GOOGLE_POI_DATA_DIR, filename)

            try:
                fileContent = pd.read_csv(filepath, encoding="utf-8")
            except UnicodeDecodeError:
                fileContent = pd.read_csv(filepath, encoding="Big5")
            except Exception as e:                
                print(f"Error reading {filename}: {e}")
                continue
            
            cols_to_keep = ["lat", "lng"]
            try:
                temp_df = fileContent[cols_to_keep]
            except KeyError:
                print(f"Error: Required columns not found in {filename}")
                continue
                
            temp_df["poi_type"] = poitype
            temp_df["category"] = ''
            df_list.append(temp_df)

    if not df_list:
        print("Warning: No CSV files found!")
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)

def load_osm_poi_data():
    """Loads and processes the OpenStreetMap (OSM) POI dataset.

    Parses GeoJSON files, aligns the Coordinate Reference System (CRS) to EPSG:3826, 
    and extracts centroid coordinates for bounding geometries.

    Returns:
        pd.DataFrame: A concatenated dataframe containing the compiled OSM 
            POI records.
    """
    print("Loading and processing OpenStreetMap POI dataset...")
    df_list = []
    
    for filename in os.listdir(OPENSTREETMAP_POI_DATA_DIR):
        if filename.endswith(".geojson") and filename.split("_")[0] == "amenity":
            filepath = os.path.join(OPENSTREETMAP_POI_DATA_DIR, filename)
            df = gpd.read_file(filepath)

            if df.empty:
                continue

            if df.crs is None or df.crs.to_epsg() != 3826:
                df = df.to_crs(epsg=3826)

            if df.geometry.type == 'Point':
                centroids = df.geometry
            else:
                centroids = df.geometry.centroid

            if 'name' in df.columns:
                poi_names = df['name'].fillna("Unknown")
            else:
                poi_names = "Unknown"

            temp_df = pd.DataFrame({
                "lat": centroids.x,          
                "lng": centroids.y,          
                "poi_type": poi_names,  
                "category": ''
            })
            
            df_list.append(temp_df)
            
    if not df_list:
        print("Warning: No GeoJSON files found!")
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)


def main():
    """Executes the complete POI data preprocessing pipeline.

    Loads diverse POI datasets, concatenates them into a unified structure, 
    and exports the final dataset to a designated CSV file.
    """
    first_poi_df = load_first_poi_data()
    google_poi_df = load_google_poi_data()
    # osm_poi_df = load_osm_poi_data()

    processed_poi_df = pd.concat([first_poi_df, google_poi_df], ignore_index=True)

    processed_poi_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Processed POI data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()