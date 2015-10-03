/*
 * ins_offline.h
 *
 *  Created on: July 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#pragma once

using namespace ins;

// private variable
//-- ENV
//-- Param
ins_param run_param;
//-- Dataset list
deque<string> ParentPaths;
vector<int> Img2PoolIdx;                    // To keep track of image_id to pool_id
vector<size_t> Pool2ParentsIdx;
vector< pair<size_t, size_t> > Pool2ImagesIdxRange;    // To map between pool id and image ids [start, end]
vector<size_t> Img2ParentsIdx;
vector<string> ImgLists;
size_t total_features;
vector<int> feature_count_per_pool;         // number of features per pool
vector<int> feature_count_per_image;        // number of features per image
vector<int> image_count_per_pool;           // number of images per pool
//-- Quantized offset
bool dataset_quantized_offset_ready;
vector<size_t> dataset_quantized_offset;

timespec startTime;

void LoadDataset(const string& ImgPath);
void SaveDatasetList();
void LoadDatasetList();
void ProcessDataset();                      // Not necessary, this is for pre-processing image
void ExtractFeature(size_t block_size);
const int LOAD_DESC = 0, LOAD_KP = 1, LOAD_ALL = 2;
void LoadFeature(size_t start_idx, size_t load_size, int load_mode, Matrix<float>& load_data);
void SavePoolinfo(const string& out, const bool print = true);
void LoadPoolinfo(const string& in);
string SamplingDatabase(int sample_size);
string SamplingOnTheFly();
void Clustering(bool save_cluster, bool hdf5 = true);
void SaveCluster(const string& out);
void ClusteringCoarseLayer();
void ImageFeaturesQuantization(bool save_quantized);
void SaveQuantizedDataset(const vector<int>& quantized_counts, const vector<int*>& quantized_indices, const vector<float*>& quantized_dists, bool append);
void LoadQuantizedDatasetOffset();
void LoadSpecificQuantizedDataset(vector<int>& quantized_counts, vector<int*>& quantized_indices, vector<float*>& quantized_dists, size_t start_idx, size_t load_size = 1);  // Load quantized index from start_idx (image_id), load_size is total images to be loaded
void ReleaseQuantizedOffset();
void Bow(bool save_bow);
void build_invert_index();

void SiftCheck();
void SiftPackRepair();
void PoolCheck();
void PoolRepair();
void QuantizedCorrectnessCheck();
void BowCorrectnessCheck();
void InvDefCorrectnessCheck();
void PoolingTester();
void ExportImgList();
//;)
