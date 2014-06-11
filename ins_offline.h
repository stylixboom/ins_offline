/*
 * ins_offline.h
 *
 *  Created on: July 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#pragma once
#include <sys/stat.h>   // file-directory existing
#include <sys/types.h>  // file-directory
#include <dirent.h>     // file-directory
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <bitset>
#include <cmath>

#include <flann/flann.hpp>
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"
#include "../lib/ins/bow.h"
#include "../lib/ins/quantizer.h"

#include "version.h"

using namespace std;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::hdf5io;
using namespace ins;

// private variable
//-- ENV
//-- Param
ins_param run_param;
//-- Dataset list
vector<string> ImgParentPaths;
vector<int> ImgListsPoolIds;                // To keep track of image_id to pool_id
vector<size_t> ImgParentsIdx;
vector<string> ImgLists;
size_t total_features;
vector<int> feature_count_per_pool;         // number of features per pool
vector<int> feature_count_per_image;        // number of features per image
//-- Quantized offset
bool dataset_quantized_offset_ready;
vector<size_t> dataset_quantized_offset;

timespec startTime;

void LoadDataset(const string& ImgPath);
void SaveDatasetList();
void LoadDatasetList();
void ProcessDataset();                      // Not necessary, this is for pre-processing image
void ExtractDataset(bool save_feature);
void PackFeature(bool by_block, size_t block_size);
const int LOAD_DESC = 0, LOAD_KP = 1, LOAD_ALL = 2;
void LoadFeature(size_t start_idx, size_t load_size, int load_mode, Matrix<float>& load_data);
void SavePoolinfo(const string& out);
void LoadPoolinfo(const string& in);
string SamplingDatabase(int sample_size, int dimension);
void Clustering(bool save_cluster, bool hdf5 = true);
void SaveCluster(const string& out);
void ImageFeaturesQuantization(bool save_quantized);
void SaveQuantizedDataset(const vector<int>& quantized_counts, const vector<int*>& quantized_indices, const vector<float*>& quantized_dists, bool append);
void LoadQuantizedDatasetOffset();
void LoadSpecificQuantizedDataset(vector<int>& quantized_counts, vector<int*>& quantized_indices, vector<float*>& quantized_dists, size_t start_idx, size_t load_size = 1);  // Load quantized index from start_idx (image_id), load_size is total images to be loaded
void ReleaseQuantizedOffset();
void Bow(bool save_bow);
void build_invert_index();
//;)
