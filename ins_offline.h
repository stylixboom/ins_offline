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
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <tr1/unordered_map>
#include <bitset>
#include <math.h>

#include <flann/flann.hpp>
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"

#include "version.h"

using namespace std;
using namespace tr1;
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
vector<int> ImgListsPoolIds;            // To keep track of image_id to pool_id
vector<size_t> ImgParentsIdx;
vector<string> ImgLists;
//-- Dataset feature
Matrix<float> dataset_keypoint;         // all keypoint
Matrix<float> dataset_descriptor;       // all feature descriptor
vector<int> feature_count_per_pool;     // number of features per pool
vector<int> feature_count_per_image;     // number of features per image
//-- Dataset cluster
Matrix<float> cluster;
int actual_cluster_amount;
//-- Dataset quantized result
vector< vector<int> > dataset_quantized_indices;
vector< vector<float> > dataset_quantized_dists;
//-- Bow
vector< vector<bow_bin_object> > bag_of_word; // dataset_id, list of feature in bow_bin

timespec startTime;

void LoadDataset(const string& ImgPath);
void SaveDatasetList(const string& out);
void LoadDatasetList(const string& in);
void ProcessDataset();                      // Not necessary, this is for pre-processing image
void ExtractDataset(bool save_feature);
void PackFeature(bool by_block, size_t block_size);
void LoadFeature(size_t start_idx, size_t load_size, bool enable_kp = false);
void SavePoolinfo(const string& out);
void LoadPoolinfo(const string& in);
void Clustering(bool save_cluster, bool hdf5 = true);
void SaveCluster(const string& out);
void LoadCluster(const string& in);
void ImageFeaturesQuantization(bool save_quantized);
void SaveQuantizedDataset(const string& out);
void LoadQuantizedDataset(const string& in);
void Bow(bool save_feature);
void SaveBow(const string& out);
void LoadBow(const string& in);
void build_invert_index();
//;)
