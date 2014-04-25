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
vector<int> ImgPoolLevels;
vector<int> ImgListsPoolIds;
vector<size_t> ImgParentsIdx;
vector<string> ImgLists;
//-- Dataset feature
Matrix<float> dataset_keypoint;         // all keypoint
Matrix<float> dataset_descriptor;   // all feature descriptor
vector<int> dataset_feature_count;      // number of features per dataset
//-- Dataset cluster
Matrix<float> cluster;
int actual_cluster_amount;
//-- Parallel clustering, clustering params
int PARALLEL_BLOCKS;
int PARALLEL_CPU;
//-- Dataset quantized result
vector< vector<int> > dataset_quantized_indices;
vector< vector<float> > dataset_quantized_dists;
//-- Bow
vector< vector<bow_bin_object> > bag_of_word; // dataset_id, list of feature in bow_bin

timespec startTime;

void LoadDataset(const string& ImgPath);
void SaveDatasetList(const string& out);
void LoadDatasetList(const string& in);
void ProcessDataset();
void ExtractDataset(bool save_feature);
void LoadFeature();
void ParallelClustering(int blocks);                                                        // [Master] Parallel clustering using MapReduce + Spawn style
    void FeatureMap(int blocks);                                                            // [Master] Map
        void SaveFeatureMap(const vector<size_t>& block_map, const string& out);            // [Master] Save map
    void ClusterReduce(int blocks);                                                         // [Master] Waiting for sub clusters
        void LoadSubCluster(int blocks);                                                    // [Master] Load sub cluster
void ClusteringJobsTracker(int blocks, int cpu);                                            // [Slave] Clustering Job tracker
    void SpawnClustering(const vector<int>& job_list);                                      // [Slave] Spawn cmd
        void SubClustering(const string& prefix, const string& input, const string& output);// [Slave] Sub-clustering
        void LoadFeatureMap(const string& in);                                              // [Slave] Load map
void Clustering(bool save_cluster, bool hdf5=true, bool runspawn=false, const string& out="");
void SaveCluster(const string& out);
void LoadCluster(const string& in);
void DatasetQuantization(bool save_feature);
void SaveQuantizedDataset(const string& out);
void LoadQuantizedDataset(const string& in);
void Bow(bool save_feature);
void SaveBow(const string& out);
void LoadBow(const string& in);
void build_invert_index();
//;)
