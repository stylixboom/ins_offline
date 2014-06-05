/*
 * bow_offset_maker
 *
 *  Created on: November 29, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#include <bitset>
#include <vector>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <algorithm> // sort
#include <cstdlib> // exit

#include "../lib/alphautils/alphautils.h"
#include "../lib/ins/invert_index.h"

using namespace std;
using namespace alphautils;
using namespace ins;

int main(int argc,char *argv[])
{
    if (argc != 2) // Explicitly run
    {
        cout << "Please specify the absolute path of bow file" << endl;
        exit(-1);
    }

    // Get path
    string in = string(argv[1]);
    //string in = "/home/stylix/webstylix/code/database/ins_offline/oxbuildings5k_sifthesaff_norm_akm_1000000_kd3_16/bow";

    // Extract offset
    ifstream InFile (in.c_str(), ios::binary);
    if (InFile)
    {
        // Dataset size
        size_t dataset_count;
        InFile.read((char*)(&dataset_count), sizeof(dataset_count));

        // Prepare offset array
        vector<size_t> xct_offset;

        // Curr Offset
        // Start after read dataset_count
        size_t curr_offset = sizeof(dataset_count);

        // Bow hist
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Keep offset
            xct_offset.push_back(curr_offset);

            // Dataset ID (read, but not use)
            size_t dataset_id_read;
            InFile.read((char*)(&dataset_id_read), sizeof(dataset_id_read));
            curr_offset += sizeof(dataset_id_read);

            // Dataset bow
            //vector<bow_bin_object> read_bow;

            // Non-zero count
            size_t bin_count;
            InFile.read((char*)(&bin_count), sizeof(bin_count));
            curr_offset += sizeof(bin_count);

            // ClusterID and FeatureIDs
            for (size_t bin_idx = 0; bin_idx < bin_count; bin_idx++)
            {
                bow_bin_object read_bin;

                // Cluster ID
                InFile.read((char*)(&(read_bin.cluster_id)), sizeof(read_bin.cluster_id));
                curr_offset += sizeof(read_bin.cluster_id);

                // Weight
                InFile.read((char*)(&(read_bin.weight)), sizeof(read_bin.weight));
                curr_offset += sizeof(read_bin.weight);

                // Feature count
                size_t feature_count;
                InFile.read((char*)(&feature_count), sizeof(feature_count));
                curr_offset += sizeof(feature_count);
                for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
                {
                    feature_object feature;

                    // Feature ID
                    InFile.read((char*)(&(feature.feature_id)), sizeof(feature.feature_id));
                    curr_offset += sizeof(feature.feature_id);
                    // x
                    InFile.read((char*)(&(feature.x)), sizeof(feature.x));
                    curr_offset += sizeof(feature.x);
                    // y
                    InFile.read((char*)(&(feature.y)), sizeof(feature.y));
                    curr_offset += sizeof(feature.y);
                    // a
                    InFile.read((char*)(&(feature.a)), sizeof(feature.a));
                    curr_offset += sizeof(feature.a);
                    // b
                    InFile.read((char*)(&(feature.b)), sizeof(feature.b));
                    curr_offset += sizeof(feature.b);
                    // c
                    InFile.read((char*)(&(feature.c)), sizeof(feature.c));
                    curr_offset += sizeof(feature.c);

                    //read_bin.features.push_back(feature);
                }

                // Keep bow
                //read_bow.push_back(read_bin);
            }

            // Keep hist
            //bag_of_word.push_back(read_bow);
        }

        // Total offset
        for (size_t offset_id = 0; offset_id < dataset_count; offset_id++)
            cout << xct_offset[offset_id] << " " << endl;

        bin_write_vector_SIZET(in + "_offset", xct_offset);

        // Close file
        InFile.close();
    }

    return 0;
}
