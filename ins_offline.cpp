/*
 * ins_offline.cpp
 *
 *  Created on: July 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */
#include <sys/stat.h>   // file-directory existing
#include <sys/types.h>  // file-directory
#include <dirent.h>     // file-directory
#include <unistd.h>     // sysconf, usleep
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <bitset>
#include <cmath>
#include <omp.h>

#include <flann/flann.hpp>

// Siriwat's header
#include "../lib/alphautils/alphautils.h"
#include "../lib/alphautils/hdf5_io.h"
#include "../lib/sifthesaff/SIFThesaff.h"
#include "../lib/ins/ins_param.h"
#include "../lib/ins/invert_index.h"
#include "../lib/ins/bow.h"
#include "../lib/ins/quantizer.h"

#include "ins_offline.h"

#include "version.h"

using namespace std;
using namespace ::flann;
using namespace alphautils;
using namespace alphautils::hdf5io;
using namespace ins;


int main(int argc,char *argv[])
{
    if (argc > 1) // Explicitly run
    {
        if (argc%2 == 0)
        {
            cout << "Command is invalid! It must be in a form of key value pair." << endl;
            //exit(1);
        }

        // Grab all params
        unordered_map<string, string> opts;
        for (int param_idx = 1; param_idx < argc; param_idx+=2)
            opts[string(argv[param_idx])] = string(argv[param_idx + 1]);

        // Explicitly run some process here
        //...
    }
    else // Main menu
    {
        char menu;

        do
        {
            cout << endl;
            cout << "======== Instance Search - Offline (" << ins_offline_AutoVersion::ins_offline_FULLVERSION_STRING << ") ========" << endl;
            cout << "[l] Load dataset" << endl;
            cout << "[k] Preprocess dataset" << endl;
            cout << "[e] Extracting feature (with save)" << endl;
            cout << "[p] Packing extracted features" << endl;
            cout << "[j] Sampling feature on-the-fly" << endl;
            cout << "[c] Clustering feature" << endl;
            cout << "[u] Clustering coarse layer" << endl;
            cout << "[v] Vector quantization" << endl;
            cout << "[b] Bag of word building" << endl;
            cout << "[i] Building invert index and save" << endl;
            cout << "[s] SIFT feature corrupt check" << endl;
            cout << "[r] SIFT Pack repair" << endl;
            cout << "[w] Pool checking" << endl;
            cout << "[x] Quantize upgrade check" << endl;
            cout << "[y] Bow upgrade check" << endl;
            cout << "[z] Inverted index definition check" << endl;
            cout << "[o] Bow Pooling Tester" << endl;
            cout << "[t] Image list exporter" << endl;
            cout << "Enter menu: ";
            cin >> menu;

            bool save_feature = true;
            bool save_cluster = true;
            bool save_quantized = true;
            bool save_bow = true;
            size_t block_size = 100;
            switch (menu)
            {
            case 'l':
                run_param.LoadPreset();
                LoadDataset(run_param.path_from_dataset);
                break;
            case 'k':
                ProcessDataset();
                break;
            case 'e':
                //char save_feature_choice = 'n';
                ExtractDataset(save_feature);
                break;
            case 'p':
                cout << "Please specify total images per block (as buffer) : "; cout.flush();
                cin >> block_size;
                PackFeature(block_size);
                break;
            case 'j':
                SamplingOnTheFly();
                break;
            case 'c':
                Clustering(save_cluster);
                break;
            case 'u':
                ClusteringCoarseLayer();
                break;
            case 'v':
                ImageFeaturesQuantization(save_quantized);
                break;
            case 'b':
                Bow(save_bow);
                break;
            case 'i':
                build_invert_index();
                break;
            case 's':
                SiftCheck();
                break;
            case 'r':
                SiftPackRepair();
                break;
            case 'w':
                PoolCheck();
                break;
            case 'x':
                QuantizedCorrectnessCheck();
                break;
            case 'y':
                BowCorrectnessCheck();
                break;
            case 'z':
                InvDefCorrectnessCheck();
                break;
            case 'o':
                PoolingTester();
                break;
            case 't':
                ExportImgList();
                break;
            }

        } while (menu != 'q');
    }
/*
    // Accessing data in matrix
    float* clusters_idx = clusters.ptr();

    for(int row = 0; row < cluster_amount; row++)
    {
        for(size_t col = 0; col < clusters.cols; col++)
        {
            cout << clusters_idx[row * clusters.cols + col] << " ";
            if(col == clusters.cols - 1)
                cout << endl;
        }
    }
*/

    //delete[] cluster.ptr();
    //delete[]

    return 0;
}

void LoadDataset(const string& DatasetPath)
{
    if (!is_path_exist(run_param.dataset_basepath_path))
    {
        stringstream dataset_path;
        dataset_path << run_param.dataset_root_dir << "/" << DatasetPath;
        cout << "dataset path: " << dataset_path.str() << endl;
        // Checking path
        if (!is_path_exist(dataset_path.str()))
        {
            cout << "Path not exist \"" << dataset_path.str() << "\"" << endl;
            return;
        }

        // Load specific set, if available
        bool with_set = is_path_exist(run_param.offline_working_path + "/using_poolname.txt");
        set<string> defined_set;
        if (with_set)
        {
            cout << "Loading specific pool list.."; cout.flush();
            defined_set = text_readline2set(run_param.offline_working_path + "/using_poolname.txt");
            cout << "done!" << endl;
        }

        // ======== Load dataset ========
        cout << "Load dataset..";
        cout.flush();
        startTime = CurrentPreciseTime();

        vector<string> ImgDirQueue;

        char currDir[] = ".";
        char parentDir[] = "..";
        size_t currParentIdx = ParentPaths.size();
        int pool_id = -1;

        // First parent dir
        deque<int> ImgPoolLevels;
        ParentPaths.push_back(DatasetPath);
        ImgPoolLevels.push_back(0);

        while (currParentIdx < ParentPaths.size())
        {
            stringstream CurrDirectory;
            CurrDirectory << run_param.dataset_root_dir << "/" << ParentPaths[currParentIdx];

            // Directory traverse
            //cout << "dir: " << CurrDirectory.str() << endl;
            DIR* dirp = opendir(CurrDirectory.str().c_str());
            dirent* dp;
            map<string, string> sorted_file_name;
            while ((dp = readdir(dirp)) != NULL)
            {
                string file_name = string(dp->d_name);
                sorted_file_name[file_name] = file_name;
            }
            closedir(dirp);

            // New temporary empty subdir
            bool first_file = false;
            vector<string> new_ParentPaths;
            vector<size_t> new_ImgPoolLevels;
            for (auto sorted_file_name_it = sorted_file_name.begin(); sorted_file_name_it != sorted_file_name.end(); sorted_file_name_it++)
            {
                string file_name = sorted_file_name_it->second;
                cout << "checking.." << file_name << string(20, ' ') << string(file_name.length() + 30, '\b');
                struct stat st;
                stat((CurrDirectory.str() + "/" + file_name).c_str(), &st);
                if (S_ISREG(st.st_mode))    // Check regular file,
                {
                    string file_ext = file_name.substr(file_name.rfind('.') + 1);
                    if ((!str_contains(run_param.dataset_prefix, "ins2013") && file_ext == "jpg") ||
                        ((str_contains(run_param.dataset_prefix, "chars74k") || str_contains(run_param.dataset_prefix, "ins2013")) && file_ext == "png"))
                    {
                        //cout << " - file - ext " << file_ext << endl;
                        ImgLists.push_back(file_name);
                        Img2ParentsIdx.push_back(currParentIdx); // lookup index to same parent

                        // Group dataset into pool_id according to the same pool level
                        if (run_param.group_level != -1 && ImgPoolLevels[currParentIdx] >= run_param.group_level)
                        {
                            if (!first_file && ImgPoolLevels[currParentIdx] == run_param.group_level) // current pool level is the same as group level
                            {
                                first_file = true;
                                Img2PoolIdx.push_back(++pool_id);

                                // Add map from pool_id to parent_id
                                Pool2ParentsIdx.push_back(currParentIdx);
                            }
                            else
                                Img2PoolIdx.push_back(pool_id);
                        }
                        else // In case image is outside pool level
                        {
                            // If group pool level is -1, all the rest images will be in different pool level
                            if (run_param.group_level == -1)
                            {
                                Img2PoolIdx.push_back(++pool_id);

                                // Add map from pool_id to parent_id
                                Pool2ParentsIdx.push_back(currParentIdx);
                            }
                            else // if image is outside its pool level, just skip by -1
                                Img2PoolIdx.push_back(-1);
                        }
                    }
                    //else
                    //    cout << " - skip" << endl;
                }
                else if(S_ISDIR(st.st_mode))    // Check directory
                {
                    // Filter only child directory
                    if(strcmp(file_name.c_str(), currDir) && strcmp(file_name.c_str(), parentDir))    // strcmp equal is 1
                    {

                        /// Skip. if not appear in the specified set
                        if (with_set && defined_set.find(file_name) == defined_set.end())
                            continue;

                        //cout << " - dir" << endl;
                        // Add sub-dir
                        stringstream SubDir;
                        SubDir << ParentPaths[currParentIdx] << "/" << file_name;
                        new_ParentPaths.push_back(SubDir.str());
                        new_ImgPoolLevels.push_back(ImgPoolLevels[currParentIdx] + 1); // next level
                    }
                    //else
                    //    cout << " - parent" << endl;
                }
            }

            // Insert subdir at under its parent
            if (new_ParentPaths.size() > 0 && new_ImgPoolLevels.size() > 0)
            {
                ParentPaths.insert(ParentPaths.begin() + currParentIdx + 1, new_ParentPaths.begin(), new_ParentPaths.end());
                ImgPoolLevels.insert(ImgPoolLevels.begin() + currParentIdx + 1, new_ImgPoolLevels.begin(), new_ImgPoolLevels.end());

            }

            // Release mem
            vector<string>().swap(new_ParentPaths);
            vector<size_t>().swap(new_ImgPoolLevels);
            map<string, string>().swap(sorted_file_name);

            // Next parent
            currParentIdx++;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        cout << "Save dataset list..";
        cout.flush();
        startTime = CurrentPreciseTime();
        make_dir_available(run_param.offline_working_path); // mkdir for this working paht
        SaveDatasetList();
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
    {
        cout << "Load dataset list..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadDatasetList();
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }

    // Checking image avalibality
    if (ImgLists.size() > 0)
    {
        cout << "== Dataset information ==" << endl;
        if (!is_path_exist(run_param.poolinfo_path))
        {
            // Check dataset info from actual dataset
            cout << "Total pool: " << Img2PoolIdx[Img2PoolIdx.size() - 1] + 1 << endl;
            cout << "Total image: " << ImgLists.size() << endl;
        }
        else
        {
            // Check dataset info from pool
            LoadPoolinfo(run_param.poolinfo_path);
            cout << "Total pool: " << feature_count_per_pool.size() << endl;
            cout << "Total image: " << feature_count_per_image.size() << endl;
            cout << "Total features: " << total_features << endl;

            //cout << "Total pool2: " << image_count_per_pool.size() << endl;
            //cout << "Total pool3: " << Pool2ParentsIdx.size() << endl;
        }
    }
    else
        cout << "No image available" << endl;
}

void SaveDatasetList()
{
    // Write parent path (dataset based path)
    ofstream OutParentFile (run_param.dataset_basepath_path.c_str());
    if (OutParentFile.is_open())
    {
        for (size_t parent_id = 0; parent_id < ParentPaths.size(); parent_id++)
            OutParentFile << parent_id << ":" << ParentPaths[parent_id] << endl;
            // parent_id:parent_path

        // Close file
        OutParentFile.close();
    }

    // Write image filename
    ofstream OutImgFile (run_param.dataset_filename_path.c_str()); // .dataset_file
    if (OutImgFile.is_open())
    {
        // Write path to image
        // img-parent:pool-parent:img-pool:path_to_image
        for (size_t image_id = 0; image_id < ImgLists.size(); image_id++)
            OutImgFile << Img2ParentsIdx[image_id] << ":" << Pool2ParentsIdx[Img2PoolIdx[image_id]] << ":" << Img2PoolIdx[image_id] << ":" << ImgLists[image_id] << endl;
            // parent_id:image_name

        // Close file
        OutImgFile.close();
    }
}

void LoadDatasetList()
{
    // Read parent path (dataset based path)
    ifstream InParentFile (run_param.dataset_basepath_path.c_str());
    if (InParentFile)
    {
        string read_line;
        while (!InParentFile.eof())
        {
            getline(InParentFile, read_line);
            if (read_line != "")
            {
                vector<string> split_line;
                // parent_id:parent_path
                StringExplode(read_line, ":", split_line);

                ParentPaths.push_back(split_line[1]);
            }
        }

        // Close file
        InParentFile.close();
    }

    // Read image filename
    ifstream InImgFile (run_param.dataset_filename_path.c_str());
    if (InImgFile)
    {
        string read_line;
        unordered_set<int> pool_set;
        while (!InImgFile.eof())
        {
            getline(InImgFile, read_line);
            if (read_line != "")
            {
                // pool_id:parent_id:image_name

                // Find first and second pos of ":", the rest is an image name
                size_t cpos_start = 0;
                size_t cpos_end = 0;
                size_t line_size = read_line.length();
                bool done_parent_id = false;
                bool done_pool2parent_id = false;
                for (cpos_end = 0; cpos_end < line_size; cpos_end++)
                {
                    if (read_line[cpos_end] == ':')
                    {
                        int idx = atoi(read_line.substr(cpos_start, cpos_end - cpos_start).c_str());
                        cpos_start = cpos_end + 1;
                        if (!done_parent_id)                // Parent id
                        {
                            Img2ParentsIdx.push_back(idx);
                            done_parent_id = true;
                        }
                        else if (!done_pool2parent_id)      // Pool to parent id
                        {
                            if (pool_set.find(idx) == pool_set.end())
                            {
                                pool_set.insert(idx);
                                Pool2ParentsIdx.push_back(idx);
                                Pool2ImagesIdxRange.push_back(pair<size_t, size_t>(ImgLists.size(), ImgLists.size()));  // Add pool2image range
                            }
                            done_pool2parent_id = true;
                            Pool2ImagesIdxRange.back().second = ImgLists.size();                                        // Update pool2image range
                        }
                        else                                // Image to pool id
                        {
                            Img2PoolIdx.push_back(idx);
                            break;                          // Stop search, the rest is image name
                        }
                    }
                }

                // Image name
                ImgLists.push_back(read_line.substr(cpos_start, line_size - cpos_start));
            }
        }

        // Close file
        InImgFile.close();

        /*for (auto it = Pool2ImagesIdxRange.begin(); it != Pool2ImagesIdxRange.end(); it++)
            cout << it->first << " - " << it->second << endl;
        cout << "Pool2ImagesIdxRange.size():" << Pool2ImagesIdxRange.size() << endl;*/
    }
}

void ProcessDataset()
{
    // Checking image avalibality
    if (ImgLists.size() == 0)
    {
        cout << "No image available" << endl;
        return;
    }

    // ======== Preparing Dataset ========
    cout << "Pre-processing dataset..";
    cout.flush();
    startTime = CurrentPreciseTime();

    // Preprocess image, edge, resize, binarize
    for(size_t img_idx = 0; img_idx != ImgLists.size(); img_idx++)
    {
        stringstream cmd;
        stringstream curr_img_path;
        stringstream curr_img_tmp_path;
        stringstream curr_img_tmp_path_edge;
        stringstream curr_img_tmp_path_pre;
        curr_img_path << run_param.dataset_root_dir << "/" << ParentPaths[Img2ParentsIdx[img_idx]] << "/" << ImgLists[img_idx];
        curr_img_tmp_path << str_replace_first(curr_img_path.str(), "dataset", "dataset_tmp");
        curr_img_tmp_path_edge << str_replace_first(curr_img_tmp_path.str(), ".png", "-edge.png");
        curr_img_tmp_path_pre << str_replace_first(curr_img_tmp_path_edge.str(), ".png", "-pre.png");

        // Prepare directory
        make_dir_available(get_directory(curr_img_tmp_path.str()));

        //cout << "Preprocessing " << ImgLists[img_idx] << "..";
        //cout.flush();

		if(!is_path_exist(curr_img_tmp_path_pre.str())) // Preprocessed file does not exist
		{
            // edge, gray, and scale command
            //cmd << "convert " << curr_img_path.str() << " -colorspace Gray -threshold 5% -edge 3 -blur 0x.5 -resize 256x256 -blur 0x.5 " << curr_img_tmp_path_pre.str();
            //cmd << "./edges " << curr_img_path.str() << " " << curr_img_tmp_path_edge.str() << " && convert " << curr_img_tmp_path_edge.str() << " -colorspace Gray -threshold 5% -resize 256x256 -blur 0x.5 " << curr_img_tmp_path_pre.str();
            cmd << "~/webstylix/code/bin/edges " << curr_img_path.str() << " " << curr_img_tmp_path_edge.str() << " && convert " << curr_img_tmp_path_edge.str() << " -colorspace Gray -resize 256x256 -blur 0x.5 " << curr_img_tmp_path_pre.str();
            //cmd << " && ~/webstylix/code/lib/sifthesaff/bin/Release/sifthesaff_extractor -i " << curr_img_tmp_path_pre.str() << " -o " << str_replace(curr_img_tmp_path_pre.str(), "dataset", "dataset_feature") << "/" << run_param.feature_name << ".sifthesaff -m b";
            cout << cmd.str();
            exec(cmd.str());
        }
        //else
        //  cout << "skipped!" << endl;

        percentout(img_idx, ImgLists.size(), 1);
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
}

void ExtractDataset(bool save_feature)
{
    // Checking image avalibality
    if (ImgLists.size() == 0)
    {
        cout << "No image available" << endl;
        return;
    }

    /*// Parallel feature extraction
    // use with really good i/o, for network disk is not good enough
    int total_machine = 1;
    int machine_no = 0;
    char empty_char;
    cout << "Tell me total running machine and machine number eg. 't4m0s' or not specify 'n' :) ? - "; cout.flush();
    cin >> empty_char;          // t
    if (empty_char != 'n')
    {
        cin >> total_machine;   // 4
        cin >> empty_char;      // m
        cin >> machine_no;      // 0
        cin >> empty_char;      // s
    }
    */

    // Skip until file
    cout << "Do you want to skip file until [n|filename]: "; cout.flush();
    string skip_to_file;
    cin >> skip_to_file;
    size_t resume_id = 0;
    if (skip_to_file != "n")
    {
        for (resume_id = 0; resume_id < ImgLists.size(); resume_id++)
        {
            if (ImgLists[resume_id] == skip_to_file)
                break;
        }
        // To start at the next file
        resume_id++;
    }

    /*// use with really good i/o, for network disk is not good enough
    resume_id += machine_no;*/

    // ======== Extracting dataset feature ========
    cout << "Extracting dataset..";
    cout.flush();
    startTime = CurrentPreciseTime();

    SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift); // color, normalize, rootsift

    //bool toggle_skip = false;
    //bool start_ls2null = false;
    //for (size_t img_idx = resume_id; img_idx < ImgLists.size(); img_idx += total_machine) // for network disk is not good enough
    for (size_t img_idx = resume_id; img_idx < ImgLists.size(); img_idx++)
    {
        /*int skip_to_rand = 0;
        int max_skip = 32;
        if (max_skip > (int)(ImgLists.size() - img_idx))
            max_skip = (int)(ImgLists.size() - img_idx);
        if (toggle_skip)
            skip_to_rand = rand() % max_skip + img_idx;
        toggle_skip = !toggle_skip;*/

        stringstream curr_img_parent;
        stringstream curr_img_path;
        stringstream curr_img_export_parent;
        stringstream curr_img_export_path;
        curr_img_parent << run_param.dataset_root_dir << "/" << ParentPaths[Img2ParentsIdx[img_idx]];
        curr_img_path << curr_img_parent.str() << "/" << ImgLists[img_idx];
        curr_img_export_parent << run_param.dataset_feature_root_dir << "/" << run_param.feature_name << "/" << ParentPaths[Img2ParentsIdx[img_idx]];
        curr_img_export_path << curr_img_export_parent.str() << "/" << ImgLists[img_idx] << ".sifthesaff";


        /// Parallel feature extraction can be done here

        // Check existing, no extracting if exist
        //if (start_ls2null)
            //ls2null(curr_img_export_path.str());
        if (!islock(curr_img_export_path.str()) && !is_path_exist(curr_img_export_path.str()))
        {
            //start_ls2null = true;

            /// Lock file
            lockfile(curr_img_export_path.str());

            /// Extracting feature
            //cout << "curr_img_path" << curr_img_path.str() << endl;
            sifthesaff_dataset.extractPerdochSIFT(curr_img_path.str());

            /// Save extracted features
            //cout << "curr_img_export_path: " << curr_img_export_path.str() << endl;
            if (save_feature)
                sifthesaff_dataset.exportKeypoints(curr_img_export_path.str(), true);

            /// Unlock file
            unlockfile(curr_img_export_path.str());


            /*if (skip_to_rand > 0)
                img_idx -= 1;*/
        }

        //percentout(img_idx, ImgLists.size(), 1);
        percentout_timeleft(img_idx, resume_id, ImgLists.size(), startTime, 1);

        // Sampling
        //img_idx += 5;
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
}

void PackFeature(size_t block_size)
{
    /// Load each image features
    // Checking image avalibality
    if (ImgLists.size() == 0)
    {
        cout << "No image available" << endl;
        return;
    }

    // Release memory
    total_features = 0;
    vector<int>().swap(feature_count_per_pool);
    vector<int>().swap(feature_count_per_image);
    vector<int>().swap(image_count_per_pool);

    // Check existing poolinfo
    if (!is_path_exist(run_param.poolinfo_path))  // Not exist, create new pool
    {
        /// Preparing poolinfo
        for (size_t pool_id = 0; pool_id < Pool2ParentsIdx.size(); pool_id++)
        {
            feature_count_per_pool.push_back(0);
            image_count_per_pool.push_back(0);
        }
        for (size_t image_id = 0; image_id < ImgLists.size(); image_id++)
            feature_count_per_image.push_back(0);

        /// ======== Loading and writing features to hdf5 block-by-block ========
        bool did_write = false;

        // Sift header
        int sift_head_len = SIFThesaff::GetSIFTHeadSize();  // keypoint dimension
        int sift_dim = SIFThesaff::GetSIFTD();              // dataset dimension

        // Resuming
        size_t resume_id = 0;
        bool repair_last_block = false;
        size_t completed_feature_count = 0;
        if (is_path_exist(run_param.feature_keypoint_path) &&       // Exist part of feature
            is_path_exist(run_param.feature_descriptor_path) &&     // Exist part of feature
            is_path_exist(run_param.poolinfo_checkpoint_path))      // Exist part of poolinfo check point
        {
            // Load poolinfo checkpoint
            LoadPoolinfo(run_param.poolinfo_checkpoint_path);

            size_t kp_row, kp_col;
            size_t desc_row, desc_col;
            HDF_get_2Ddimension(run_param.feature_keypoint_path, "keypoint", kp_row, kp_col);
            HDF_get_2Ddimension(run_param.feature_descriptor_path, "descriptor", desc_row, desc_col);
            if (kp_row != desc_row || kp_row != total_features)
            {
                char opt = 'n';
                cout << "Fatal ERROR! feature_keypoint, feature_descriptor, and pool checkpoint are miss match!!" << endl;
                cout << "keypoint pack size: " << kp_row << endl;
                cout << "descriptor pack size: " << desc_row << endl;
                cout << "checkpoint pack size: " << total_features << endl;
                cout << "Trying to repair pack..?"; cout.flush();
                cin >> opt;
                if (opt == 'y')
                {
                    cout << "Please choose a img_idx to check resume possibility:" << endl;
                    size_t check_id;
                    cin >> check_id;
                    cout << "Calculating feature amount correctness.."; cout.flush();
                    for (size_t img_id = 0; img_id < check_id; img_id++)
                    {
                        completed_feature_count += feature_count_per_image[img_id];
                    }
                    if (completed_feature_count != total_features)
                    {
                        cout << "Resuming feature and checkpoint feature miss match" << endl;
                        cout << "completed_feature_count != total_features - " << completed_feature_count << " " << total_features << endl;
                        cin.clear();
                        cin.get();
                        return;
                    }
                    repair_last_block = true;
                }
                else
                    return;
            }
            else
            {
                completed_feature_count = kp_row;
            }

            did_write = true;
            cout << "Resuming available at feature_id:" << completed_feature_count << endl;

            cout << "WARNING, PLEASE choose a correct img_idx to be resumed:" << endl;
            cin >> resume_id;

            size_t double_check_resumable = 0;
            cout << "Calculating feature amount correctness.."; cout.flush();
            for (size_t img_id = 0; img_id < resume_id; img_id++)
            {
                double_check_resumable += feature_count_per_image[img_id];
            }
            if (double_check_resumable != completed_feature_count)
            {
                cout << "Resuming actual feature and checkpoint feature miss match" << endl;
                cout << "double_check_resumable != completed_feature_count - " << double_check_resumable << " " << completed_feature_count << endl;
                cin.clear();
                cin.get();
                return;
            }
        }

        string log_file = run_param.offline_working_path + "/packing.log";
        text_write(log_file, "==== Extracting and Packing log for " + run_param.dataset_header + " ====\n", false);

        try
        {
            cout << "Extracting and Packing dataset keypoint and descriptor..";
            cout.flush();
            startTime = CurrentPreciseTime();
            for (size_t img_idx = resume_id; img_idx < ImgLists.size();)
            {
                text_write(log_file, currentDateTime() + " img_id:" + toString(img_idx), true);

                // Preparing block
                size_t available_block = block_size;
                if (available_block > ImgLists.size() - img_idx)
                    available_block = ImgLists.size() - img_idx;

                text_write(log_file, " available_block:" + toString(available_block) + " next img_id:" + toString(img_idx + available_block), true);

                //==== Packing keypoint and descriptor
                size_t block_feature_count = 0;
                vector<float*> kp_buffer(available_block);
                vector<float*> desc_buffer(available_block);
                vector<int> numkp_buffer(available_block);

                /// Parallel packing
                #pragma omp parallel shared(available_block,block_feature_count,kp_buffer,desc_buffer,numkp_buffer,feature_count_per_pool,feature_count_per_image,image_count_per_pool,run_param,ParentPaths,Img2ParentsIdx,ImgLists)
                {
                    #pragma omp for schedule(dynamic,1) reduction(+ : block_feature_count)
                    for (size_t block_idx = 0; block_idx < available_block; block_idx++)
                    {
                        size_t curr_img_idx = img_idx + block_idx;
                        //string curr_img_feature_path = run_param.dataset_feature_root_dir + "/" + run_param.feature_name + "/" + ParentPaths[Img2ParentsIdx[curr_img_idx]] + "/" + ImgLists[curr_img_idx] + ".sifthesaff";
                        string curr_img_path = run_param.dataset_root_dir + "/" + ParentPaths[Img2ParentsIdx[curr_img_idx]] + "/" + ImgLists[curr_img_idx];

                        SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift, false);
                        //sifthesaff_dataset.importKeypoints(curr_img_feature_path);
                        sifthesaff_dataset.extractPerdochSIFT(curr_img_path);

                        /// Packing feature
                        int num_kp = sifthesaff_dataset.num_kp;
                        block_feature_count += num_kp;

                        /// Building Poolinfo
                        #pragma omp atomic
                        feature_count_per_pool[Img2PoolIdx[curr_img_idx]] += num_kp; // Accumulating feature count in the pool
                        feature_count_per_image[curr_img_idx] = num_kp;              // Keep feature for each image
                        #pragma omp atomic
                        image_count_per_pool[Img2PoolIdx[curr_img_idx]]++;           // Accumulating total image per pool


                        float* kp_dat = new float[num_kp * sift_head_len];      // keypoint data
                        float* desc_dat = new float[num_kp * sift_dim];         // descriptor data
                        for(int row = 0; row < num_kp; row++)
                        {
                            //== keypoint
                            for(int col = 0; col < sift_head_len; col++)
                                kp_dat[row * sift_head_len + col] = sifthesaff_dataset.kp[row][col];

                            //== descriptor
                            for(int col = 0; col < sift_dim; col++)
                                desc_dat[row * sift_dim + col] = sifthesaff_dataset.desc[row][col];
                        }
                        kp_buffer[block_idx] = kp_dat;
                        desc_buffer[block_idx] = desc_dat;
                        numkp_buffer[block_idx] = num_kp;
                    }
                }

                /// Flushing out to HDF5 file
                if (block_feature_count)
                {

                    total_features += block_feature_count;

                    text_write(log_file, " total_features:" + toString(total_features) + " block_feature_count:" + toString(block_feature_count), true);

                    float* block_kp = new float[block_feature_count * sift_head_len];
                    float* block_desc = new float[block_feature_count * sift_dim];
                    // Packing all features into one before writing to HDF5
                    size_t kp_offset = 0;
                    size_t desc_offset = 0;
                    for(size_t block_id = 0; block_id < available_block; block_id++)
                    {
                        // Packing kp
                        size_t kp_size = sift_head_len * numkp_buffer[block_id];
                        for(size_t col = 0; col < kp_size; col++)
                            block_kp[kp_offset + col] = kp_buffer[block_id][col];
                        kp_offset += kp_size;

                        // Pack desc
                        size_t desc_size = sift_dim * numkp_buffer[block_id];
                        for(size_t col = 0; col < desc_size; col++)
                            block_desc[desc_offset + col] = desc_buffer[block_id][col];
                        desc_offset += desc_size;
                    }

                    if (repair_last_block)
                    {
                        text_write(log_file, " repair kp..", true);
                        HDF_update_row_2DFLOAT(run_param.feature_keypoint_path, "keypoint", block_kp, completed_feature_count, block_feature_count, sift_head_len);
                        text_write(log_file, "ok desc..", true);
                        HDF_update_row_2DFLOAT(run_param.feature_descriptor_path, "descriptor", block_desc, completed_feature_count, block_feature_count, sift_dim);
                        text_write(log_file, "ok", true);
                        repair_last_block = false;
                    }
                    else
                    {
                        text_write(log_file, " kp..", true);
                        HDF_write_append_2DFLOAT(run_param.feature_keypoint_path, did_write, "keypoint", block_kp, block_feature_count, sift_head_len);
                        text_write(log_file, "ok desc..", true);
                        HDF_write_append_2DFLOAT(run_param.feature_descriptor_path, did_write, "descriptor", block_desc, block_feature_count, sift_dim);
                        text_write(log_file, "ok", true);
                    }

                    did_write = true;

                    // Release memory
                    for (size_t block_id = 0; block_id < available_block; block_id++)
                    {
                        delete[] kp_buffer[block_id];
                        delete[] desc_buffer[block_id];
                    }
                    delete[] block_kp;
                    delete[] block_desc;

                    img_idx += available_block;

                    text_write(log_file, " done\n", true);
                }

                // Save Poolinfo check point
                SavePoolinfo(run_param.poolinfo_checkpoint_path, false);

                // Clear buffer (already delete[] after flush)
                vector<float*>().swap(kp_buffer);
                vector<float*>().swap(desc_buffer);
                vector<int>().swap(numkp_buffer);

                //percentout(img_idx, ImgLists.size(), 1);
                percentout_timeleft(img_idx, resume_id, ImgLists.size(), startTime, 1);
            }
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            text_write(log_file, currentDateTime() + " Packing done\n", true);

            // Save Poolinfo
            SavePoolinfo(run_param.poolinfo_path);

            // Remove poolinfo checkpoint
            if (is_path_exist(run_param.poolinfo_path))
                remove(run_param.poolinfo_checkpoint_path.c_str());

            // Sending mail
            string mail_cmd = "echo 'Feature extraction DONE!! :)' | mail -s 'ins_offline notification..' k_siriwatk@hotmail.com";
            exec(mail_cmd);
        }
        catch (...)
        {
            cout << "Exception occured!" << endl;

            string mail_cmd = "echo 'Feature extraction error!\n\nPlease check' | mail -s 'ins_offline notification..' k_siriwatk@hotmail.com";
            exec(mail_cmd);
        }
    }
    else    // Exist, load existing pool
    {
        LoadPoolinfo(run_param.poolinfo_path);
    }

    cout << "Dataset contains " << Img2PoolIdx.back() + 1 << " pools of image" << endl; // pool of last image
    cout << "Total " << feature_count_per_image.size() << " images" << endl;
    cout << "Total " << total_features << " keypoints" << endl;
}

void LoadFeature(size_t start_idx, size_t load_size, int load_mode, Matrix<float>& load_data)
{
    // Feature Header
    int sift_head_len = SIFThesaff::GetSIFTHeadSize();
    int sift_dim = SIFThesaff::GetSIFTD();

    if (load_mode == LOAD_KP)  // Load with keypoint
    {
        float* kp_dat;      // keypoint data
        //kp_dat = NULL;

        /*cout << "Loading feature keypoint..";
        cout.flush();
        startTime = CurrentPreciseTime();*/
        HDF_read_row_2DFLOAT(run_param.feature_keypoint_path, "keypoint", kp_dat, start_idx, load_size);
        /*cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
        */
        // Wrap keypoint to matrix for flann knn search
        load_data = Matrix<float>(kp_dat, load_size, sift_head_len);

        /*
        cout << "kp [" << load_size << "," << 5 << "]" << endl;
        for (size_t row = 0; row < load_size; row++)
        {
            for (size_t col = 0; col < 5; col++)
                cout << kp_dat[row * 5 + col] << " ";
            cout << endl;
        }*/
    }

    if (load_mode == LOAD_DESC)  // Load with desciptor
    {
        float* desc_dat;    // descriptor data
        //desc_dat = NULL;

        /*cout << "Loading feature descriptor..";
        cout.flush();
        startTime = CurrentPreciseTime();*/
        HDF_read_row_2DFLOAT(run_param.feature_descriptor_path, "descriptor", desc_dat, start_idx, load_size);
        /*cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
        */

        // Wrap descriptor to matrix for flann knn search
        load_data = Matrix<float>(desc_dat, load_size, sift_dim);
    }
}

void SavePoolinfo(const string& out, const bool print)
{
    //==== Save feature pooling data
    if (print)
    {
        cout << "Saving pooling info..";
        cout.flush();
        startTime = CurrentPreciseTime();
    }
    ofstream PoolFile (out.c_str(), ios::binary);
    if (PoolFile.is_open())
    {
        // Write pool_size
        size_t pool_size = feature_count_per_pool.size();
        PoolFile.write(reinterpret_cast<char*>(&pool_size), sizeof(pool_size));

        // Write feature_count_per_pool
        PoolFile.write(reinterpret_cast<char*>(&feature_count_per_pool[0]), feature_count_per_pool.size() * sizeof(feature_count_per_pool[0]));

        // Write dataset_size
        size_t dataset_size = feature_count_per_image.size();
        PoolFile.write(reinterpret_cast<char*>(&dataset_size), sizeof(dataset_size));

        // Write pool
        PoolFile.write(reinterpret_cast<char*>(&feature_count_per_image[0]), feature_count_per_image.size() * sizeof(feature_count_per_image[0]));

        // Close file
        PoolFile.close();
    }
    if (print)
    {
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
}

void LoadPoolinfo(const string& in)
{
    // Consistency check
    size_t pool_features = 0;
    size_t img_features = 0;

    cout << "Loading pooling info..";
    cout.flush();
    startTime = CurrentPreciseTime();
    ifstream PoolFile (in.c_str(), ios::binary);
    if (PoolFile)
    {
        // Clear existing
        total_features = 0;
        vector<int>().swap(feature_count_per_pool);
        vector<int>().swap(feature_count_per_image);
        vector<int>().swap(image_count_per_pool);

        // Read pool_size
        size_t pool_size;
        PoolFile.read((char*)(&pool_size), sizeof(pool_size));

        // Read feature_count_per_pool
        for (size_t pool_idx = 0; pool_idx < pool_size; pool_idx++)
        {
            int feature_count;
            PoolFile.read((char*)(&feature_count), sizeof(feature_count));
            feature_count_per_pool.push_back(feature_count);
            pool_features += feature_count;
        }

        // Read dataset_size
        size_t dataset_size;
        PoolFile.read((char*)(&dataset_size), sizeof(dataset_size));

        // Read feature_count_per_image
        for (size_t dataset_idx = 0; dataset_idx < dataset_size; dataset_idx++)
        {
            int feature_count;
            PoolFile.read((char*)(&feature_count), sizeof(feature_count));
            feature_count_per_image.push_back(feature_count);

            img_features += feature_count;
        }

        // Close file
        PoolFile.close();

        // Calculate total image per pool
        for (size_t img_idx = 0; img_idx < Img2PoolIdx.size(); img_idx++)
        {
            if (image_count_per_pool.size() < size_t(Img2PoolIdx[img_idx] + 1))
                image_count_per_pool.push_back(0);
            image_count_per_pool[Img2PoolIdx[img_idx]]++;
        }

        if (pool_features != img_features)
        {
            cout << "Wrong poolinfo!! pool_features != img_features [ " << pool_features << " != " << img_features << "] " << endl;
            //exit(-1);
        }
    }
    else
    {
        cout << "Poolinfo not found! [" << in << "]" << endl;
        exit(-1);
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    total_features = img_features;
}

string SamplingDatabase(int sample_size)
{
    size_t total_image = feature_count_per_image.size();

    string feature_descriptor_sample_path = run_param.feature_descriptor_path + "_img-" + toString(sample_size);

    bool* sampling_mask = new bool[total_image]();  // Zero initialize

    // Randomize
    cout << "Randomize image..";
    cout.flush();
    startTime = CurrentPreciseTime();
    for (int sample_count = 0; sample_count < sample_size;)
    {
        size_t rand_img_id = random() % total_image;

        // Check this image has features more than zero
        if (feature_count_per_image[rand_img_id] > 0)
        {
            sampling_mask[rand_img_id] = true;
            sample_count++;

            percentout(sample_count, sample_size);
        }
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Sampling from HDF5 to HDF5 file
    cout << "Sampling..";
    cout.flush();
    startTime = CurrentPreciseTime();
    bool did_write = false;
    size_t offset_feature_id = 0;
    size_t current_feature_amount = 0;
    size_t total_feature_amount = 0;
    size_t dimension = SIFThesaff::GetSIFTD();
    for (size_t img_id = 0; img_id < total_image; img_id++)
    {
        current_feature_amount = feature_count_per_image[img_id];

        if (sampling_mask[img_id])
        {
            float* desc_dat;

            // Load from offset
            HDF_read_row_2DFLOAT(run_param.feature_descriptor_path, "descriptor", desc_dat, offset_feature_id, current_feature_amount);

            // Save append
            HDF_write_append_2DFLOAT(feature_descriptor_sample_path, did_write, "descriptor", desc_dat, current_feature_amount, dimension);
            did_write = true;

            // Accumulate total sampled features
            total_feature_amount += current_feature_amount;

            // Release memory
            delete[] desc_dat;

            //percentout(img_id, total_image);
            percentout_timeleft(img_id, 0, total_image, startTime, 5);
        }

        // Next offset
        offset_feature_id += current_feature_amount;
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    cout << "Total features: " << offset_feature_id << endl;
    cout << "Total sampled features: " << total_feature_amount << endl;

    exec("mv " + feature_descriptor_sample_path + " " + feature_descriptor_sample_path + "_feature-" + toString(total_feature_amount));
    feature_descriptor_sample_path = feature_descriptor_sample_path + "_feature-" + toString(total_feature_amount);

    // Release memory
    delete[] sampling_mask;

    return feature_descriptor_sample_path;
}

string SamplingOnTheFly()
{
    /// Load each image features
    // Checking image avalibality
    if (ImgLists.size() == 0)
    {
        cout << "No image available" << endl;
    }

    size_t target_feature_amount;
    cout << "Target features amount: "; cout.flush();
    cin >> target_feature_amount;
    string feature_descriptor_sample_path = run_param.feature_descriptor_path + "_target_" + toString(target_feature_amount);

    // Extracting feature from sampled image to HDF5 file
    cout << "Sampling on-the-fly..";
    // Release memory
    size_t block_size = 1024;
    size_t user_buffer;
    cout << "Do you want to set buffer size (default:1024)? 0 = no | #number = yes:"; cout.flush();
    cin >> user_buffer;
    if (user_buffer > 0)
        block_size = user_buffer;

    /// ======== Loading and writing features to hdf5 block-by-block ========
    bool did_write = false;

    // Sift header
    int sift_dim = SIFThesaff::GetSIFTD();              // dataset dimension

    // Sampling variables
    size_t total_image = ImgLists.size();
    bool* sampling_mask = new bool[total_image]();
    size_t sampled_image = 0;
    size_t sampled_feature = 0;

    cout << "Extracting and Packing dataset descriptor..";
    cout.flush();
    startTime = CurrentPreciseTime();
    while (sampled_feature < target_feature_amount)
    {
        // Sampled image_id
        vector<size_t> sampled_id;

        // Sampling
        for (size_t img_idx = 0; img_idx < block_size; )
        {
            size_t rand_img_id = random() % total_image;
            if (!sampling_mask[rand_img_id])
            {
                sampling_mask[rand_img_id] = true;
                sampled_id.push_back(rand_img_id);
                img_idx++;
            }
        }

        //==== Packing keypoint and descriptor
        size_t block_feature_count = 0;
        vector<float*> desc_buffer(block_size);
        vector<int> numkp_buffer(block_size);

        /// Parallel packing
        #pragma omp parallel shared(sampled_id,block_size,block_feature_count,desc_buffer,numkp_buffer,run_param,ParentPaths,Img2ParentsIdx,ImgLists)
        {
            #pragma omp for schedule(dynamic,1) reduction(+ : block_feature_count)
            for (size_t block_idx = 0; block_idx < block_size; block_idx++)
            {
                size_t curr_img_idx = sampled_id[block_idx];
                //string curr_img_feature_path = run_param.dataset_feature_root_dir + "/" + run_param.feature_name + "/" + ParentPaths[Img2ParentsIdx[curr_img_idx]] + "/" + ImgLists[curr_img_idx] + ".sifthesaff";
                string curr_img_path = run_param.dataset_root_dir + "/" + ParentPaths[Img2ParentsIdx[curr_img_idx]] + "/" + ImgLists[curr_img_idx];

                SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift, false);
                //sifthesaff_dataset.importKeypoints(curr_img_feature_path);
                sifthesaff_dataset.extractPerdochSIFT(curr_img_path);

                /// Packing feature
                int num_kp = sifthesaff_dataset.num_kp;
                block_feature_count += num_kp;

                float* desc_dat = new float[num_kp * sift_dim];         // descriptor data
                for(int row = 0; row < num_kp; row++)
                {
                    //== descriptor
                    for(int col = 0; col < sift_dim; col++)
                        desc_dat[row * sift_dim + col] = sifthesaff_dataset.desc[row][col];
                }
                desc_buffer[block_idx] = desc_dat;
                numkp_buffer[block_idx] = num_kp;
            }
        }

        /// Flushing out to HDF5 file
        if (block_feature_count)
        {
            float* block_desc = new float[block_feature_count * sift_dim];
            // Packing all features into one before writing to HDF5
            size_t desc_offset = 0;
            for(size_t block_id = 0; block_id < block_size; block_id++)
            {
                // Pack desc
                size_t desc_size = sift_dim * numkp_buffer[block_id];
                for(size_t col = 0; col < desc_size; col++)
                    block_desc[desc_offset + col] = desc_buffer[block_id][col];
                desc_offset += desc_size;
            }

            HDF_write_append_2DFLOAT(feature_descriptor_sample_path, did_write, "descriptor", block_desc, block_feature_count, sift_dim);

            did_write = true;

            // Release memory
            for (size_t block_id = 0; block_id < block_size; block_id++)
                delete[] desc_buffer[block_id];
            delete[] block_desc;

            sampled_image += block_size;
            sampled_feature += block_feature_count;
        }

        // Clear buffer (already delete[] after flush)
        vector<float*>().swap(desc_buffer);
        vector<int>().swap(numkp_buffer);

        //percentout(img_idx, ImgLists.size(), 1);
        percentout_timeleft(sampled_feature, 0, target_feature_amount, startTime, 1);
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Release memory
    delete[] sampling_mask;

    cout << "Total " << sampled_image << " images" << endl;
    cout << "Total " << sampled_feature << " keypoints" << endl;

    exec("mv " + feature_descriptor_sample_path + " " + run_param.feature_descriptor_path + "_img-" + toString(sampled_image) + "_feature-" + toString(sampled_feature));
    feature_descriptor_sample_path = run_param.feature_descriptor_path + "_img-" + toString(sampled_image) + "_feature-" + toString(sampled_feature);

    return feature_descriptor_sample_path;
}

void Clustering(bool save_cluster, bool hdf5)
{
    // Check existing cluster
    if (!is_path_exist(run_param.cluster_path))
    {
        // Check existing poolinfo
        if (feature_count_per_pool.size() == 0)
            LoadPoolinfo(run_param.poolinfo_path);

        char choice;
        bool with_mpi = true;
        cout << "==== CLuster configuration ====" << endl;
        cout << "Cluster size: " << run_param.CLUSTER_SIZE << endl;
        cout << "Dataset size: " << feature_count_per_image.size() << endl;
        cout << "Do you want to run with MPI? [y|n]:";
        cin >> choice;
        if (choice == 'n')
            with_mpi = false;

        // ======== Clustering Dataset ========
        cout << "Clustering..";
        cout.flush();
        startTime = CurrentPreciseTime();

        // Cluster preparation
        int km_cluster_size = run_param.CLUSTER_SIZE;
        int km_iteration = 50;

        if (hdf5) // MPI-AKM clustering
        {
            string vgg_fastcluster_path = run_param.offline_working_path + "/vgg_fastcluster.py";
            string start_clustering_path = run_param.offline_working_path + "/start_clustering.sh";

            char opt;
            int sample_size = feature_count_per_image.size();
            cout << "Do you want to sampling images/features [n|i|f] : "; cout.flush();
            cin >> opt;
            if (opt != 'n')
            {
                cout << "Recommend 35% of images = " << feature_count_per_image.size() * 35 / 100 << " , 35% of features = " << total_features *  35 / 100 << endl;
                cout << "Your preference: "; cout.flush();
                cin >> sample_size;

                // Convert feature to image based sampling
                if (opt == 'f')
                {
                    float sampling_ratio = total_features / sample_size;
                    sample_size = int(feature_count_per_image.size() / sampling_ratio);
                    cout << "approx. ~" << sample_size << " images" << endl;
                }

                run_param.feature_descriptor_path = SamplingDatabase(sample_size);
            }


            // Write vgg_fastcluster.py script
            /*
            #!/usr/bin/env python
            import time;
            import fastcluster;
			import ctypes;
			libfastcluster = ctypes.CDLL('libfastcluster.so');
            tic = time.time();
            fastcluster.kmeans("./cluster", "./feature_descriptor", km_cluster_size, km_iteration);
            toc = time.time();
            print "fastcluster.kmeans(dataset_header, km_cluster_size centers, km_iteration iterations) time: %.0f" %(toc-tic);
			libfastcluster.finalize();
            */
            ofstream PyFile (vgg_fastcluster_path.c_str());
            if (PyFile.is_open())
            {
                // Script
                PyFile << "#!/usr/bin/env python" << endl;
                PyFile << "import time;" << endl;
                PyFile << "import fastcluster;" << endl;
                PyFile << "import ctypes;" << endl;
                PyFile << "libfastcluster = ctypes.CDLL('libfastcluster.so');" << endl;
                PyFile << "tic = time.time();" << endl;
                PyFile << "fastcluster.kmeans(\"" << run_param.cluster_path << "\", \"" << run_param.feature_descriptor_path << "\", " << km_cluster_size << ", " << km_iteration << ");" << endl;
                PyFile << "toc = time.time();" << endl;
                PyFile << "print \"fastcluster.kmeans(" << run_param.dataset_header << ", " << km_cluster_size << " centers, " << km_iteration << " iterations) time: %.0f\" %(toc-tic);" << endl;
                PyFile << "libfastcluster.finalize();" << endl;

                // Close file
                PyFile.close();
            }

            // Write start_mpi_clustering.sh script
            /*
            mpirun -n run_param.MAXCPU python vgg_fastcluster.py
            */
            ofstream StartFile (start_clustering_path.c_str());
            if (StartFile.is_open())
            {
                // Script
                if (with_mpi)
                    StartFile << "mpirun -n " << run_param.MAXCPU << " python " << vgg_fastcluster_path << endl;
                else
                    StartFile << "python " << vgg_fastcluster_path << endl;

                // Close file
                StartFile.close();
            }

            // Start clustering
            exec("chmod +x " + start_clustering_path);

            cout << "Please run this command in another terminal \"" << start_clustering_path << "\"" << endl;
            cout << "Waiting for cluster result..."; cout.flush();
            // Waiting for cluster file
            while (!is_path_exist(run_param.cluster_path))
            {
                ls2null(run_param.cluster_path);
                usleep(1000000); // 1 second
            }
        }
        else    // FLANN HKM ckustering
        {
            /*
            // Release memory
            delete[] cluster.ptr();
            int actual_cluster_amount; = 0;

            // Flann k-mean param
            int k = 96;
            int km_branching = ((km_cluster_size - 1) / k + 1) + 1; // last +2 for higher than hierarchicalClustering tree cut method
            //int km_iteration = -1; // iterate until convergence
            KMeansIndexParams km_params(km_branching, km_iteration, FLANN_CENTERS_KMEANSPP);

            // Clustering
            // Prepare memory
            // Preparing empty cluster
            float* empty_cluster = new float[km_cluster_size * dimension];
            size_t total_idx = km_cluster_size * dimension;
            for (size_t idx = 0; idx < total_idx; idx++)
                empty_cluster[idx] = 0;
            Matrix<float> new_cluster(empty_cluster, km_cluster_size, dimension);
            actual_cluster_amount = hierarchicalClustering< ::flann::L2<float> >(dataset_descriptor, new_cluster, km_params);

            // Save cluster
            cout << "Saving cluster..";
            cout.flush();
            startTime = CurrentPreciseTime();
            if (save_cluster)
                SaveCluster(run_param.cluster_path);
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            // Keep cluster
            cluster = new_cluster;

            cout << actual_cluster_amount << " cluster(s) found" << endl;
            */
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
        cout << "Cluster is available, clustering not necessary." << endl;
}

void SaveCluster(const string& out)
{
    /*// Check directory
    string dir_name = get_directory(out);
    make_dir_available(dir_name);

    size_t cluster_dimension = cluster.cols;
    //size_t cluster_amount = actual_cluster_amount;
    size_t cluster_amount = run_param.CLUSTER_SIZE; // in case actual cluster is not equal as expected

    // Save to HDF5
    HDF_write_2DFLOAT(out, "clusters", cluster.ptr(), cluster_amount, cluster_dimension);*/
}

void ClusteringCoarseLayer()
{
    cout << "Please specify original cluster full-path:"; cout.flush();
    string source_cluster_path;
    cin >> source_cluster_path;

    /// Load source cluster
    cout << "Load source cluster.."; cout.flush();
        size_t source_cluster_amount;      // Cluster size
        size_t source_cluster_dimension;   // Feature dimension

        // Get HDF5 header
        HDF_get_2Ddimension(source_cluster_path, "clusters", source_cluster_amount, source_cluster_dimension);
        cout << "[" << source_cluster_dimension << ", " << source_cluster_amount << "].."; cout.flush();

        // Wrap data to matrix for flann knn searching
        float* load_cluster;   // will be replaced by read HDF5

        // Read from HDF5
        HDF_read_2DFLOAT(source_cluster_path, "clusters", load_cluster, source_cluster_amount, source_cluster_dimension);
        Matrix<float> source_cluster(load_cluster, source_cluster_amount, source_cluster_dimension);
    cout << "done!" << endl;

    /// Load current running cluster
    cout << "Load coarse layer cluster.."; cout.flush();
        Matrix<int> source_result_index;
        Matrix<float> source_result_dist;
        quantizer ann;
        ann.init(run_param);
    cout << "done!" << endl;

    /// Quantization
    cout << "Searching coarse assignment.."; cout.flush();
        ann.quantize(source_cluster, source_cluster_amount, source_result_index, source_result_dist);
    cout << "done!" << endl;

    /// Saving
    cout << "Saving map.."; cout.flush();
        int* source_result_index_ptr = source_result_index.ptr();
        float* source_result_dist_ptr = source_result_dist.ptr();
        fstream OutFile((run_param.offline_working_path + "/coarse_cluster_lut_" + toString(source_cluster_amount) + "-" + toString(run_param.CLUSTER_SIZE)).c_str(), ios::binary | ios::out);
        if (OutFile.is_open())
        {
            // Source cluster count
            OutFile.write(reinterpret_cast<char*>(&source_cluster_amount), sizeof(source_cluster_amount));

            // Quantize index and distance
            for (size_t lut_idx = 0; lut_idx < source_cluster_amount; lut_idx++)
            {
                // Index
                int quantized_index = source_result_index_ptr[lut_idx];
                OutFile.write(reinterpret_cast<char*>(&quantized_index), sizeof(quantized_index));
                // Dist
                float quantized_dist = source_result_dist_ptr[lut_idx];
                OutFile.write(reinterpret_cast<char*>(&quantized_dist), sizeof(quantized_dist));
            }

            // Close file
            OutFile.close();
        }
    cout << "done!" << endl;

    // Release memory
    delete[] source_cluster.ptr();
    delete[] source_result_index.ptr();
    delete[] source_result_dist.ptr();
}

void ImageFeaturesQuantization(bool save_quantized)
{
    // Check existing poolinfo
    if (feature_count_per_image.size() == 0)
        LoadPoolinfo(run_param.poolinfo_path);

    cout << "KNN searching..";
    cout.flush();
    startTime = CurrentPreciseTime();

    /// Create Quantizer
    quantizer ann;
    ann.init(run_param);

    /// Per image vector quantization
    // Feature vector per block preparation
    size_t accumulative_feature_amount = 0;
    bool is_write = false;
    size_t quantization_buffer_limit = 1024;
    size_t user_buffer;
    cout << "Do you want to set buffer size (default:1024)? 0 = no | #number = yes:"; cout.flush();
    cin >> user_buffer;
    if (user_buffer > 0)
        quantization_buffer_limit = user_buffer;
    //int dimension = dataset_descriptor.cols;
    /// Resuming checkpoint
    // Checking existing quantized data
    size_t resume_idx = 0;
    if (is_path_exist(run_param.quantized_path))
    {
        ifstream QuantizedFile(run_param.quantized_path.c_str(), ios::binary | ios::in);
        ifstream QuantizedOffsetFile(run_param.quantized_offset_path.c_str(), ios::binary | ios::in);
        if (QuantizedFile.is_open() && QuantizedOffsetFile.is_open())
        {
            // Checking quantized
            size_t quantized_count;
            size_t quantized_offset_count;

            // Quantized count
            QuantizedFile.read((char*)(&quantized_count), sizeof(quantized_count));
            QuantizedOffsetFile.read((char*)(&quantized_offset_count), sizeof(quantized_offset_count));
            // Checking header count is correct
            if (quantized_count == quantized_offset_count)
            {
                /// Checking actual data integrity
                // Reading last quantized_offset
                size_t last_quantized_offset_offset = sizeof(quantized_offset_count) + (quantized_offset_count - 1) * sizeof(size_t);
                size_t last_quantized_offset;
                QuantizedOffsetFile.seekg(last_quantized_offset_offset, QuantizedOffsetFile.beg);
                QuantizedOffsetFile.read((char*)(&last_quantized_offset), sizeof(last_quantized_offset));
                // Reading last quantized count
                size_t quantized_feature_count;
                QuantizedFile.seekg(last_quantized_offset, QuantizedFile.beg);
                QuantizedFile.read((char*)(&quantized_feature_count), sizeof(quantized_feature_count));

                // Checking last quantized feature count with feature_count_per_image
                if (int(quantized_feature_count) != feature_count_per_image[quantized_count - 1])
                {
                    cout << "Cannot resume, quantized file and quantized_offset file are not match." << endl;
                    return;
                }

                /// Pass!! Set resuming position
                resume_idx = quantized_count;

                // Flag for continue appending mode
                is_write = true;
            }
            else
            {
                cout << "Cannot resume, quantized file and quantized_offset file are not match." << endl;
                return;
            }

            QuantizedOffsetFile.close();
            QuantizedFile.close();
        }

        // Accumulating previous feature
        for (size_t image_id = 0; image_id < resume_idx; image_id++)
            accumulative_feature_amount += feature_count_per_image[image_id];

        cout << "Resuming at image_id:" << resume_idx << endl;
	}
/*
	// Repair
	is_write = true;
	cout << "---- REPAIR MODE ----" << endl;
	cout << "Enter img_id:"; cout.flush();
	size_t rp_img_idx = 0;
	cin >> rp_img_idx;
	{
	    if (!dataset_quantized_offset_ready)
            LoadQuantizedDatasetOffset();

	    // Accumulating feature amount
	    accumulative_feature_amount = 0;
	    for (size_t image_id = 0; image_id < rp_img_idx; image_id++)
            accumulative_feature_amount += feature_count_per_image[image_id];

	    size_t current_feature_amount = feature_count_per_image[rp_img_idx];

        // Resuming from resume_idx
        vector<int> quantized_counts;         // Total number per each quantized (one quantized is one image)
        vector<int*> quantized_indices;
        vector<float*> quantized_dists;
        Matrix<float> dataset_descriptor;

        // Load specific dataset_descriptor
        //LoadFeature(accumulative_feature_amount, current_feature_amount, LOAD_DESC, dataset_descriptor);
        SIFThesaff runtime_sift;
        runtime_sift.init(run_param.colorspace, run_param.normpoint, run_param.rootsift);
        cout << "Extracting image: " << run_param.dataset_root_dir << "/" << ParentPaths[Img2ParentsIdx[rp_img_idx]] << "/" << ImgLists[rp_img_idx] << endl;
        runtime_sift.extractPerdochSIFT(run_param.dataset_root_dir + "/" + ParentPaths[Img2ParentsIdx[rp_img_idx]] + "/" + ImgLists[rp_img_idx]);
        int sift_len = SIFThesaff::GetSIFTD();
        float* desc_dat = new float[runtime_sift.num_kp * sift_len];
        for (int kp_idx = 0; kp_idx < runtime_sift.num_kp; kp_idx++)
            for (int desc_idx = 0; desc_idx < sift_len; desc_idx++)
            desc_dat[kp_idx * sift_len + desc_idx] = runtime_sift.desc[kp_idx][desc_idx];

        dataset_descriptor = Matrix<float>(desc_dat, runtime_sift.num_kp, sift_len);

        //Matrix<float> feature_data(current_feature, current_feature_amount, dimension);
        Matrix<int> result_index; // size = feature_amount x knn
        Matrix<float> result_dist;

        ann.quantize(dataset_descriptor, current_feature_amount, result_index, result_dist);

        // Keep result
        // Keep in a buffer
        quantized_counts.push_back(current_feature_amount);
        quantized_indices.push_back(result_index.ptr());
        quantized_dists.push_back(result_dist.ptr());

        // Flushing buffer to disk
        size_t current_quantized_offset = dataset_quantized_offset[rp_img_idx];

        fstream OutFile(run_param.quantized_path.c_str(), ios::binary | ios::in | ios::out);
        if (OutFile.is_open())
        {
            char rp = 'n';
            // Quantize index and distance
            for (size_t quantized_idx = 0; quantized_idx < quantized_counts.size(); quantized_idx++)
            {
                // Feature size
                OutFile.seekg(current_quantized_offset, OutFile.beg);
                size_t feature_count = 0;
                OutFile.read((char*)(&feature_count), sizeof(feature_count));
                cout << "Replace feature_count " << feature_count << " -> " << quantized_counts[quantized_idx] << " ? [y|n]"; cout.flush();
                cin.clear(); cin >> rp;
                if (rp == 'y')
                {
                    OutFile.seekp(current_quantized_offset, OutFile.beg);
                    feature_count = quantized_counts[quantized_idx];
                    OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
                }
                current_quantized_offset += sizeof(feature_count);

                //cout << "feature_count: " << feature_count << endl;
                // Feature quantized index and distance to index
                for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
                {
                    // Index
                    cout << "feature_idx:" << feature_idx << endl;
                    OutFile.seekg(current_quantized_offset, OutFile.beg);
                    int quantized_index = 0;
                    OutFile.read((char*)(&quantized_index), sizeof(quantized_index));
                    if (quantized_index != quantized_indices[quantized_idx][feature_idx])
                    {
                        if (rp == 'y')
                        {
                            cout << "Replace quantized_index " << quantized_index << " -> " << quantized_indices[quantized_idx][feature_idx] << endl;
                            OutFile.seekp(current_quantized_offset, OutFile.beg);
                            quantized_index = quantized_indices[quantized_idx][feature_idx];
                            OutFile.write(reinterpret_cast<char*>(&quantized_index), sizeof(quantized_index));
                        }
                    }
                    current_quantized_offset += sizeof(quantized_index);
                    // Dist
                    OutFile.seekg(current_quantized_offset, OutFile.beg);
                    float quantized_dist = 0;
                    OutFile.read((char*)(&quantized_dist), sizeof(quantized_dist));
                    if (quantized_index != quantized_indices[quantized_idx][feature_idx])
                    {
                        if (rp == 'y')
                        {
                            cout << "Replace quantized_dist " << quantized_dist << " -> " << quantized_dists[quantized_idx][feature_idx] << endl;
                            OutFile.seekp(current_quantized_offset, OutFile.beg);
                            quantized_dist = quantized_dists[quantized_idx][feature_idx];
                            OutFile.write(reinterpret_cast<char*>(&quantized_dist), sizeof(quantized_dist));
                        }
                    }
                    current_quantized_offset += sizeof(quantized_dist);

                    //cout << "quantized_index: " << quantized_index << " quantized_dist: " << quantized_dist << endl;
                }
            }

            // Close file
            OutFile.close();
        }

        /// Clear written quantized buffer
        // Clear buffer
        for (size_t quantized_idx = 0; quantized_idx < quantized_indices.size(); quantized_idx++)
        {
            delete[] quantized_indices[quantized_idx];
            delete[] quantized_dists[quantized_idx];
        }
        quantized_indices.clear();
        quantized_dists.clear();
        // Clear size
        quantized_counts.clear();

        // Release memory
        delete[] dataset_descriptor.ptr();

        cout << "done!" << endl;

        // Stop the rest
        return;
    }
*/

    size_t dataset_size = feature_count_per_image.size();
	if (resume_idx < dataset_size)
    {
        // Resuming from resume_idx
        vector<int> quantized_counts;           // Total number of feature per image
        vector<int*> quantized_indices;         // quantized result for this block, one result int* is one image
        vector<float*> quantized_dists;         // quantized distance for this block, one result int* is one image
        for (size_t image_id = resume_idx; image_id < dataset_size; )
        {
            Matrix<float> dataset_descriptor_block;
            size_t block_feature_count = 0;

            // Calculating block_count
            size_t block_available = dataset_size - image_id;
            if (block_available > quantization_buffer_limit)
                block_available = quantization_buffer_limit;
            for (size_t block_idx = 0; block_idx < block_available; block_idx++)
            {
                size_t current_feature_amount = feature_count_per_image[image_id++];
                block_feature_count += current_feature_amount;
                // Keep feature count
                quantized_counts.push_back(current_feature_amount);
            }

            // Load specific dataset_descriptor as a block
            LoadFeature(accumulative_feature_amount, block_feature_count, LOAD_DESC, dataset_descriptor_block);

            // Accumulate offset of total feature per image for the next load
            accumulative_feature_amount += block_feature_count;

            // Quantizing block
            Matrix<int> block_result_index; // size = feature_amount x knn
            Matrix<float> block_result_dist;
            ann.quantize(dataset_descriptor_block, block_feature_count, block_result_index, block_result_dist);

            // Split results into each separated image
            size_t done_feature_offset = 0;
            int* block_result_index_ptr = block_result_index.ptr();
            float* block_result_dist_ptr = block_result_dist.ptr();
            for (size_t block_idx = 0; block_idx < block_available; block_idx++)
            {
                size_t feature_count = quantized_counts[block_idx];

                int* result_index = new int[feature_count];
                float* result_dist = new float[feature_count];

                for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
                {
                    result_index[feature_idx] = block_result_index_ptr[done_feature_offset + feature_idx];
                    result_dist[feature_idx] = block_result_dist_ptr[done_feature_offset + feature_idx];
                }
                done_feature_offset += feature_count;

                // Keep result
                // Keep in a buffer
                quantized_indices.push_back(result_index);
                quantized_dists.push_back(result_dist);
            }

            // Flushing buffer to disk
            SaveQuantizedDataset(quantized_counts, quantized_indices, quantized_dists, is_write);
            is_write = true;

            /// Clear written quantized buffer
            // Clear buffer
            for (size_t quantized_idx = 0; quantized_idx < quantized_indices.size(); quantized_idx++)
            {
                delete[] quantized_indices[quantized_idx];
                delete[] quantized_dists[quantized_idx];
            }
            vector<int*>().swap(quantized_indices);
            vector<float*>().swap(quantized_dists);
            // Clear size
            vector<int>().swap(quantized_counts);


            percentout_timeleft(image_id, resume_idx, dataset_size, startTime, 4);

            // Release memory
            delete[] dataset_descriptor_block.ptr();
            delete[] block_result_index.ptr();
            delete[] block_result_dist.ptr();
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
        cout << "Quantizing not necessary. Dataset has been quantized." << endl;

/*
    // Accessing data in matrix
    int* result_index_idx = result_index.ptr();

    for(size_t row = 0; row < result_index.rows; row++)
    {
        for(size_t col = 0; col < result_index.cols; col++)
        {
            cout << result_index_idx[row * result_index.cols + col] << " ";
            if(col == result_index.cols - 1)
                cout << endl;
        }
    }
*/
    ///---- AKM Search End ----///
}

void SaveQuantizedDataset(const vector<int>& quantized_counts, const vector<int*>& quantized_indices, const vector<float*>& quantized_dists, bool append)
{
    size_t quantized_count;             // quantized_count is image count
    size_t current_quantized_offset;
    vector<size_t> quantized_offset;

    fstream OutFile;
    if (append)
        OutFile.open(run_param.quantized_path.c_str(), ios::binary | ios::in | ios::out);
    else
        OutFile.open(run_param.quantized_path.c_str(), ios::binary | ios::out);
    if (OutFile.is_open())
    {
        // Quantized count
        // If appending, read existing quantized_count from its header
        if (append)
        {
            // Read current quantized_count
            OutFile.seekg(0, OutFile.beg);
            OutFile.read((char*)(&quantized_count), sizeof(quantized_count));

            // Update quantized_count for append mode
            OutFile.seekp(0, OutFile.beg);
            quantized_count += quantized_counts.size();
            OutFile.write(reinterpret_cast<char*>(&quantized_count), sizeof(quantized_count));

            // Go to the end of stream
            OutFile.seekp(0, OutFile.end);
            current_quantized_offset = OutFile.tellp();
        }
        else
        {
            // Write at the beginning of stream for normal mode
            quantized_count = quantized_counts.size();
            OutFile.write(reinterpret_cast<char*>(&quantized_count), sizeof(quantized_count));

            // Start after read quantized_count
            current_quantized_offset = sizeof(quantized_count);
        }

        // Quantize index and distance
        for (size_t quantized_idx = 0; quantized_idx < quantized_counts.size(); quantized_idx++)
        {
            // Keep offset
            quantized_offset.push_back(current_quantized_offset);

            // Feature size
            size_t feature_count = quantized_counts[quantized_idx];
            OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
            current_quantized_offset += sizeof(feature_count);

            //cout << "feature_count: " << feature_count << endl;
            // Feature quantized index and distance to index
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                // Index
                int quantized_index = quantized_indices[quantized_idx][feature_idx];
                OutFile.write(reinterpret_cast<char*>(&quantized_index), sizeof(quantized_index));
                current_quantized_offset += sizeof(quantized_index);
                // Dist
                float quantized_dist = quantized_dists[quantized_idx][feature_idx];
                OutFile.write(reinterpret_cast<char*>(&quantized_dist), sizeof(quantized_dist));
                current_quantized_offset += sizeof(quantized_dist);

                //cout << "quantized_index: " << quantized_index << " quantized_dist: " << quantized_dist << endl;
            }
        }

        // Write offset
        bin_write_vector_SIZET(run_param.quantized_offset_path, quantized_offset, append);

        // Close file
        OutFile.close();

        // Release memory
        vector<size_t>().swap(quantized_offset);
    }
}

void LoadQuantizedDatasetOffset()
{
    //cout << "Loading quantized dataset offset..."; cout.flush();
    if (!bin_read_vector_SIZET(run_param.quantized_offset_path, dataset_quantized_offset))
    {
        cout << "Quantized dataset offset file does not exits, (" << run_param.quantized_offset_path << ")" << endl;
        exit(-1);
    }
    //cout << "done!" << endl;

    dataset_quantized_offset_ready = true;
}

void LoadSpecificQuantizedDataset(vector<int>& quantized_counts, vector<int*>& quantized_indices, vector<float*>& quantized_dists, size_t start_idx, size_t load_size)
{
    // Load quantized offset
    if (!dataset_quantized_offset_ready)
        LoadQuantizedDatasetOffset();

    ifstream InFile (run_param.quantized_path.c_str(), ios::binary);
    if (InFile)
    {
        /// Skip to quantized index of specific start_idx
        size_t curr_offset = dataset_quantized_offset[start_idx];
        InFile.seekg(curr_offset, InFile.beg);

        for (size_t quantized_idx = 0; quantized_idx < load_size; quantized_idx++)
        {
            // Feature size
            size_t feature_count;
            InFile.read((char*)(&feature_count), sizeof(feature_count));

            // Feature quantized index and distance to index
            int* quantized_index = new int[feature_count];
            float* quantized_dist = new float[feature_count];
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                // Index
                InFile.read((char*)(&(quantized_index[feature_idx])), sizeof(quantized_index[feature_idx]));
                // Dist
                InFile.read((char*)(&(quantized_dist[feature_idx])), sizeof(quantized_dist[feature_idx]));
            }
            quantized_counts.push_back(feature_count);
            quantized_indices.push_back(quantized_index);
            quantized_dists.push_back(quantized_dist);
            //cout << "quantized_indices.size(): " << quantized_indices.size() << endl;
        }

        // Close file
        InFile.close();
    }
}

void ReleaseQuantizedOffset()
{
    // Release memory
    if (dataset_quantized_offset_ready)
    {
        // Clear offset
        dataset_quantized_offset_ready = false;
        vector<size_t>().swap(dataset_quantized_offset);
    }
}

void Bow(bool save_bow)
{
    /// Prerequisite check
    // Checking quantized dataset availability
    if (!is_path_exist(run_param.quantized_path))
    {
        cout << "No quantized dataset available, please run knn first" << endl;
        return;
    }

    /// Building Bow
    cout << "Building Bow..";
    cout.flush();
    startTime = CurrentPreciseTime();

    /// Create bow builder object
    bow bow_builder;
    bow_builder.init(run_param);

    /// Resuming position variable
    size_t accumulative_feature_amount = 0;
    bool is_write = false;

    /// Resuming checkpoint
    // Checking existing bow file
    size_t resume_idx = 0;
    if (is_path_exist(run_param.bow_path))
    {
        ifstream BowFile(run_param.bow_path.c_str(), ios::binary | ios::in);
        ifstream BowOffsetFile(run_param.bow_offset_path.c_str(), ios::binary | ios::in);
        ifstream BowPoolFile(run_param.bow_pool_path.c_str(), ios::binary | ios::in);
        ifstream BowPoolOffsetFile(run_param.bow_pool_offset_path.c_str(), ios::binary | ios::in);
        if (BowFile.is_open() && BowOffsetFile.is_open() &&
            (!run_param.pooling_enable || (BowPoolFile.is_open() && BowPoolOffsetFile.is_open())))
        {
            // Checking bow and bow_pool
            size_t bow_count;
            size_t bow_offset_count;
            size_t bow_pool_count;
            size_t bow_pool_offset_count;

            // Bow count and BowPool count
            BowFile.read((char*)(&bow_count), sizeof(bow_count));
            BowOffsetFile.read((char*)(&bow_offset_count), sizeof(bow_offset_count));
            if (run_param.pooling_enable)
            {
                BowPoolFile.read((char*)(&bow_pool_count), sizeof(bow_pool_count));
                BowPoolOffsetFile.read((char*)(&bow_pool_offset_count), sizeof(bow_pool_offset_count));
            }
            // Checking header count is correct
            if (bow_count == bow_offset_count &&
                (!run_param.pooling_enable || (bow_pool_count == bow_pool_offset_count)))
            {
                /// We assume if header is correct, data integrity should be corrected for both bow and bow_pool

                /// Pass!! Set resuming position
                resume_idx = bow_count;

                // Flag for continue appending mode
                is_write = true;
            }
            else
            {
                cout << "Cannot resume, bow file and bow_offset or bow_pool file and bow_pool_offset file are not match." << endl;
                return;
            }

            BowPoolOffsetFile.close();
            BowPoolFile.close();
            BowOffsetFile.close();
            BowFile.close();
        }

        // Accumulating previous feature
        for (size_t image_id = 0; image_id < resume_idx; image_id++)
            accumulative_feature_amount += feature_count_per_image[image_id];

        cout << "Resuming at image_id:" << resume_idx << endl;
	}
	//cout << "feature_count_per_image.size(): " << feature_count_per_image.size() << endl;
	//cout << "feature_count_per_pool.size(): " << feature_count_per_pool.size() << endl;
    if (resume_idx < feature_count_per_image.size())
    {
        int current_feature_amount = 0;
        int current_pool_image_count = 0;

        /// For each image
        vector<int> quantized_counts;         // Total number per each quantized (one quantized is one image)
        vector<int*> quantized_indices;
        vector<float*> quantized_dists;
        for (size_t image_id = resume_idx; image_id < feature_count_per_image.size(); image_id++)
        {
            Matrix<float> dataset_keypoint;

            current_feature_amount = feature_count_per_image[image_id];

            /* Skip the rest if nothing here <---   not good to do this,
                                                    since it will reduce total number of bow
                                                    or keep only non empty bow,
                                                    then we cannot access to original bow idx any more*/
            /*if (current_feature_amount == 0)
                continue;*/

            // Load each quantized index
            LoadSpecificQuantizedDataset(quantized_counts, quantized_indices, quantized_dists, image_id);       // Load quantized_indices, quantized_dists
            // Load each keypoint
            LoadFeature(accumulative_feature_amount, current_feature_amount, LOAD_KP, dataset_keypoint);        // Load dataset_key
            // Convert keypoint to vector< vector<float> >
            size_t dimension = dataset_keypoint.cols;
            float* dataset_keypoint_idx = dataset_keypoint.ptr();
            // Feature collection
            vector<float*> features;
            for (int feature_id = 0; feature_id < current_feature_amount; feature_id++)
            {
                size_t feature_offset = feature_id * dimension;

                // Create feature keypoint
                float* kp = new float[dimension];                                   // <---- new memory create here, please delete
                kp[0] = dataset_keypoint_idx[feature_offset + 0];  // x
                kp[1] = dataset_keypoint_idx[feature_offset + 1];  // y
                kp[2] = dataset_keypoint_idx[feature_offset + 2];  // a
                kp[3] = dataset_keypoint_idx[feature_offset + 3];  // b
                kp[4] = dataset_keypoint_idx[feature_offset + 4];  // c

                // Keep new kp into its corresponding bin (cluster_id)
                features.push_back(kp);
            }
            accumulative_feature_amount += current_feature_amount;
            // Accumulate pool counter
            current_pool_image_count++;

            // Build BoW
            bow_builder.build_bow(quantized_indices[0], features, Img2PoolIdx[image_id]);   // Build with image_id (no pooling) pool_id (pooling)

            // Powerlaw
            if (run_param.powerlaw_enable)
                bow_builder.rooting_lastbow();

            /****
            // Calculate total image per pool
            for (size_t image_id = 0; image_id < Img2PoolIdx.size(); image_id++)
            {
                if (image_count_per_pool.size() < size_t(Img2PoolIdx[image_id] + 1))
                    image_count_per_pool.push_back(0);
                image_count_per_pool[Img2PoolIdx[image_id]]++;
            }*/

            // If total features reach total features in the pool, do pooling then flush to disk
            if (current_pool_image_count == image_count_per_pool[Img2PoolIdx[image_id]])
            {
                // Flush bow to disk
                bow_builder.flush_bow(is_write);
                if (run_param.pooling_enable)
                {
                    // Pooling from internal multiple bow
                    // with internally logtf_unitnormalize before do pooling
                    bow_builder.build_pool(run_param.pooling_mode);
                    // Flush bow_pool to disk
                    bow_builder.flush_bow_pool(is_write);
                }

                is_write = true;

                // Reset pool counter
                current_pool_image_count = 0;

                // Release memory
                bow_builder.reset_bow();
                if (run_param.pooling_enable)
                    bow_builder.reset_bow_pool();
            }

            //percentout(image_id, feature_count_per_image.size(), 20);
            percentout_timeleft(image_id, resume_idx, feature_count_per_image.size(), startTime, 20);

            // Release memory
            delete[] dataset_keypoint.ptr();

            /// Release quantized data
            // Clear buffer
            for (size_t quantized_idx = 0; quantized_idx < quantized_counts.size(); quantized_idx++)
            {
                delete[] quantized_indices[quantized_idx];
                delete[] quantized_dists[quantized_idx];
            }
            vector<int>().swap(quantized_counts);
            vector<int*>().swap(quantized_indices);
            vector<float*>().swap(quantized_dists);
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
        cout << "Building BoW does not necessary. BoW has been built." << endl;


    // Release memory
    ReleaseQuantizedOffset();
}

void build_invert_index()
{
    cout << "Build Invert Index" << endl;
    cout << "Header: " << run_param.inv_header_path << endl;
    cout << "Data: " << run_param.inv_data_path << endl;

    // Checking bag_of_word avalibality
    if (!is_path_exist(run_param.bow_path))
    {
        cout << "No bag of word available" << endl;
        return;
    }
    else
        cout << "BOW OK" << endl;

    char word_major_opt = ' ';
    size_t word_buffer_size = run_param.CLUSTER_SIZE;
    size_t word_start = 0;
    size_t word_end = run_param.CLUSTER_SIZE - 1;
    cout << "Since large scale inverted index may have a problem with slow I/O." << endl;
    cout << "Do you want to build inverted index by word-major? [y|n]:"; cout.flush();
    cin >> word_major_opt;
    if (word_major_opt == 'y')
    {
        cout << "Please specify.." << endl;
        cout << "Total words per one flush: "; cout.flush();
        cin >> word_buffer_size;
        cout << "Word start: "; cout.flush();
        cin >> word_start;
        cout << "Word end: "; cout.flush();
        cin >> word_end;
        if (word_end > run_param.CLUSTER_SIZE - 1)
            word_end = run_param.CLUSTER_SIZE - 1;
    }

    size_t bow_buffer_limit = 1024;
    size_t user_buffer;
    cout << "Do you want to set buffer size (default:1024)? 0 = no | #number = yes:"; cout.flush();
    cin >> user_buffer;
    if (user_buffer > 0)
        bow_buffer_limit = user_buffer;

    bool inv_resume = false;
    if (word_start > 0)
    {
        inv_resume = true;
        cout << "Resume at " << word_start << endl;
    }
    cout << "Building inverted index database.." << endl;
    // Create invert index database
    invert_index invert_hist;
    invert_hist.init(run_param, inv_resume);
    // Create bow builder object (use as bow loader)
    bow bow_builder;
    bow_builder.init(run_param);

    // Checking BOW file integrity
    if (image_count_per_pool.size() != bow_builder.get_running_bow_size())
    {
        cout << "Bow/BowPool size is incorrect" << endl;
        cout << "expected:" << image_count_per_pool.size() << " actual:" << bow_builder.get_running_bow_size() << endl;
        exit(EXIT_FAILURE);
    }

    startTime = CurrentPreciseTime();

    size_t pool_size = feature_count_per_pool.size();
    while (word_start <= word_end)
    {
        size_t word_lower_bound = word_start;
        size_t word_upper_bound = word_start + word_buffer_size - 1;
        if (word_upper_bound > run_param.CLUSTER_SIZE)
            word_upper_bound = run_param.CLUSTER_SIZE - 1;
        timespec block_start = CurrentPreciseTime();
        cout << "[" << word_lower_bound << " - " << word_upper_bound << "] "; cout.flush();

        /// Building inverted index for each pool
        // Read existing bow
        for (size_t pool_id = 0; pool_id < pool_size; )
        {
            // Prepare block space
            size_t block_available = pool_size - pool_id;
            if (block_available > bow_buffer_limit)
                block_available = bow_buffer_limit;
            vector<size_t> read_id_list(block_available);
            vector< vector<bow_bin_object*> > read_bows(block_available);
            for (size_t block_idx = 0; block_idx < block_available; block_idx++)
                read_id_list[block_idx] = pool_id++;
            // Load
            bow_builder.load_running_bows(read_id_list, read_bows);    // <---- This create kp new memory inside, please delete

            // Adding to inverted index
            for (size_t block_idx = 0; block_idx < block_available; block_idx++)
            {
                vector<bow_bin_object*>& read_bow = read_bows[block_idx];

                // Normalization before use
                bow_builder.logtf_unitnormalize(read_bow);

                // Filter out of range cluster
                if (word_major_opt == 'y')
                    bow_builder.bow_filter(read_bow, word_lower_bound, word_upper_bound);

                // Add to inverted hist
                invert_hist.add(read_id_list[block_idx], read_bow);

                // Release bow_bin_object
                // since invert_hist use just feature_object and kp[]
                // Then it create an internal new dataset_object()
                // Deleting bow_bin_object is necessary here
                size_t bin_size = read_bow.size();
                for (size_t bin_id = 0; bin_id < bin_size; bin_id++)
                    delete read_bow[bin_id];                // delete bow_bin_object
                vector<bow_bin_object*>().swap(read_bow);
            }
            // Release memory
            vector<size_t>().swap(read_id_list);
            vector< vector<bow_bin_object*> >().swap(read_bows);

            //percentout(pool_id, feature_count_per_pool.size(), 20);
            percentout_timeleft(pool_id, 0, pool_size, block_start, 1);
        }
        // Flush to disk
        invert_hist.flush();

        // Continue next block
        word_start = word_upper_bound + 1;

        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(block_start) << " s)" << endl;
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Rebuild header
    char rebuild_opt = 'n';
    cout << "Do you want to re-build inverted index header? [y|n]:"; cout.flush();
    cin >> rebuild_opt;
    startTime = CurrentPreciseTime();
    if (rebuild_opt == 'y')
    {
        cout << "Building inverted index header.."; cout.flush();
        invert_hist.build_header_from_bow_file();
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
}


// Compatible checking
void SiftCheck()
{
    cout << "Checking.."; cout.flush();
    startTime = CurrentPreciseTime();
    #pragma omp parallel shared(startTime,run_param,ParentPaths,Img2ParentsIdx,ImgLists)
    {
        #pragma omp for schedule(dynamic,1) nowait
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            SIFThesaff siftchecker;
            stringstream curr_img_feature_path;
            curr_img_feature_path << run_param.dataset_feature_root_dir << "/" << run_param.feature_name << "/" << ParentPaths[Img2ParentsIdx[img_idx]] << "/" << ImgLists[img_idx] << ".sifthesaff";

            siftchecker.init(run_param.colorspace, run_param.normpoint, run_param.rootsift, false);
            siftchecker.checkNumKp(curr_img_feature_path.str());

            percentout_timeleft(img_idx, 0, ImgLists.size(), startTime, 20);
        }
    }
    cout << "done!" << endl;
}

void SiftPackRepair()
{
    size_t img_idx;
    cout << "Please input img_idx to be repaired: "; cout.flush();
    cin >> img_idx;

    cout << "Start extracting.."; cout.flush();
    string img_path = run_param.dataset_root_dir + "/" + ParentPaths[Img2ParentsIdx[img_idx]] + "/" + ImgLists[img_idx];
    SIFThesaff siftextract;
    siftextract.init(run_param.colorspace, run_param.normpoint, run_param.rootsift, false);
    siftextract.extractPerdochSIFT(img_path);
    cout << "done!" << endl;

    if (siftextract.num_kp == feature_count_per_image[img_idx])
    {
        cout << "Start updating.."; cout.flush();
        // Packing feature
        int sift_head_len = SIFThesaff::GetSIFTHeadSize();
        int sift_dim = SIFThesaff::GetSIFTD();
        float* kp_dat = new float[siftextract.num_kp * sift_head_len];
        float* desc_dat = new float[siftextract.num_kp * sift_dim];
        for(int row = 0; row < siftextract.num_kp; row++)
        {
            // Slowdown by copying here***

            //== keypoint
            for(int col = 0; col < sift_head_len; col++)
                kp_dat[row * sift_head_len + col] = siftextract.kp[row][col];

            //== descriptor
            for(int col = 0; col < sift_dim; col++)
                desc_dat[row * sift_dim + col] = siftextract.desc[row][col];
        }

        // Finding offset start
        size_t accumulated_offset = 0;
        for (size_t id = 0; id < img_idx; id++)
            accumulated_offset += feature_count_per_image[id];

        HDF_update_row_2DFLOAT(run_param.feature_keypoint_path, "keypoint", kp_dat, accumulated_offset, siftextract.num_kp, sift_head_len);
        HDF_update_row_2DFLOAT(run_param.feature_descriptor_path, "descriptor", desc_dat, accumulated_offset, siftextract.num_kp, sift_dim);
        cout << "done!" << endl;

        // Release memory
        delete[] kp_dat;
        delete[] desc_dat;
    }
    else
    {
        cout << "Total keypoint is not equal siftextract.num_kp != feature_count_per_image[img_idx] - " << siftextract.num_kp  << " != " << feature_count_per_image[img_idx] << endl;
    }
}

void PoolCheck()
{
    if (is_path_exist(run_param.poolinfo_path))
    {
        LoadPoolinfo(run_param.poolinfo_path);

        cout << "Checking total features from feature_count_per_image.."; cout.flush();
        size_t sum_features = 0;
        for (size_t img_idx = 0; img_idx < feature_count_per_image.size(); img_idx++)
            sum_features += feature_count_per_image[img_idx];
        if (sum_features == total_features)
            cout << "done!" << endl;
        else
            cout << "error, sum_features != total_features" << endl;

        cout << "Checking total features from feature_count_per_pool.."; cout.flush();
        sum_features = 0;
        for (size_t pool_idx = 0; pool_idx < feature_count_per_pool.size(); pool_idx++)
            sum_features += feature_count_per_pool[pool_idx];
        if (sum_features == total_features)
            cout << "done!" << endl;
        else
            cout << "error, sum_features != total_features" << endl;

        cout << "Checking total images.."; cout.flush();
        size_t sum_images = 0;
        for (size_t pool_idx = 0; pool_idx < image_count_per_pool.size(); pool_idx++)
            sum_images += image_count_per_pool[pool_idx];
        if (sum_images == feature_count_per_image.size())
            cout << "done!" << endl;
        else
            cout << "error, sum_images != feature_count_per_image.size()" << endl;

        /*cout << "Checking image_count_per_pool.."; cout.flush();
        for (size_t pool_idx = 0; pool_idx < image_count_per_pool.size(); pool_idx++)
        {
            if (1 == image_count_per_pool[pool_idx])
                cout << "pool is one at " << ParentPaths[Pool2ParentsIdx[pool_idx]] << " " << Pool2ParentsIdx[pool_idx] << " " << pool_idx << " - " << image_count_per_pool[pool_idx] << endl;
            if (0 == image_count_per_pool[pool_idx])
                cout << "pool is zero at " << ParentPaths[Pool2ParentsIdx[pool_idx]] << " " << Pool2ParentsIdx[pool_idx] << " " << pool_idx << " - " << image_count_per_pool[pool_idx] << endl;
        }
        cout << "done!" << endl;*/

        cout << "Checking feature per pool.."; cout.flush();
        for (size_t pool_idx = 0, img_idx = 0; pool_idx < image_count_per_pool.size(); pool_idx++)
        {
            int curr_total_features = 0;
            for (int sub_img_idx = 0; sub_img_idx < image_count_per_pool[pool_idx]; sub_img_idx++)
                curr_total_features += feature_count_per_image[sub_img_idx + img_idx];
            if (curr_total_features != feature_count_per_pool[pool_idx])
            {
                cout << "Poolinfo is wrong at pool " << pool_idx << " " << curr_total_features << "!=" << feature_count_per_pool[pool_idx] << endl;
                cout << "Parent " << ParentPaths[Pool2ParentsIdx[pool_idx]] << endl;
            }
            img_idx += image_count_per_pool[pool_idx];
        }
        cout << "done!" << endl;
    }
    else
        cout << "Pooling info not found" << endl;

    char rp_opt = 'n';
    cout << "Do you want to re-build or repair poolinfo? [y|n]:"; cout.flush();
    cin >> rp_opt;
    if (rp_opt == 'y')
        PoolRepair();
}

void PoolRepair()
{
    cout << "Repairing pool..";
    cout.flush();
    startTime = CurrentPreciseTime();

    // Clear previous value
    vector<int>().swap(feature_count_per_pool);
    vector<int>().swap(image_count_per_pool);
    vector<int>().swap(feature_count_per_image);

    /// Preparing poolinfo
    for (size_t pool_id = 0; pool_id < Pool2ParentsIdx.size(); pool_id++)
    {
        feature_count_per_pool.push_back(0);
        image_count_per_pool.push_back(0);
    }
    for (size_t image_id = 0; image_id < ImgLists.size(); image_id++)
        feature_count_per_image.push_back(0);

    /// Parallel importing sift
    #pragma omp parallel shared(feature_count_per_pool,feature_count_per_image,image_count_per_pool,run_param,ParentPaths,Img2ParentsIdx,ImgLists)
    {
        #pragma omp for schedule(dynamic,1)
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            string curr_img_feature_path = run_param.dataset_feature_root_dir + "/" + run_param.feature_name + "/" + ParentPaths[Img2ParentsIdx[img_idx]] + "/" + ImgLists[img_idx] + ".sifthesaff";

            SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift, false);
            sifthesaff_dataset.checkNumKp(curr_img_feature_path);

            /// Packing feature
            int num_kp = sifthesaff_dataset.num_kp;

            /// Building Poolinfo
            #pragma omp atomic
            feature_count_per_pool[Img2PoolIdx[img_idx]] += num_kp; // Accumulating feature count in the pool
            feature_count_per_image[img_idx] = num_kp;              // Keep feature for each image
            #pragma omp atomic
            image_count_per_pool[Img2PoolIdx[img_idx]]++;           // Accumulating total image per pool

            percentout_timeleft(img_idx, 0, ImgLists.size(), startTime, 20);
        }
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Save Poolinfo
    SavePoolinfo(run_param.poolinfo_path);
}

void QuantizedCorrectnessCheck()
{/*
    // Quantized version 1
    size_t q_datacount_v1;
    vector<int> q_count_v1;
    vector<vector<int>> q_idx_v1;
    vector<vector<float>> q_dist_v1;
    ifstream V1File ((run_param.quantized_path + "_v1").c_str(), ios::binary);
    if (V1File)
    {
        cout << "Reading quantized_v1.." << endl;
        // Dataset size
        V1File.read((char*)(&q_datacount_v1), sizeof(q_datacount_v1));

        timespec start_v1 = CurrentPreciseTime();
        // Indices
        for (size_t dataset_id = 0; dataset_id < q_datacount_v1; dataset_id++)
        {
            // Feature size
            size_t feature_count;
            V1File.read((char*)(&feature_count), sizeof(feature_count));
            q_count_v1.push_back(feature_count);

            // Feature quantized index and distance to index
            vector<int> q_sub_idx_v1;
            vector<float> q_sub_dist_v1;
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                // Index
                int index_data;
                V1File.read((char*)(&index_data), sizeof(index_data));
                q_sub_idx_v1.push_back(index_data);
                // Dist
                float dist_data;
                V1File.read((char*)(&dist_data), sizeof(dist_data));
                q_sub_dist_v1.push_back(dist_data);

                if (size_t(index_data) > run_param.CLUSTER_SIZE - 1)
                    cout << "wrong cluster_id:" << index_data << " img_id:" << dataset_id << " path:" << ParentPaths[Img2ParentsIdx[dataset_id]] << "/" << ImgLists[dataset_id] << endl;
            }
            q_idx_v1.push_back(q_sub_idx_v1);
            q_dist_v1.push_back(q_sub_dist_v1);

            percentout_timeleft(dataset_id, 0, q_datacount_v1, start_v1, 20);
        }

        // Close file
        V1File.close();
    }
*/
    if (!dataset_quantized_offset_ready)
        LoadQuantizedDatasetOffset();
    // Quantized version 2
    size_t q_datacount_v2;
    vector<int> q_count_v2;
    vector<vector<int>> q_idx_v2;
    vector<vector<float>> q_dist_v2;
    ifstream V2File ((run_param.quantized_path + "_v2").c_str(), ios::binary);
    if (V2File)
    {
        cout << "Reading quantized_v2.." << endl;

        // Dataset size
        V2File.read((char*)(&q_datacount_v2), sizeof(q_datacount_v2));

        timespec start_v2 = CurrentPreciseTime();
        cout << "Skip to:"; cout.flush();
        size_t to_id = 0;
        cin >> to_id;
        cout << "dataset_quantized_offset.size() = " << dataset_quantized_offset.size() << endl;
        if (to_id)
            V2File.seekg(dataset_quantized_offset[to_id]);
        for (size_t dataset_id = to_id; dataset_id < q_datacount_v2; dataset_id++)
        {
            // Feature size
            size_t feature_count;
            V2File.read((char*)(&feature_count), sizeof(feature_count));
            q_count_v2.push_back(feature_count);

            // Feature quantized index and distance to index
            vector<int> q_sub_idx_v2;
            vector<float> q_sub_dist_v2;
            bool show = false;
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                // Index
                int index_data;
                V2File.read((char*)(&index_data), sizeof(index_data));
                q_sub_idx_v2.push_back(index_data);
                // Dist
                float dist_data;
                V2File.read((char*)(&dist_data), sizeof(dist_data));
                q_sub_dist_v2.push_back(dist_data);

                if (size_t(index_data) > run_param.CLUSTER_SIZE - 1)
                    show = true;
                if (show)
                    cout << "wrong feature_idx:" << feature_idx << " cluster_id:" << index_data << " dist:" << dist_data << " img_id:" << dataset_id << " path:" << ParentPaths[Img2ParentsIdx[dataset_id]] << "/" << ImgLists[dataset_id] << endl;
            }
            q_idx_v2.push_back(q_sub_idx_v2);
            q_dist_v2.push_back(q_sub_dist_v2);

            percentout_timeleft(dataset_id, 0, q_datacount_v2, start_v2, 20);
        }

        // Close file
        V2File.close();
    }

/*
    // Checking
    cout << "Dataset size" << endl;
    cout << q_datacount_v1 << " - " << q_datacount_v2 << endl;
    cout << "Checking integrity.."; cout.flush();
    if (q_datacount_v1 == q_datacount_v2 && q_datacount_v1 == q_count_v1.size() && q_datacount_v2 == q_count_v2.size())
    {
        for (size_t data_idx = 0; data_idx < q_count_v1.size(); data_idx++)
        {
            if (q_count_v1[data_idx] == q_count_v2[data_idx])
            {
                for (size_t fea_idx = 0; fea_idx < q_idx_v1[data_idx].size(); fea_idx++)
                {
                    if (q_idx_v1[data_idx][fea_idx] != q_idx_v2[data_idx][fea_idx])
                        cout << data_idx << ":" << fea_idx << " " << q_idx_v1[data_idx][fea_idx] << " - " << q_idx_v2[data_idx][fea_idx] << " not equal" << endl;
                }
            }
            else
                cout << data_idx << " " << q_count_v1[data_idx] << " - " << q_count_v2[data_idx] << " not equal" << endl;

            percentout(data_idx, q_datacount_v1, 20);
        }
    }
    else
    {
        cout << "Data size not equal" << endl;
    }
    cout << "done" << endl;
*/
}

void BowCorrectnessCheck()
{
    vector<size_t> bow_offset_v1;
    cout << "Loading BOW offset v1..."; cout.flush();
    if (!bin_read_vector_SIZET(run_param.bow_offset_path + "_v1", bow_offset_v1))
    {
        cout << "BOW offset v1 file does not exits, (" << run_param.bow_offset_path << ")" << endl;
        exit(-1);
    }
    cout << "done!" << endl;
    vector<size_t> bow_offset_v2;
    cout << "Loading BOW offset v2..."; cout.flush();
    if (!bin_read_vector_SIZET(run_param.bow_offset_path + "_v2", bow_offset_v2))
    {
        cout << "BOW offset v2 file does not exits, (" << run_param.bow_offset_path << ")" << endl;
        exit(-1);
    }
    cout << "done!" << endl;

    ifstream V1File ((run_param.bow_path + "_v1").c_str(), ios::binary);
    ifstream V2File ((run_param.bow_path + "_v2").c_str(), ios::binary);
    if (V1File && V2File)
    {
        cout << "Checking integrity.."; cout.flush();
        for (size_t image_id = 0; image_id < bow_offset_v1.size(); image_id++)
        {
            vector<bow_bin_object*> bow_sig_v1;
            // Bow v1
            {
                /// Prepare bow buffer
                size_t curr_offset = bow_offset_v1[image_id];
                size_t buffer_size = 0;
                if (image_id < bow_offset_v1.size() - 1)
                    buffer_size = bow_offset_v1[image_id + 1] - curr_offset;
                else
                {
                    V1File.seekg(0, V1File.end);
                    buffer_size = V1File.tellg();
                    buffer_size -= curr_offset;
                }
                char* bow_buffer = new char[buffer_size];
                char* bow_buffer_ptr = bow_buffer;
                V1File.seekg(curr_offset, V1File.beg);
                V1File.read(bow_buffer, buffer_size);

                /// Bow hist
                // Dataset ID (read, but not use)
                size_t dataset_id_read = *((size_t*)bow_buffer_ptr);
                bow_buffer_ptr += sizeof(dataset_id_read);

                // Non-zero count
                size_t bin_count = *((size_t*)bow_buffer_ptr);
                bow_buffer_ptr += sizeof(bin_count);

                // ClusterID and FeatureIDs
                int head_size = SIFThesaff::GetSIFTHeadSize();
                for (size_t bin_idx = 0; bin_idx < bin_count; bin_idx++)
                {
                    // Create bin_obj
                    bow_bin_object* read_bin = new bow_bin_object();

                    // Cluster ID
                    read_bin->cluster_id = *((size_t*)bow_buffer_ptr);
                    bow_buffer_ptr += sizeof(read_bin->cluster_id);

                    // Weight
                    read_bin->weight = *((float*)bow_buffer_ptr);
                    bow_buffer_ptr += sizeof(read_bin->weight);

                    // Feature count
                    size_t feature_count;
                    feature_count = *((size_t*)bow_buffer_ptr);
                    bow_buffer_ptr += sizeof(feature_count);
                    for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
                    {
                        feature_object* feature = new feature_object();

                        // Feature ID as sequence_id
                        feature->sequence_id = *((size_t*)bow_buffer_ptr);
                        bow_buffer_ptr += sizeof(feature->sequence_id);
                        // x y a b c
                        feature->kp = new float[head_size];
                        for (int head_idx = 0; head_idx < head_size; head_idx++)
                        {
                            feature->kp[head_idx] = *((float*)bow_buffer_ptr);
                            bow_buffer_ptr += sizeof(feature->kp[head_idx]);
                        }

                        read_bin->features.push_back(feature);
                    }

                    // Keep bow
                    bow_sig_v1.push_back(read_bin);
                }

                // Release bow_buffer
                delete[] bow_buffer;
            }

            map<size_t, bow_bin_object*> bow_sig_v2_map;
            vector<bow_bin_object*> bow_sig_v2;
            // Bow v1
            {
                /// Prepare bow buffer
                size_t curr_offset = bow_offset_v2[image_id];
                size_t buffer_size = 0;
                if (image_id < bow_offset_v2.size() - 1)
                    buffer_size = bow_offset_v2[image_id + 1] - curr_offset;
                else
                {
                    V2File.seekg(0, V2File.end);
                    buffer_size = V2File.tellg();
                    buffer_size -= curr_offset;
                }
                char* bow_buffer = new char[buffer_size];
                char* bow_buffer_ptr = bow_buffer;
                V2File.seekg(curr_offset, V2File.beg);
                V2File.read(bow_buffer, buffer_size);

                /// Bow hist
                // Dataset ID (read, but not use)
                size_t dataset_id_read = *((size_t*)bow_buffer_ptr);
                bow_buffer_ptr += sizeof(dataset_id_read);

                // Non-zero count
                size_t bin_count = *((size_t*)bow_buffer_ptr);
                bow_buffer_ptr += sizeof(bin_count);

                // ClusterID and FeatureIDs
                int head_size = SIFThesaff::GetSIFTHeadSize();
                for (size_t bin_idx = 0; bin_idx < bin_count; bin_idx++)
                {
                    // Create bin_obj
                    bow_bin_object* read_bin = new bow_bin_object();

                    // Cluster ID
                    read_bin->cluster_id = *((size_t*)bow_buffer_ptr);
                    bow_buffer_ptr += sizeof(read_bin->cluster_id);

                    // Weight
                    read_bin->weight = *((float*)bow_buffer_ptr);
                    bow_buffer_ptr += sizeof(read_bin->weight);

                    // Feature count
                    size_t feature_count;
                    feature_count = *((size_t*)bow_buffer_ptr);
                    bow_buffer_ptr += sizeof(feature_count);
                    for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
                    {
                        feature_object* feature = new feature_object();

                        // Feature ID as sequence_id
                        feature->sequence_id = *((size_t*)bow_buffer_ptr);
                        bow_buffer_ptr += sizeof(feature->sequence_id);
                        // x y a b c
                        feature->kp = new float[head_size];
                        for (int head_idx = 0; head_idx < head_size; head_idx++)
                        {
                            feature->kp[head_idx] = *((float*)bow_buffer_ptr);
                            bow_buffer_ptr += sizeof(feature->kp[head_idx]);
                        }

                        read_bin->features.push_back(feature);
                    }

                    // Keep bow
                    bow_sig_v2_map[read_bin->cluster_id] = read_bin;
                }

                for (auto bow_sig_v2_it = bow_sig_v2_map.begin(); bow_sig_v2_it != bow_sig_v2_map.end(); bow_sig_v2_it++)
                    bow_sig_v2.push_back(bow_sig_v2_it->second);
                bow_sig_v2_map.clear();

                // Release bow_buffer
                delete[] bow_buffer;
            }

            // Checking
            if (bow_sig_v1.size() == bow_sig_v2.size())
            {
                // tf-unitnormalize
                bow bow_builder;
                bow_builder.init(run_param);
                bow_builder.logtf_unitnormalize(bow_sig_v2);

                for (size_t bin_idx = 0; bin_idx < bow_sig_v1.size(); bin_idx++)
                {
                    if (bow_sig_v1[bin_idx]->cluster_id == bow_sig_v2[bin_idx]->cluster_id)
                    {
                        if (bow_sig_v1[bin_idx]->weight != bow_sig_v2[bin_idx]->weight)
                            cout << setprecision(20) << fixed << "weight not equal " << bow_sig_v1[bin_idx]->weight << " - " << bow_sig_v2[bin_idx]->weight << endl;
                    }
                    else
                        cout << "bin_cluster not equal, " << bow_sig_v1[bin_idx]->cluster_id << " - " << bow_sig_v2[bin_idx]->cluster_id << endl;
                }
            }
            else
                cout << "bow_size not equal, " << bow_sig_v1.size() << " - " << bow_sig_v2.size() << endl;

            // Release bow sig v1
            for (size_t bin_idx = 0; bin_idx < bow_sig_v1.size(); bin_idx++)
            {
                for (size_t feature_idx = 0; feature_idx < bow_sig_v1[bin_idx]->features.size(); feature_idx++)
                {
                    delete[] bow_sig_v1[bin_idx]->features[feature_idx]->kp;  // delete float* kp[]
                    delete bow_sig_v1[bin_idx]->features[feature_idx];        // delete feature_object*
                }
                bow_sig_v1[bin_idx]->features.clear();
                delete bow_sig_v1[bin_idx];                                   // delete bow_bin_object*
            }
            bow_sig_v1.clear();

            // Release bow sig v2
            for (size_t bin_idx = 0; bin_idx < bow_sig_v2.size(); bin_idx++)
            {
                for (size_t feature_idx = 0; feature_idx < bow_sig_v2[bin_idx]->features.size(); feature_idx++)
                {
                    delete[] bow_sig_v2[bin_idx]->features[feature_idx]->kp;  // delete float* kp[]
                    delete bow_sig_v2[bin_idx]->features[feature_idx];        // delete feature_object*
                }
                bow_sig_v2[bin_idx]->features.clear();
                delete bow_sig_v2[bin_idx];                                   // delete bow_bin_object*
            }
            bow_sig_v2.clear();


            percentout(image_id, bow_offset_v1.size(), 20);
        }
        cout << "done!" << endl;

        // Close file
        V1File.close();
        V2File.close();
    }
    else
        cout << "Bow path incorrect! : " << run_param.bow_path << endl;
}

void InvDefCorrectnessCheck()
{
    cout << "Read inverted header v1.."; cout.flush();
    // Load existing header V1
    size_t total_df_v1 = 0;
    size_t data_size_v1 = 0;
    size_t* actual_cluster_amount_v1 = new size_t[run_param.CLUSTER_SIZE];
    float* idf_v1 = new float[run_param.CLUSTER_SIZE];
    ifstream V1File ((run_param.inv_data_path + "_v1").c_str(), ios::binary);
    if (V1File)
    {
        // Load data_size_v1
        V1File.read((char*)(&data_size_v1), sizeof(data_size_v1));

        // Load cluster_amount
        for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        {
            // Read cluster_amount
            size_t read_cluster_amount;
            V1File.read((char*)(&read_cluster_amount), sizeof(read_cluster_amount));
            total_df_v1 += actual_cluster_amount_v1[cluster_id] = read_cluster_amount;
        }

        // Load idf
        for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        {
            // Read idf
            float read_idf;
            V1File.read((char*)(&read_idf), sizeof(read_idf));
            idf_v1[cluster_id] = read_idf;
        }

        // Close file
        V1File.close();
    }
    cout << "done!" << endl;


    cout << "Read inverted header v2.."; cout.flush();
    // Load existing header V2
    size_t total_df_v2 = 0;
    size_t data_size_v2 = 0;
    size_t* actual_cluster_amount_v2 = new size_t[run_param.CLUSTER_SIZE];
    float* idf_v2 = new float[run_param.CLUSTER_SIZE];
    ifstream V2File ((run_param.inv_data_path + "_v2").c_str(), ios::binary);
    if (V2File)
    {
        // Load total_df
        V2File.read((char*)(&total_df_v2), sizeof(total_df_v2));

        // Load data_size_v2
        V2File.read((char*)(&data_size_v2), sizeof(data_size_v2));

        // Load cluster_amount
        for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            V2File.read((char*)(&actual_cluster_amount_v2[cluster_id]), sizeof(actual_cluster_amount_v2[cluster_id]));

        // Load idf
        for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            V2File.read((char*)(&idf_v2[cluster_id]), sizeof(idf_v2[cluster_id]));

        /*
        // Load written_word
        for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            V2File.read((char*)(&written_word[cluster_id]), sizeof(written_word[cluster_id]));
        */

        // Close file
        V2File.close();
    }
    cout << "done!" << endl;

    cout << "Checking integrity.."; cout.flush();
    if (total_df_v1 == total_df_v2)
    {
        // Check data size
        if (data_size_v1 == data_size_v2)
        {
            for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            {
                // Check cluster_amount
                if (actual_cluster_amount_v1[cluster_id] == actual_cluster_amount_v2[cluster_id])
                {

                }
                else
                    cout << "cluster_amount not equal " << actual_cluster_amount_v1[cluster_id] << " - " << actual_cluster_amount_v2[cluster_id] << endl;

                // Check idf
                if (idf_v1[cluster_id] == idf_v2[cluster_id])
                {

                }
                else
                    cout << setprecision(6) << fixed << "idf not equal " << idf_v1[cluster_id] << " - " << idf_v2[cluster_id] << endl;
            }
        }
        else
            cout << "data_size not equal " << data_size_v1 << " - " << data_size_v2 << endl;
    }
    else
        cout << "total_df not equal " << total_df_v1 << " - " << total_df_v2 << endl;
    cout << "done!" << endl;

    // Release memory
    delete[] actual_cluster_amount_v1;
    delete[] idf_v1;
    delete[] actual_cluster_amount_v2;
    delete[] idf_v2;
}

void PoolingTester()
{
    bow bow_tester;
    bow_tester.init(run_param);

    float* k1 = new float[5];
    float* k2 = new float[5];
    float* k3 = new float[5];
    float* k4 = new float[5];
    float* k5 = new float[5];
    float* k6 = new float[5];
    float* k7 = new float[5];
    float* k8 = new float[5];
    float* k9 = new float[5];
    int* index1 = new int[6];
    vector<float*> kp1;
    index1[0] = 10;
    index1[1] = 10;
    index1[2] = 10;
    index1[3] = 10;
    index1[4] = 55;
    index1[5] = 55;
    kp1.push_back(k1);
    kp1.push_back(k2);
    kp1.push_back(k3);
    kp1.push_back(k4);
    kp1.push_back(k5);
    kp1.push_back(k6);
    int* index2 = new int[3];
    vector<float*> kp2;
    index2[0] = 10;
    index2[1] = 55;
    index2[2] = 70;
    kp2.push_back(k7);
    kp2.push_back(k8);
    kp2.push_back(k9);

    bow_tester.build_bow(index1, kp1, 0);
    bow_tester.build_bow(index2, kp2, 0);

    vector<bow_bin_object*> bow_sig1 = bow_tester.get_bow_at(0);
    vector<bow_bin_object*> bow_sig2 = bow_tester.get_bow_at(1);

    bow_tester.build_pool(run_param.pooling_mode);

    vector<bow_bin_object*> bow_sig_pool = bow_tester.get_bow_pool_at(0);

    cout << "---- bow_sig1 ---- ";
    cout << "size():" << bow_sig1.size() << endl;
    for (size_t bin_id = 0; bin_id < bow_sig1.size(); bin_id++)
    {
        cout << "bin id:" << bin_id << "\t";
        cout << "cluster_id:" << bow_sig1[bin_id]->cluster_id << "\t";
        cout << "weight:" << bow_sig1[bin_id]->weight << "\t";
        cout << "features.size():" << bow_sig1[bin_id]->features.size() << endl;
    }
    cout << endl;

    cout << "---- bow_sig2 ---- ";
    cout << "size():" << bow_sig2.size() << endl;
    for (size_t bin_id = 0; bin_id < bow_sig2.size(); bin_id++)
    {
        cout << "bin id:" << bin_id << "\t";
        cout << "cluster_id:" << bow_sig2[bin_id]->cluster_id << "\t";
        cout << "weight:" << bow_sig2[bin_id]->weight << "\t";
        cout << "features.size():" << bow_sig2[bin_id]->features.size() << endl;
    }
    cout << endl;

    cout << "---- bow_sig_pool ---- ";
    cout << "size():" << bow_sig_pool.size() << endl;
    for (size_t bin_id = 0; bin_id < bow_sig_pool.size(); bin_id++)
    {
        cout << "bin id:" << bin_id << "\t";
        cout << "cluster_id:" << bow_sig_pool[bin_id]->cluster_id << "\t";
        cout << "weight:" << bow_sig_pool[bin_id]->weight << "\t";
        cout << "features.size():" << bow_sig_pool[bin_id]->features.size() << endl;
    }
    cout << endl;

    cout << "Inverted df check" << endl;
    ifstream iv_header_File (run_param.inv_header_path.c_str(), ios::binary);
    if (iv_header_File)
    {
        // Load total_df
        size_t total_df;
        iv_header_File.read((char*)(&total_df), sizeof(total_df));

        // Load dataset_size
        size_t dataset_size;
        iv_header_File.read((char*)(&dataset_size), sizeof(dataset_size));

        // Load cluster_amount
        size_t* actual_cluster_amount = new size_t[run_param.CLUSTER_SIZE];
        size_t tmp_read;
        for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
        {
            iv_header_File.read((char*)(&tmp_read), sizeof(tmp_read));
            actual_cluster_amount[cluster_id] = tmp_read;
        }

        vector< pair<size_t, size_t> > cluster_amount_sorted;
        for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            cluster_amount_sorted.push_back(pair<size_t, size_t>(cluster_id, actual_cluster_amount[cluster_id]));
        sort(cluster_amount_sorted.begin(), cluster_amount_sorted.end(), compare_pair_second<>());

        cout << "df histogram:" << endl;
        stringstream hist_string;
        for(size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            hist_string << cluster_amount_sorted[cluster_id].second << endl;
        text_write(run_param.offline_working_path + "/df.csv", hist_string.str());
    }
}

void ExportImgList()
{
    stringstream out_imglist;
    for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
    {
        stringstream dataset_path;
        dataset_path << ParentPaths[Img2ParentsIdx[img_idx]] << "/" << ImgLists[img_idx];
        out_imglist << dataset_path.str() << endl;
        percentout(img_idx, ImgLists.size(), 1);
    }

    text_write(run_param.offline_working_path + "/imglist.txt", out_imglist.str());
}
//;)
