/*
 * ins_offline.cpp
 *
 *  Created on: July 11, 2013
 *      Author: Siriwat Kasamwattanarote
 */

#include "ins_offline.h"

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
            cout << "Command invalid! It must be in a form of key value pair." << endl;
            exit(1);
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
            cout << "[f] Loading extracted features" << endl;
            cout << "[c] Clustering feature" << endl;
            cout << "[v] Vector quantization" << endl;
            cout << "[b] Bag of word building" << endl;
            cout << "[i] Building invert index and save" << endl;
            cout << "Enter menu: ";
            cin >> menu;

            bool save_feature = true;
            bool save_cluster = true;
            bool save_quantized = true;
            bool save_bow = true;
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
            case 'f':
                LoadFeature();
                break;
            case 'c':
                Clustering(save_cluster);
                break;
            case 'v':
                DatasetQuantization(save_quantized);
                break;
            case 'b':
                Bow(save_bow);
                break;
            case 'i':
                build_invert_index();
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
    stringstream dataset_saved_path;
    stringstream dataset_saved_list;
    dataset_saved_path << run_param.database_root_dir << "/" << run_param.dataset_header;
    dataset_saved_list << dataset_saved_path.str() << "/dataset";
    if (!is_path_exist(dataset_saved_list.str() + "_basepath"))
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

        // ======== Load dataset ========
        cout << "Load dataset..";
        cout.flush();
        startTime = CurrentPreciseTime();

        vector<string> ImgDirQueue;

        //unsigned char isDir = 0x4; // DT_DIR
        unsigned char isFile = 0x8; // DT_REG
        char currDir[] = ".";
        char parentDir[] = "..";
        size_t currParentIdx = ImgParentPaths.size();
        int pool_id = -1;

        // First parent dir
        vector<int> ImgPoolLevels;
        ImgParentPaths.push_back(DatasetPath);
        ImgPoolLevels.push_back(0);

        while (currParentIdx < ImgParentPaths.size())
        {
            stringstream CurrDirectory;
            CurrDirectory << run_param.dataset_root_dir << "/" << ImgParentPaths[currParentIdx];

            // New temporary empty subdir
            vector<string> new_ImgParentPaths;
            vector<size_t> new_ImgPoolLevels;

            // Directory traverse
            bool first_file = false;
            DIR* dirp = opendir(CurrDirectory.str().c_str());
            dirent* dp;
            while ((dp = readdir(dirp)) != NULL)
            {
                //cout << static_cast<unsigned>(dp->d_type) << " " << static_cast<unsigned>(DT_REG) << endl;
                if (dp->d_type == isFile || string::npos != string(dp->d_name).find(".jpg") || string::npos != string(dp->d_name).find(".png") || string::npos != string(dp->d_name).find(".txt")) // Check file (and jpg file, error from some linux)
                {

                    //cout << "curr directory: " << CurrDirectory.str() << endl;
                    //cout << "file: " << string(dp->d_name) << endl;
                    // Filter preprocessed file
                    if (string::npos == string(dp->d_name).find("txt") && (run_param.dataset_preset_index != 0 && string::npos == string(dp->d_name).find(".png")) && (run_param.dataset_preset_index != 0 || string::npos == string(dp->d_name).find("edge"))) // filter out edge file for chars74k
                    {
                        ImgLists.push_back(string(dp->d_name));
                        ImgParentsIdx.push_back(currParentIdx); // lookup index to same parent

                        // Group dataset into pool_id according to the same pool level
                        if (run_param.group_level != -1 && ImgPoolLevels[currParentIdx] >= run_param.group_level)
                        {
                            if (!first_file && ImgPoolLevels[currParentIdx] == run_param.group_level) // current pool level is the same as group level
                            {
                                first_file = true;
                                ImgListsPoolIds.push_back(++pool_id);
                            }
                            else
                                ImgListsPoolIds.push_back(pool_id);
                        }
                        else // In case image is outside pool level
                        {
                            // If group pool level is -1, all the rest images will be in different pool level
                            if (run_param.group_level == -1)
                                ImgListsPoolIds.push_back(++pool_id);
                            else // if image is outside its pool level, just skip by -1
                                ImgListsPoolIds.push_back(-1);
                        }
                    }
                }
                else // Check directory
                {
                    //cout << "curr directory: " << CurrDirectory.str() << endl;
                    //cout << "directory: " << string(dp->d_name) << endl;
                    // Filter only child directory
                    if(strcmp(dp->d_name, currDir) && strcmp(dp->d_name, parentDir))
                    {
                        // Add sub-dir
                        stringstream SubDir;
                        SubDir << ImgParentPaths[currParentIdx] << "/" << dp->d_name;
                        new_ImgParentPaths.push_back(SubDir.str());
                        new_ImgPoolLevels.push_back(ImgPoolLevels[currParentIdx] + 1); // next level
                    }
                }
            }
            closedir(dirp);
            // Insert subdir at under its parent
            if (new_ImgParentPaths.size() > 0 && new_ImgPoolLevels.size() > 0)
            {
                ImgParentPaths.insert(ImgParentPaths.begin() + currParentIdx + 1, new_ImgParentPaths.begin(), new_ImgParentPaths.end());
                ImgPoolLevels.insert(ImgPoolLevels.begin() + currParentIdx + 1, new_ImgPoolLevels.begin(), new_ImgPoolLevels.end());
            }

            // Release mem
            new_ImgParentPaths.clear();
            new_ImgPoolLevels.clear();

            // Next parent
            currParentIdx++;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        cout << "Save dataset list..";
        cout.flush();
        startTime = CurrentPreciseTime();
        make_dir_available(dataset_saved_path.str()); // mkdir for dataset prefix name
        SaveDatasetList(dataset_saved_list.str());
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
    {
        cout << "Load dataset list..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadDatasetList(dataset_saved_list.str());
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }

    // Checking image avalibality
    if (ImgLists.size() > 0)
    {
        cout << "== Dataset information ==" << endl;
        cout << "Total directory: " << ImgParentPaths.size() << endl;
        cout << "Total image: " << ImgLists.size() << endl;
    }
    else
        cout << "No image available" << endl;
}

void SaveDatasetList(const string& out)
{
    // Write parent path (dataset based path)
    ofstream OutParentFile ((out + "_basepath").c_str());
    if (OutParentFile.is_open())
    {
        for (size_t parent_id = 0; parent_id < ImgParentPaths.size(); parent_id++)
            OutParentFile << parent_id << ":" << ImgParentPaths[parent_id] << endl;
            // parent_id:parent_path

        // Close file
        OutParentFile.close();
    }

    // Write image filename
    ofstream OutImgFile ((out + "_filename").c_str()); // .dataset_file
    if (OutImgFile.is_open())
    {
        // Write path to image
        // pool_id:parent_idx:path_to_image
        for (size_t image_id = 0; image_id < ImgLists.size(); image_id++)
            OutImgFile << ImgListsPoolIds[image_id] << ":" << ImgParentsIdx[image_id] << ":" << ImgLists[image_id] << endl;
            // parent_id:image_name

        // Close file
        OutImgFile.close();
    }
}

void LoadDatasetList(const string& in)
{
    // Read parent path (dataset based path)
    ifstream InParentFile ((in + "_basepath").c_str());
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
                const char* delimsColon = ":";

                string_splitter(read_line, delimsColon, split_line);

                ImgParentPaths.push_back(split_line[1]);
            }
        }

        // Close file
        InParentFile.close();
    }

    // Read image filename
    ifstream InImgFile ((in + "_filename").c_str());
    if (InImgFile)
    {
        string read_line;
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
                bool done_pool_id = false;
                for (cpos_end = 0; cpos_end < line_size; cpos_end++)
                {
                    if (read_line[cpos_end] == ':')
                    {
                        if (!done_pool_id)  // Pool id
                        {
                            ImgListsPoolIds.push_back(atoi(read_line.substr(cpos_start, cpos_end - cpos_start).c_str()));
                            cpos_start = cpos_end + 1;
                            done_pool_id = true;
                        }
                        else                // Parent id
                        {
                            ImgParentsIdx.push_back(atoi(read_line.substr(cpos_start, cpos_end - cpos_start).c_str()));
                            cpos_start = cpos_end + 1;
                            break;          // Stop search, the rest is image name
                        }
                    }
                }

                // Image name
                ImgLists.push_back(read_line.substr(cpos_start, line_size - cpos_start));
            }
        }

        // Close file
        InImgFile.close();
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
        curr_img_path << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[img_idx]] << "/" << ImgLists[img_idx];
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

    // ======== Extracting dataset feature ========
    cout << "Extracting dataset..";
    cout.flush();
    startTime = CurrentPreciseTime();

    SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift); // color, normalize, rootsift

    /* use with really good i/o, for network disk is not good enough
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
    size_t skip_id = 0;
    if (skip_to_file != "n")
    {
        for (skip_id = 0; skip_id < ImgLists.size(); skip_id++)
        {
            if (ImgLists[skip_id] == skip_to_file)
                break;
        }
    }

    // use with really good i/o, for network disk is not good enough
    //skip_id += machine_no;

    //bool toggle_skip = false;
    //bool start_ls2null = false;
    //for (size_t img_idx = skip_id + 1; img_idx < ImgLists.size(); img_idx += total_machine) // for network disk is not good enough
    for (size_t img_idx = skip_id + 1; img_idx < ImgLists.size(); img_idx++)
    {
        /*int skip_to_rand = 0;
        int max_skip = 32;
        if (max_skip > (int)(ImgLists.size() - img_idx))
            max_skip = (int)(ImgLists.size() - img_idx);
        if (toggle_skip)
            skip_to_rand = rand() % max_skip + img_idx;
        toggle_skip = !toggle_skip;*/

        stringstream curr_img_parent;
        stringstream curr_img_export_parent;
        stringstream curr_img_pre;
        stringstream curr_img_path;
        stringstream curr_img_export_path;
        /*if (run_param.dataset_preset_index == 0) // chars74k
            curr_img_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_tmp") << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
        else*/
            curr_img_parent << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
        curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
        /*if (run_param.dataset_preset_index == 0) // chars74k
            curr_img_pre << str_replace_first(ImgLists[img_idx], ".png", "-edge-pre.png");
        else*/
            curr_img_pre << ImgLists[img_idx];
        curr_img_path << curr_img_parent.str() << "/" << curr_img_pre.str();
        curr_img_export_path << curr_img_export_parent.str() << "/" << curr_img_pre.str() << ".sifthesaff";


        /// Parallel feature extraction can be done here

        //cout << curr_img_pre.str() << " ... "; cout.flush();
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

        percentout(img_idx, ImgLists.size(), 1);

        // Sampling
        //img_idx += 5;
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
}

void LoadFeature()
{
    stringstream poolinfo_path;
    stringstream feature_keypoint_path;
    stringstream feature_descriptor_path;
    poolinfo_path << run_param.database_root_dir << "/" << run_param.dataset_header << "/poolinfo";
    feature_keypoint_path << run_param.database_root_dir << "/" << run_param.dataset_header << "/feature_keypoint";
    feature_descriptor_path << run_param.database_root_dir << "/" << run_param.dataset_header << "/feature_descriptor";

    // Release memory
    delete[] dataset_keypoint.ptr();
    delete[] dataset_descriptor.ptr();
    feature_count_per_pool.clear();

    // Feature space
    size_t row_size_dataset = 0;    // dataset size
    size_t col_size_dataset = 0;    // dataset dimension
    size_t col_kp_size = 0;         // keypoint dimension

    float* kp_dat;      // keypoint data
    float* desc_dat;    // descriptor data

    // if never pack feature
    if (!is_path_exist(poolinfo_path.str()))
    {
        /// Load each image features
        // Checking image avalibality
        if (ImgLists.size() == 0)
        {
            cout << "No image available" << endl;
            return;
        }

        // ======== Loading dataset feature ========
        // SIFT extractor, loader
        SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift);
        col_kp_size = sifthesaff_dataset.GetSIFTHeadSize();
        col_size_dataset = sifthesaff_dataset.GetSIFTD();

        //==== Calculating feature size and pooling info
        cout << "Calculating dataset feature size..";
        cout.flush();
        startTime = CurrentPreciseTime();
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            //==== Path preparing
            stringstream curr_img_parent;
            stringstream curr_img_export_parent;
            stringstream curr_img_pre;
            //stringstream curr_img_path;
            stringstream curr_img_export_path;
            /*if (run_param.dataset_preset_index == 0) // chars74k
                curr_img_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_tmp") << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            else*/
                curr_img_parent << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            /*if (run_param.dataset_preset_index == 0) // chars74k
                curr_img_pre << str_replace_first(ImgLists[img_idx], ".png", "-edge-pre.png");
            else*/
                curr_img_pre << ImgLists[img_idx];
            //curr_img_path << curr_img_parent.str() << "/" << curr_img_pre.str();
            curr_img_export_path << curr_img_export_parent.str() << "/" << curr_img_pre.str() << ".sifthesaff";

            // Check existing
            if (is_path_exist(curr_img_export_path.str()))
            {
                /// Loading features header
                int num_kp = sifthesaff_dataset.checkNumKp(curr_img_export_path.str(), true);

                /// Feature pooling
                if ((int)feature_count_per_pool.size() < ImgListsPoolIds[img_idx] + 1)
                    feature_count_per_pool.push_back(0);
                feature_count_per_pool[ImgListsPoolIds[img_idx]] += num_kp; // Accumulating feature count in the pool
                feature_count_per_image.push_back(num_kp);                  // Keep feature for each image

                row_size_dataset += num_kp;
            }
            // Not exist
            else
            {
                cout << "Feature \"" << curr_img_export_path.str() << "\" does not exist!" << endl;
                exit(1);
            }
            //cout << sifthesaff_dataset.num_kp << " key points" << endl;

            percentout(img_idx, ImgLists.size(), 1);

            // Sampling
            //img_idx += 5;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        //==== Save feature pooling data
        cout << "Saving pooling info..";
        cout.flush();
        startTime = CurrentPreciseTime();
        ofstream PoolFile (poolinfo_path.str().c_str(), ios::binary);
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
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        //==== Packing keypoint and descriptor
        size_t pack_rowDone = 0;
        kp_dat = new float[row_size_dataset * col_kp_size];
        desc_dat = new float[row_size_dataset * col_size_dataset];

        //==== Loading and packing feature
        cout << "Packing dataset keypoint and descriptor..";
        cout.flush();
        startTime = CurrentPreciseTime();
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            //==== Path preparing
            stringstream curr_img_parent;
            stringstream curr_img_export_parent;
            stringstream curr_img_pre;
            //stringstream curr_img_path;
            stringstream curr_img_export_path;
            /*if (run_param.dataset_preset_index == 0) // chars74k
                curr_img_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_tmp") << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            else*/
                curr_img_parent << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            /*if (run_param.dataset_preset_index == 0) // chars74k
                curr_img_pre << str_replace_first(ImgLists[img_idx], ".png", "-edge-pre.png");
            else*/
                curr_img_pre << ImgLists[img_idx];
            //curr_img_path << curr_img_parent.str() << "/" << curr_img_pre.str();
            curr_img_export_path << curr_img_export_parent.str() << "/" << curr_img_pre.str() << ".sifthesaff";

            cout << curr_img_export_path.str() << endl;

            // Check existing
            if (is_path_exist(curr_img_export_path.str()))
            {
                /// Loading features
                sifthesaff_dataset.importKeypoints(curr_img_export_path.str(), true);

                /// Packing feature
                size_t currRow_size = sifthesaff_dataset.num_kp;
                for(size_t row = 0; row != currRow_size; row++)
                {
                    //desc_dat[prevDescBlock + currDesc + currCell]

                    //== keypoint
                    for(size_t col = 0; col != col_kp_size; col++)
                        kp_dat[pack_rowDone * col_kp_size + row * col_kp_size + col] = sifthesaff_dataset.kp[row][col];

                    //== descriptor
                    for(size_t col = 0; col != col_size_dataset; col++)
                        desc_dat[pack_rowDone * col_size_dataset + row * col_size_dataset + col] = sifthesaff_dataset.desc[row][col];
                }
                pack_rowDone += currRow_size;
            }
            // Not exist
            else
            {
                cout << "Feature \"" << curr_img_export_path.str() << "\" does not exist!" << endl;
                exit(1);
            }
            //cout << sifthesaff_dataset.num_kp << " key points" << endl;

            percentout(img_idx, ImgLists.size(), 1);

            // Sampling
            //img_idx += 5;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        cout << "Saving to HDF5 format..";
        cout.flush();
        startTime = CurrentPreciseTime();
        cout << "keypoint..";
        cout.flush();
        HDF_write_2DFLOAT(feature_keypoint_path.str(), "keypoint", kp_dat, row_size_dataset, col_kp_size);
        cout << "descriptor..";
        cout.flush();
        HDF_write_2DFLOAT(feature_descriptor_path.str(), "descriptor", desc_dat, row_size_dataset, col_size_dataset);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    // Load feature keypoint and feature descriptor from hdf5 file
    else
    {
        cout << "Loading pooling info..";
        cout.flush();
        startTime = CurrentPreciseTime();
        ifstream PoolFile (poolinfo_path.str().c_str(), ios::binary);
        if (PoolFile)
        {
            // Read pool_size
            size_t pool_size;
            PoolFile.read((char*)(&pool_size), sizeof(pool_size));

            // Read feature_count_per_pool
            for (size_t pool_idx = 0; pool_idx < pool_size; pool_idx++)
            {
                int feature_count;
                PoolFile.read((char*)(&feature_count), sizeof(feature_count));
                feature_count_per_pool.push_back(feature_count);
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
            }

            // Close file
            PoolFile.close();
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        kp_dat = NULL;
        desc_dat = NULL;

        cout << "Loading feature keypoint..";
        cout.flush();
        startTime = CurrentPreciseTime();
        HDF_read_2DFLOAT(feature_keypoint_path.str(), "keypoint", kp_dat, row_size_dataset, col_kp_size);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;


        cout << "Loading feature descriptor..";
        cout.flush();
        startTime = CurrentPreciseTime();
        HDF_read_2DFLOAT(feature_descriptor_path.str(), "descriptor", desc_dat, row_size_dataset, col_size_dataset);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }

    cout << "Dataset contains " << ImgListsPoolIds[ImgListsPoolIds.size() - 1] + 1 << " pool(s) of image" << endl;
    cout << "Total " << row_size_dataset << " keypoint(s)" << endl;

	// Wrap descriptor and keypoint to matrix for flann knn search
	Matrix<float> ret_keypoint(kp_dat, row_size_dataset, col_kp_size);
	Matrix<float> ret_feature_vector(desc_dat, row_size_dataset, col_size_dataset);

    // Keep dataset pack
    dataset_keypoint = ret_keypoint;
	dataset_descriptor = ret_feature_vector;
}

void Clustering(bool save_cluster, bool hdf5, bool runspawn, const string& out)
{
    // Release memory
    delete[] cluster.ptr();
    actual_cluster_amount = 0;

    string cluster_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/cluster";

    // Switch cluster output path to spawn out
    if (runspawn)
        cluster_path = out;

    // Check existing cluster
    if (!is_path_exist(cluster_path))
    {
        // Checking dataset packed avalibality
        if (dataset_descriptor.cols == 0){
            cout << "No dataset packed available" << endl;
            return;
        }

        char choice;
        bool with_mpi = true;
        cout << "==== CLuster configuration ====" << endl;
        cout << "Cluster size: " << run_param.CLUSTER_SIZE << endl;
        cout << "Dataset size: " << dataset_descriptor.rows << endl;
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
        int dimension = dataset_descriptor.cols;
        int km_iteration = 50;

        // Preparing empty cluster
        float* empty_cluster;

        if (hdf5) // MPI-akmeans
        {
            string feature_descriptor_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/feature_descriptor";
            string vgg_fastcluster_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/vgg_fastcluster.py";
            string start_clustering_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/start_clustering.sh";

            // Write vgg_fastcluster.py script
            /*
            #!/usr/bin/env python
            import time;
            import fastcluster;
            tic = time.time();
            fastcluster.kmeans("./cluster", "./feature_descriptor", km_cluster_size, km_iteration);
            toc = time.time();
            print "fastcluster.kmeans(dataset_header, km_cluster_size centers, km_iteration iterations) time: %.0f" %(toc-tic);
            */
            ofstream PyFile (vgg_fastcluster_path.c_str());
            if (PyFile.is_open())
            {
                // Script
                PyFile << "#!/usr/bin/env python" << endl;
                PyFile << "import time;" << endl;
                PyFile << "import fastcluster;" << endl;
                PyFile << "tic = time.time();" << endl;
                PyFile << "fastcluster.kmeans(\"" << cluster_path << "\", \"" << feature_descriptor_path << "\", " << km_cluster_size << ", " << km_iteration << ");" << endl;
                PyFile << "toc = time.time();" << endl;
                PyFile << "print \"fastcluster.kmeans(" << run_param.dataset_header << ", " << km_cluster_size << " centers, " << km_iteration << " iterations) time: %.0f\" %(toc-tic);" << endl;

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
            while (!is_path_exist(cluster_path))
            {
                ls2null(cluster_path);
                usleep(1000000); // 1 second
            }

            // Load written cluster
            LoadCluster(cluster_path);
        }
        else
        {
            // Flann k-mean param
            int k = 96;
            int km_branching = ((km_cluster_size - 1) / k + 1) + 1; // last +2 for higher than hierarchicalClustering tree cut method
            //int km_iteration = -1; // iterate until convergence
            KMeansIndexParams km_params(km_branching, km_iteration, FLANN_CENTERS_KMEANSPP);

            // Clustering
            // Prepare memory
            empty_cluster = new float[km_cluster_size * dimension];
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
                SaveCluster(cluster_path);
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            // Keep cluster
            delete[] cluster.ptr();
            cluster = new_cluster;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
        cout << actual_cluster_amount << " cluster(s) found" << endl;
    }
    else
    {
        // Load existing cluster
        cout << "Loading cluster..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadCluster(cluster_path);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
}

void SaveCluster(const string& out)
{
    // Check directory
    string dir_name = get_directory(out);
    make_dir_available(dir_name);

    size_t cluster_dimension = cluster.cols;
    //size_t cluster_amount = actual_cluster_amount;
    size_t cluster_amount = run_param.CLUSTER_SIZE; // in case actual cluster is not equal as expected

    // Save to HDF5
    HDF_write_2DFLOAT(out, "clusters", cluster.ptr(), cluster_amount, cluster_dimension);
}

void LoadCluster(const string& in)
{
    // Release memory
    delete[] cluster.ptr();

    size_t cluster_amount;      // Cluster size
    size_t cluster_dimension;   // Feature dimension

    // Get HDF5 header
    HDF_get_2Ddimension(in, "clusters", cluster_amount, cluster_dimension);

    // Wrap data to maxrix for flann knn searching
    float* empty_cluster = new float[cluster_amount * cluster_dimension];

    // Read from HDF5
    HDF_read_2DFLOAT(in, "clusters", empty_cluster, cluster_amount, cluster_dimension);

    Matrix<float> new_cluster(empty_cluster, cluster_amount, cluster_dimension);

    // Keep cluster
    cluster = new_cluster;
    actual_cluster_amount = cluster_amount;
}

void DatasetQuantization(bool save_quantized)
{
    string dataset_quantized_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/quantized";
    string search_index_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/searchindex";

    // Release memory
    dataset_quantized_indices.clear();
    dataset_quantized_dists.clear();

    // Checking existing quantized data
    if (!is_path_exist(dataset_quantized_path))
    {
        //Index< ::flann::L2<float> > flann_search_index(AutotunedIndexParams(0.9, 0.01, 0.5, 1));
        Index< ::flann::L2<float> > flann_search_index(KDTreeIndexParams((int)run_param.KDTREE));

        // Check existing search index
        if (!is_path_exist(search_index_path))
        {
            // Checking dataset packed avalibality
            if (dataset_descriptor.cols == 0){
                cout << "No dataset packed available" << endl;
                return;
            }
            // Checking exising cluster
            if (cluster.cols == 0){
                cout << "No cluster loaded or processed" << endl;
                return;
            }

            //Index< ::flann::L2<float> > search_index(AutotunedIndexParams(0.9, 0.01, 0.5, 1));
            Index< ::flann::L2<float> > search_index(KDTreeIndexParams((int)run_param.KDTREE));

            // Building search index
            cout << "Build FLANN search index..";
            cout.flush();
            startTime = CurrentPreciseTime();
            search_index.buildIndex(cluster);
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            // Saving search index
            cout << "Saving index..";
            cout.flush();
            startTime = CurrentPreciseTime();
            search_index.save(search_index_path);
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            // Keep search index
            flann_search_index = search_index;
        }
        else // Load existing search index
        {
            cout << "Load FLANN search index..";
            cout.flush();
            startTime = CurrentPreciseTime();
            Index< ::flann::L2<float> > search_index(cluster, SavedIndexParams(search_index_path)); // load index with provided dataset
            cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

            // Keep search index
            flann_search_index = search_index;
        }

        // KNN search
        SearchParams sparams = SearchParams();
        //sparams.checks = FLANN_CHECKS_AUTOTUNED;
        //sparams.checks = FLANN_CHECKS_UNLIMITED; // for only one tree
        sparams.checks = 512;
        sparams.cores = run_param.MAXCPU;
        size_t knn = 1;

        cout << "KNN searching..";
        cout.flush();
        startTime = CurrentPreciseTime();

        // Feature vector per dataset preparation
        size_t accumulative_feature_amount = 0;
        size_t current_feature_amount = 0;
        size_t dimension = dataset_descriptor.cols;
        float* dataset_feature_idx = dataset_descriptor.ptr();
        for (size_t dataset_id = 0; dataset_id < feature_count_per_pool.size(); dataset_id++)
        {
            current_feature_amount = feature_count_per_pool[dataset_id];
            // Prepare feature vector to be quantized
            float* current_feature = new float[current_feature_amount * dimension];
            for (size_t row = 0; row < current_feature_amount; row++)
            {
                for (size_t col = 0; col < dimension; col++)
                {
                    current_feature[row * dimension + col] = dataset_feature_idx[(accumulative_feature_amount + row) * dimension + col];
                    //cout << dataset_feature_idx[(accumulative_feature_amount + row) * dimension + col] << " ";
                }
            }
            accumulative_feature_amount += current_feature_amount;

            Matrix<float> feature_data(current_feature, current_feature_amount, dimension);
            Matrix<int> result_index(new int[current_feature_amount * knn], current_feature_amount, knn); // size = feature_amount x knn
            Matrix<float> result_dist(new float[current_feature_amount * knn], current_feature_amount, knn);

            flann_search_index.knnSearch(feature_data, result_index, result_dist, knn, sparams);

            /*for(size_t row = 0; row < 3; row++)
            {
                for(size_t col = 0; col < dimension; col++)
                {
                    cout << feature_data.ptr()[row * dimension + col] << " ";
                    if(col == dimension - 1)
                        cout << endl;
                }
            }*/

            // Keep result
            int* result_index_idx = result_index.ptr();
            float* result_dist_idx = result_dist.ptr();
            // row base result
            // col is nn number
            for(size_t col = 0; col < knn; col++)
            {
                vector<int> result_index_vector;
                vector<float> result_dist_vector;
                for(size_t row = 0; row < result_index.rows; row++)
                {
                    result_index_vector.push_back(result_index_idx[row * knn + col]);
                    result_dist_vector.push_back(result_dist_idx[row * knn + col]);
                }
                dataset_quantized_indices.push_back(result_index_vector);
                dataset_quantized_dists.push_back(result_dist_vector);
            }

            percentout(dataset_id, feature_count_per_pool.size(), 1);

            // Release memory
            delete[] feature_data.ptr();
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Save quantized data
        cout << "Saving quantized data..";
        cout.flush();
        startTime = CurrentPreciseTime();
        if (save_quantized)
            SaveQuantizedDataset(dataset_quantized_path);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
	}
	else
	{
        // Load existing quantized data
        cout << "Loading quantized data..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadQuantizedDataset(dataset_quantized_path);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
	}

/*
    // Preview
    for(size_t query_idx = 0; query_idx < 2; query_idx++)
    {
        for(size_t result_idx = 0; result_idx < dataset_quantized_indices[query_idx].size(); result_idx++)
        {
            cout << dataset_quantized_indices[query_idx][result_idx] << " ";
        }
        cout << endl;
    }
*/
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

void SaveQuantizedDataset(const string& out)
{
    // Check directory
    string dir_name = get_directory(out);
    make_dir_available(dir_name);

    ofstream OutFile (out.c_str(), ios::binary);
    if (OutFile.is_open())
    {
        // Dataset size
        size_t dataset_count = dataset_quantized_indices.size();
        OutFile.write(reinterpret_cast<char*>(&dataset_count), sizeof(dataset_count));

        // Indices
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Feature size
            size_t feature_count = dataset_quantized_indices[dataset_id].size();
            OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));

            // Feature quantized index data
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
                OutFile.write(reinterpret_cast<char*>(&dataset_quantized_indices[dataset_id][feature_idx]), sizeof(dataset_quantized_indices[dataset_id][feature_idx]));
        }

        // Dists
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Feature size
            size_t feature_count = dataset_quantized_dists[dataset_id].size();
            OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));

            // Feature quantized dist data
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
                OutFile.write(reinterpret_cast<char*>(&dataset_quantized_dists[dataset_id][feature_idx]), sizeof(dataset_quantized_dists[dataset_id][feature_idx]));
        }

        // Close file
        OutFile.close();
    }
}

void LoadQuantizedDataset(const string& in)
{
    ifstream InFile (in.c_str(), ios::binary);
    if (InFile)
    {
        // Dataset size
        size_t dataset_count;
        InFile.read((char*)(&dataset_count), sizeof(dataset_count));

        // Indices
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Feature size
            size_t feature_count;
            InFile.read((char*)(&feature_count), sizeof(feature_count));

            // Feature quantized index data
            vector<int> dataset_quantized_index;
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                int index_data;
                InFile.read((char*)(&index_data), sizeof(index_data));
                dataset_quantized_index.push_back(index_data);
            }
            dataset_quantized_indices.push_back(dataset_quantized_index);
        }

        // Dists
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Feature size
            size_t feature_count;
            InFile.read((char*)(&feature_count), sizeof(feature_count));

            // Feature quantized dist data
            vector<float> dataset_quantized_dist;
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                float dist_data;
                InFile.read((char*)(&dist_data), sizeof(dist_data));
                dataset_quantized_dist.push_back(dist_data);
            }
            dataset_quantized_dists.push_back(dataset_quantized_dist);
        }

        // Close file
        InFile.close();
    }
}

void Bow(bool save_bow)
{
    string bow_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/bow";

    // Release memory
    if (bag_of_word.size() > 0)
    {
        for (size_t dataset_id = 0; dataset_id < bag_of_word.size(); dataset_id++)
        {
            for (size_t bin_id = 0; bin_id < bag_of_word[dataset_id].size(); bin_id++)
                bag_of_word[dataset_id][bin_id].features.clear();
            bag_of_word[dataset_id].clear();
        }
        bag_of_word.clear();
    }

    // Checking existing bow file
    if (!is_path_exist(bow_path))
    {
        // Checking dataset keypoint avalibality
        if (dataset_keypoint.cols == 0)
        {
            cout << "No dataset keypoint available" << endl;
            return;
        }
        // Checking quantized dataset avalibality
        if (dataset_quantized_indices.size() == 0)
        {
            cout << "No quantized dataset available" << endl;
            return;
        }

        // Bow gen
        cout << "Building Bow..";
        cout.flush();
        startTime = CurrentPreciseTime();
        size_t accumulative_kp_amount = 0;
        size_t current_kp_amount = 0;
        size_t dimension = dataset_keypoint.cols;
        float* dataset_keypoint_idx = dataset_keypoint.ptr();
        for (size_t dataset_id = 0; dataset_id < dataset_quantized_indices.size(); dataset_id++)
        {
            // Initialize blank sparse bow
            unordered_map<size_t, vector<feature_object> > curr_sparse_bow; // sparse of feature

            // Set bow
            // Add feature to curr_sparse_bow at cluster_id
            // Frequency of bow is curr_sparse_bow[].size()
            current_kp_amount = feature_count_per_pool[dataset_id];
            for (size_t feature_id = 0; feature_id < dataset_quantized_indices[dataset_id].size(); feature_id++)
            {
                // Get cluster from quantizad index of feature
                size_t cluster_id = dataset_quantized_indices[dataset_id][feature_id];

                // Create new feature object with feature_id and geo location, x,y,a,b,c
                feature_object feature;
                feature.feature_id = feature_id;
                feature.x = dataset_keypoint_idx[(accumulative_kp_amount + feature_id) * dimension + 0];
                feature.y = dataset_keypoint_idx[(accumulative_kp_amount + feature_id) * dimension + 1];
                feature.a = dataset_keypoint_idx[(accumulative_kp_amount + feature_id) * dimension + 2];
                feature.b = dataset_keypoint_idx[(accumulative_kp_amount + feature_id) * dimension + 3];
                feature.c = dataset_keypoint_idx[(accumulative_kp_amount + feature_id) * dimension + 4];

                // Keep new feature into its corresponding bin (cluster_id)
                curr_sparse_bow[cluster_id].push_back(feature);

            }
            accumulative_kp_amount += current_kp_amount;

            /// Make compact bow, with tf frequency
            vector<bow_bin_object> curr_compact_bow;
            for (size_t cluster_id = 0; cluster_id < run_param.CLUSTER_SIZE; cluster_id++)
            {
                // Looking for non-zero bin of cluster,
                // then put that bin together with specified cluster_id
                if (curr_sparse_bow[cluster_id].size())
                {
                    // Create new bin with cluster_id, frequency, and its features
                    bow_bin_object bow_bin;
                    bow_bin.cluster_id = cluster_id;

                    // tf
                    float feature_weight = 0.0f;
                    if (curr_sparse_bow[cluster_id].size() > 0)
                        feature_weight = 1 + log10(curr_sparse_bow[cluster_id].size()); // tf = 1 + log10(freq)                             # better
                        //feature_weight = curr_sparse_bow[cluster_id].size() / current_kp_amount; // tf = freq / total_word_for_this_doc

                    bow_bin.freq = feature_weight;
                    bow_bin.features = curr_sparse_bow[cluster_id];

                    // Keep new bin into compact_bow
                    curr_compact_bow.push_back(bow_bin);
                }
            }

            /// Normalization
            // Unit length
            float sum_of_square = 0.0f;
            float unit_length = 0.0f;
            for (size_t bin_idx = 0; bin_idx < curr_compact_bow.size(); bin_idx++)
                sum_of_square += curr_compact_bow[bin_idx].freq * curr_compact_bow[bin_idx].freq;
            unit_length = sqrt(sum_of_square);

            // Normalizing
            for (size_t bin_idx = 0; bin_idx < curr_compact_bow.size(); bin_idx++)
                curr_compact_bow[bin_idx].freq = curr_compact_bow[bin_idx].freq / unit_length;

            // Keep compact bow together, to make bag of words
            bag_of_word.push_back(curr_compact_bow);

            percentout(dataset_id, dataset_quantized_indices.size(), 1);
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Save Bow
        cout << "Saving Bow..";
        cout.flush();
        startTime = CurrentPreciseTime();
        if (save_bow)
            SaveBow(bow_path);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    }
    else
    {
        // Load Bow
        cout << "Loading Bow..";
        cout.flush();
        startTime = CurrentPreciseTime();
        LoadBow(bow_path);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
}

void SaveBow(const string& out)
{
    // Check directory
    string dir_name = get_directory(out);
    make_dir_available(dir_name);

    ofstream OutFile (out.c_str(), ios::binary);
    if (OutFile.is_open())
    {
        // Dataset size
        size_t dataset_count = bag_of_word.size();
        OutFile.write(reinterpret_cast<char*>(&dataset_count), sizeof(dataset_count));

        // Prepare offset array
        vector<size_t> bow_offset;

        // Curr Offset
        // Start after read dataset_count
        size_t curr_offset = sizeof(dataset_count);

        // Bow hist
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Keep offset
            bow_offset.push_back(curr_offset);

            // Dataset ID
            OutFile.write(reinterpret_cast<char*>(&dataset_id), sizeof(dataset_id));
            curr_offset += sizeof(dataset_id);

            // Non-zero count
            size_t bin_count = bag_of_word[dataset_id].size();
            OutFile.write(reinterpret_cast<char*>(&bin_count), sizeof(bin_count));
            curr_offset += sizeof(bin_count);

            // ClusterID
            for (size_t bin_id = 0; bin_id < bin_count; bin_id++)
            {
                // Cluster ID
                size_t cluster_id = bag_of_word[dataset_id][bin_id].cluster_id;
                OutFile.write(reinterpret_cast<char*>(&cluster_id), sizeof(cluster_id));
                curr_offset += sizeof(cluster_id);

                // Frequency
                float freq = bag_of_word[dataset_id][bin_id].freq;
                OutFile.write(reinterpret_cast<char*>(&freq), sizeof(freq));
                curr_offset += sizeof(freq);

                // Feature Count
                size_t feature_count = bag_of_word[dataset_id][bin_id].features.size();
                OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
                curr_offset += sizeof(feature_count);
                for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
                {
                    // Write all features from bin
                    feature_object feature = bag_of_word[dataset_id][bin_id].features[bow_feature_id];
                    // Feature ID
                    OutFile.write(reinterpret_cast<char*>(&(feature.feature_id)), sizeof(feature.feature_id));
                    curr_offset += sizeof(feature.feature_id);
                    // x
                    OutFile.write(reinterpret_cast<char*>(&(feature.x)), sizeof(feature.x));
                    curr_offset += sizeof(feature.x);
                    // y
                    OutFile.write(reinterpret_cast<char*>(&(feature.y)), sizeof(feature.y));
                    curr_offset += sizeof(feature.y);
                    // a
                    OutFile.write(reinterpret_cast<char*>(&(feature.a)), sizeof(feature.a));
                    curr_offset += sizeof(feature.a);
                    // b
                    OutFile.write(reinterpret_cast<char*>(&(feature.b)), sizeof(feature.b));
                    curr_offset += sizeof(feature.b);
                    // c
                    OutFile.write(reinterpret_cast<char*>(&(feature.c)), sizeof(feature.c));
                    curr_offset += sizeof(feature.c);
                }
            }
        }

        // Write offset
        bin_write_vector_SIZET(out + "_offset", bow_offset);

        // Close file
        OutFile.close();
    }
}

void LoadBow(const string& in)
{
    ifstream InFile (in.c_str(), ios::binary);
    if (InFile)
    {
        // Dataset size
        size_t dataset_count;
        InFile.read((char*)(&dataset_count), sizeof(dataset_count));

        // Bow hist
        for (size_t dataset_id = 0; dataset_id < dataset_count; dataset_id++)
        {
            // Dataset ID (read, but not use)
            size_t dataset_id_read;
            InFile.read((char*)(&dataset_id_read), sizeof(dataset_id_read));

            // Dataset bow
            vector<bow_bin_object> read_bow;

            // Non-zero count
            size_t bin_count;
            InFile.read((char*)(&bin_count), sizeof(bin_count));

            // ClusterID and FeatureIDs
            for (size_t bin_idx = 0; bin_idx < bin_count; bin_idx++)
            {
                bow_bin_object read_bin;

                // Cluster ID
                InFile.read((char*)(&(read_bin.cluster_id)), sizeof(read_bin.cluster_id));

                // Frequency
                InFile.read((char*)(&(read_bin.freq)), sizeof(read_bin.freq));

                // Feature count
                size_t feature_count;
                InFile.read((char*)(&feature_count), sizeof(feature_count));
                for (size_t bow_feature_id = 0; bow_feature_id < feature_count; bow_feature_id++)
                {
                    feature_object feature;

                    // Feature ID
                    InFile.read((char*)(&(feature.feature_id)), sizeof(feature.feature_id));
                    // x
                    InFile.read((char*)(&(feature.x)), sizeof(feature.x));
                    // y
                    InFile.read((char*)(&(feature.y)), sizeof(feature.y));
                    // a
                    InFile.read((char*)(&(feature.a)), sizeof(feature.a));
                    // b
                    InFile.read((char*)(&(feature.b)), sizeof(feature.b));
                    // c
                    InFile.read((char*)(&(feature.c)), sizeof(feature.c));

                    read_bin.features.push_back(feature);
                }

                // Keep bow
                read_bow.push_back(read_bin);
            }

            // Keep hist
            bag_of_word.push_back(read_bow);
        }

        // Close file
        InFile.close();
    }
}

void build_invert_index()
{
    cout << "Build Invert Index" << endl;
    string inv_path = run_param.database_root_dir + "/" + run_param.dataset_header + "/invdata_" + run_param.dataset_header;
    cout << "Path " << inv_path << endl;

    // Checking bag_of_word avalibality
    if (bag_of_word.size() == 0)
    {
        cout << "No bag of word available" << endl;
        return;
    }
    else
        cout << "BOW OK" << endl;

    // Provide invert index directory
    make_dir_available(inv_path);

    // Create invert index database
    invert_index invert_hist;
    invert_hist.reset();
    invert_hist.init(run_param.CLUSTER_SIZE, inv_path);

    cout << "Building invert index database..";
    cout.flush();
    startTime = CurrentPreciseTime();
    for (size_t dataset_id = 0; dataset_id < bag_of_word.size(); dataset_id++)
    {
        invert_hist.add(dataset_id, bag_of_word[dataset_id]);
        percentout(dataset_id, bag_of_word.size(), 20);
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    cout << "Updating idf..";
    cout.flush();
    startTime = CurrentPreciseTime();
    invert_hist.update_idf();
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    cout << "Saving invert index database..";
    cout.flush();
    startTime = CurrentPreciseTime();
    invert_hist.save_invfile();
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    // Release mem
    invert_hist.reset();
}
//;)
