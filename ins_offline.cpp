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
            cout << "Command is invalid! It must be in a form of key value pair." << endl;
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
            cout << "[p] Packing extracted features" << endl;
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
            char pack_opt;
            bool by_block = false;
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
                cout << "Do you want to pack features block-by-block? [y|n] : "; cout.flush();
                cin >> pack_opt;
                if (pack_opt == 'y')
                {
                    by_block = true;
                    cout << "Block size (number of images) : "; cout.flush();
                    cin >> block_size;
                }
                PackFeature(by_block, block_size);
                break;
            case 'c':
                Clustering(save_cluster);
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
        //cout << "Total directory: " << ImgParentPaths.size() << endl;
        cout << "Total pool: " << ImgListsPoolIds[ImgListsPoolIds.size() - 1] + 1 << endl;
        cout << "Total image: " << ImgLists.size() << endl;

        // Check existing poolinfo
        if (is_path_exist(run_param.poolinfo_path))
        {
            LoadPoolinfo(run_param.poolinfo_path);
            cout << "Total features: " << total_features << endl;
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
        for (size_t parent_id = 0; parent_id < ImgParentPaths.size(); parent_id++)
            OutParentFile << parent_id << ":" << ImgParentPaths[parent_id] << endl;
            // parent_id:parent_path

        // Close file
        OutParentFile.close();
    }

    // Write image filename
    ofstream OutImgFile (run_param.dataset_filename_path.c_str()); // .dataset_file
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
                const char* delimsColon = ":";

                string_splitter(read_line, delimsColon, split_line);

                ImgParentPaths.push_back(split_line[1]);
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
        stringstream curr_img_path;
        stringstream curr_img_export_parent;
        stringstream curr_img_export_path;
        curr_img_parent << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
        curr_img_path << curr_img_parent.str() << "/" << ImgLists[img_idx];
        curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
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

        percentout(img_idx, ImgLists.size(), 1);

        // Sampling
        //img_idx += 5;
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
}

void PackFeature(bool by_block, size_t block_size)
{
    // Release memory
    total_features = 0;
    feature_count_per_pool.clear();
    feature_count_per_image.clear();

    // Feature space
    int row_size_dataset = 0;    // dataset size
    int col_size_dataset = 0;    // dataset dimension
    int col_kp_size = 0;         // keypoint dimension

    /// Load each image features
    // Checking image avalibality
    if (ImgLists.size() == 0)
    {
        cout << "No image available" << endl;
        return;
    }

    // ======== Loading dataset feature ========
    // SIFT extractor, loader
    bool check_sift_exist = true;  // Turn off for gaining a little speed
    SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift, check_sift_exist);
    col_kp_size = sifthesaff_dataset.GetSIFTHeadSize();
    col_size_dataset = sifthesaff_dataset.GetSIFTD();

    // Check existing poolinfo
    if (!is_path_exist(run_param.poolinfo_path))  // Not exist, create new pool
    {
        //==== Calculating feature size and pooling info
        cout << "Calculating dataset feature size..";
        cout.flush();
        startTime = CurrentPreciseTime();
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            //==== Path preparing
            stringstream curr_img_export_parent;
            stringstream curr_img_export_path;
            curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            curr_img_export_path << curr_img_export_parent.str() << "/" << ImgLists[img_idx] << ".sifthesaff";

            /// Loading features header
            int num_kp = sifthesaff_dataset.checkNumKp(curr_img_export_path.str(), true);

            /// Feature pooling
            if ((int)feature_count_per_pool.size() < ImgListsPoolIds[img_idx] + 1)
                feature_count_per_pool.push_back(0);
            feature_count_per_pool[ImgListsPoolIds[img_idx]] += num_kp; // Accumulating feature count in the pool
            feature_count_per_image.push_back(num_kp);                  // Keep feature for each image

            row_size_dataset += num_kp;
            //cout << sifthesaff_dataset.num_kp << " key points" << endl;

            percentout(img_idx, ImgLists.size(), 1);

            // Sampling
            //img_idx += 5;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Save Poolinfo
        SavePoolinfo(run_param.poolinfo_path);
    }
    else    // Exist, load existing pool
    {
        LoadPoolinfo(run_param.poolinfo_path);
        row_size_dataset = feature_count_per_image.size();
    }

    /// Packing
    if (by_block)
    {
        bool did_write = false;
        ///======== Loading and writing features to hdf5 block-by-block ========
        //==== Packing keypoint and descriptor
        size_t pack_rowDone = 0;
        size_t current_feature_count = 0;

        float* kp_dat = NULL;      // keypoint data
        float* desc_dat = NULL;    // descriptor data

        cout << "Packing dataset keypoint and descriptor..";
        cout.flush();
        startTime = CurrentPreciseTime();
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            // Accumulating features per block
            while (current_feature_count == 0)
            {
                for (size_t block_img_idx = img_idx, block_left = block_size; block_left > 0 && block_img_idx < ImgLists.size(); block_img_idx++, block_left--)
                {
                    //cout << "current_feature_count: " << current_feature_count << " feature_count_per_image[" << block_img_idx << "]: " << feature_count_per_image[block_img_idx] << endl;
                    current_feature_count += feature_count_per_image[block_img_idx];
                }
                // Skipping this block in case no feature extracted
                if (current_feature_count == 0)
                    img_idx += block_size;
                else
                {
                    kp_dat = new float[current_feature_count * col_kp_size];
                    desc_dat = new float[current_feature_count * col_size_dataset];
                }
            }

            //==== Path preparing
            stringstream curr_img_export_parent;
            stringstream curr_img_export_path;
            curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            curr_img_export_path << curr_img_export_parent.str() << "/" << ImgLists[img_idx] << ".sifthesaff";

            //cout << curr_img_export_path.str() << endl;


            /// Loading features
            sifthesaff_dataset.importKeypoints(curr_img_export_path.str(), true);

            /// Packing feature
            int currRow_size = sifthesaff_dataset.num_kp;

            // Debug
            if (currRow_size != feature_count_per_image[img_idx])
            {
                cout << "ERROR!! currRow_size != feature_count_per_image[img_idx]" << endl;
                cout << "currRow_size:" << currRow_size << " img_idx:" << img_idx << " feature_count_per_image[img_idx]:" << feature_count_per_image[img_idx] << endl;
                exit(0);
            }

            for(int row = 0; row < currRow_size; row++)
            {
                //desc_dat[prevDescBlock + currDesc + currCell]
                //== keypoint
                //cout << "[" << currRow_size << "] ";
                for(int col = 0; col < col_kp_size; col++)
                {
                    //cout << pack_rowDone * col_kp_size + row * col_kp_size + col << " ";
                    kp_dat[pack_rowDone * col_kp_size + row * col_kp_size + col] = sifthesaff_dataset.kp[row][col];
                }
                //cout << endl;

                //== descriptor
                for(int col = 0; col < col_size_dataset; col++)
                    desc_dat[pack_rowDone * col_size_dataset + row * col_size_dataset + col] = sifthesaff_dataset.desc[row][col];
            }
            pack_rowDone += currRow_size;


            /// Flushing out to HDF5 file if this block is full as accumulated in current_feature_count
            if (pack_rowDone == current_feature_count)
            {
                // Debug
                /*
                cout << "kp_dat[" << current_feature_count << "," << col_kp_size << "]" << endl;

                for(size_t row = 0; row < current_feature_count; row++)
                {
                    for(int col = 0; col < col_kp_size; col++)
                    {
                        cout << kp_dat[row * col_kp_size + col] << " ";
                        if(col == col_kp_size - 1)
                            cout << endl;
                    }
                }
                */

                /*cout << "Saving to HDF5 format..";
                cout.flush();
                startTime = CurrentPreciseTime();
                cout << "keypoint..";
                cout.flush();
                */
                HDF_write_append_2DFLOAT(run_param.feature_keypoint_path, did_write, "keypoint", kp_dat, current_feature_count, col_kp_size);
                /*cout << "descriptor..";
                cout.flush();
                */
                HDF_write_append_2DFLOAT(run_param.feature_descriptor_path, did_write, "descriptor", desc_dat, current_feature_count, col_size_dataset);
                /*cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
                */
                did_write = true;

                pack_rowDone = 0;
                current_feature_count = 0;

                delete[] kp_dat;
                delete[] desc_dat;
            }

            percentout(img_idx, ImgLists.size(), 1);

            // Sampling
            //img_idx += 5;
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
    {
        ///======== Loading all features then writing to HDF5 once ======== (not good for large dataset)
        size_t pack_rowDone = 0;

        float* kp_dat = new float[row_size_dataset * col_kp_size];          // keypoint dat
        float* desc_dat = new float[row_size_dataset * col_size_dataset];   // descriptor data

        //==== Loading and packing feature
        cout << "Packing dataset keypoint and descriptor..";
        cout.flush();
        startTime = CurrentPreciseTime();
        for (size_t img_idx = 0; img_idx < ImgLists.size(); img_idx++)
        {
            //==== Path preparing
            stringstream curr_img_export_parent;
            stringstream curr_img_export_path;
            curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[img_idx]];
            curr_img_export_path << curr_img_export_parent.str() << "/" << ImgLists[img_idx] << ".sifthesaff";

            //cout << curr_img_export_path.str() << endl;

            /// Loading features
            sifthesaff_dataset.importKeypoints(curr_img_export_path.str(), true);

            /// Packing feature
            int currRow_size = sifthesaff_dataset.num_kp;
            for(int row = 0; row != currRow_size; row++)
            {
                //desc_dat[prevDescBlock + currDesc + currCell]

                //== keypoint
                for(int col = 0; col != col_kp_size; col++)
                    kp_dat[pack_rowDone * col_kp_size + row * col_kp_size + col] = sifthesaff_dataset.kp[row][col];

                //== descriptor
                for(int col = 0; col != col_size_dataset; col++)
                    desc_dat[pack_rowDone * col_size_dataset + row * col_size_dataset + col] = sifthesaff_dataset.desc[row][col];
            }
            pack_rowDone += currRow_size;
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
        HDF_write_2DFLOAT(run_param.feature_keypoint_path, "keypoint", kp_dat, row_size_dataset, col_kp_size);
        cout << "descriptor..";
        cout.flush();
        HDF_write_2DFLOAT(run_param.feature_descriptor_path, "descriptor", desc_dat, row_size_dataset, col_size_dataset);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Release memory
        delete[] kp_dat;
        delete[] desc_dat;
    }

    cout << "Dataset contains " << ImgListsPoolIds[ImgListsPoolIds.size() - 1] + 1 << " pool(s) of image" << endl;
    cout << "Total " << row_size_dataset << " keypoint(s)" << endl;
}

void LoadFeature(size_t start_idx, size_t load_size, int load_mode)
{
    // Feature Header
    SIFThesaff sifthesaff_dataset(run_param.colorspace, run_param.normpoint, run_param.rootsift);
    int col_kp_size = sifthesaff_dataset.GetSIFTHeadSize();
    int col_size_dataset = sifthesaff_dataset.GetSIFTD();

    if (load_mode == LOAD_KP || load_mode == LOAD_ALL)  // Load with keypoint
    {
        delete[] dataset_keypoint.ptr();

        float* kp_dat;      // keypoint data
        //kp_dat = NULL;

        /*cout << "Loading feature keypoint..";
        cout.flush();
        startTime = CurrentPreciseTime();*/
        HDF_read_row_2DFLOAT(run_param.feature_keypoint_path, "keypoint", kp_dat, start_idx, load_size);
        /*cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
        */
        // Wrap keypoint to matrix for flann knn search
        Matrix<float> ret_keypoint(kp_dat, load_size, col_kp_size);

        /*
        cout << "kp [" << load_size << "," << 5 << "]" << endl;
        for (size_t row = 0; row < load_size; row++)
        {
            for (size_t col = 0; col < 5; col++)
                cout << kp_dat[row * 5 + col] << " ";
            cout << endl;
        }*/

        // Keep dataset pack
        swap(dataset_keypoint, ret_keypoint);
        // dataset_keypoint = ret_keypoint; // may have issue on memory leak
    }

    if (load_mode == LOAD_DESC || load_mode == LOAD_ALL)  // Load with desciptor
    {
        delete[] dataset_descriptor.ptr();

        float* desc_dat;    // descriptor data
        //desc_dat = NULL;

        /*cout << "Loading feature descriptor..";
        cout.flush();
        startTime = CurrentPreciseTime();*/
        HDF_read_row_2DFLOAT(run_param.feature_descriptor_path, "descriptor", desc_dat, start_idx, load_size);
        /*cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
        */

        // Wrap descriptor to matrix for flann knn search
        Matrix<float> ret_feature_vector(desc_dat, load_size, col_size_dataset);

        // Keep dataset pack
        swap(dataset_descriptor, ret_feature_vector);
        //dataset_descriptor = ret_feature_vector;  // may have issue on memory leak
    }
}

void SavePoolinfo(const string& out)
{
    //==== Save feature pooling data
    cout << "Saving pooling info..";
    cout.flush();
    startTime = CurrentPreciseTime();
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
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
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
        feature_count_per_pool.clear();
        feature_count_per_image.clear();

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

            /*
            if (feature_count == -1)
            {
                // Image
                stringstream curr_img_parent;
                stringstream curr_img_path;
                curr_img_parent << run_param.dataset_root_dir << "/" << ImgParentPaths[ImgParentsIdx[dataset_idx]];
                curr_img_path << curr_img_parent.str() << "/" << ImgLists[dataset_idx];

                // SIFT path
                stringstream curr_img_export_parent;
                stringstream curr_img_export_path;
                curr_img_export_parent << str_replace_first(run_param.dataset_root_dir, "dataset", "dataset_feature") << "/" << run_param.feature_name << "/" << ImgParentPaths[ImgParentsIdx[dataset_idx]];
                curr_img_export_path << curr_img_export_parent.str() << "/" << ImgLists[dataset_idx] << ".sifthesaff";

                string cmd = "/home/stylix/webstylix/code/lib/sifthesaff/bin/Release/sifthesaff_extractor -in " + curr_img_path.str() + " -out " + curr_img_export_path.str() + " -m b" << endl;
                //cout << curr_img_path.str() << endl;
                //cout << curr_img_export_path.str() << endl;
                cout << "Launch emergency database repair!!" << endl;
                cout << "cmd: " << cmd;
                exec(cmd);
            }
            */
        }

        // Close file
        PoolFile.close();

        if (pool_features != img_features)
        {
            cout << "Wrong poolinfo!! pool_features != img_features [ " << pool_features << " != " << img_features << "] " << endl;
            exit(-1);
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

string SamplingDatabase(int sample_size, int dimension)
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
    for (size_t img_id = 0; img_id < total_image; img_id++)
    {
        current_feature_amount = feature_count_per_image[img_id];

        if (sampling_mask[img_id])
        {
            float* desc_dat = new float[current_feature_amount * dimension];

            // Load from offset
            HDF_read_row_2DFLOAT(run_param.feature_descriptor_path, "descriptor", desc_dat, offset_feature_id, current_feature_amount);

            // Save append
            HDF_write_append_2DFLOAT(feature_descriptor_sample_path, did_write, "descriptor", desc_dat, current_feature_amount, dimension);
            did_write = true;

            // Accumulate total sampled features
            total_feature_amount += current_feature_amount;

            // Release memory
            delete[] desc_dat;

            percentout(img_id, total_image);
        }

        // Next offset
        offset_feature_id += current_feature_amount;
    }
    cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

    cout << "Total features: " << offset_feature_id << endl;
    cout << "Total sampled features: " << total_feature_amount << endl;

    feature_descriptor_sample_path = feature_descriptor_sample_path + "_feature-" + toString(total_feature_amount);
    exec("mv " + run_param.feature_descriptor_path + " " + feature_descriptor_sample_path);

    // Release memory
    delete[] sampling_mask;

    return feature_descriptor_sample_path;
}

void Clustering(bool save_cluster, bool hdf5)
{
    // Release memory
    delete[] cluster.ptr();
    actual_cluster_amount = 0;

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

        // SIFT detail
        SIFThesaff sift_obj;

        // Cluster preparation
        int km_cluster_size = run_param.CLUSTER_SIZE;
        int dimension = sift_obj.GetSIFTD();
        int km_iteration = 50;

        // Preparing empty cluster
        float* empty_cluster;

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
                string sampling;
                cout << "Recommend 35% of images = " << feature_count_per_image.size() / 35 << " , 35% of features = " << total_features / 35 << endl;
                cout << "Your preference: "; cout.flush();
                cin >> sampling;
                sample_size = atoi(sampling.c_str());

                // Convert feature to image based sampling
                if (opt == 'f')
                {
                    float sampling_ratio = total_features / sample_size;
                    sample_size = int(feature_count_per_image.size() / sampling_ratio);
                    cout << "approx. ~" << sample_size << " images" << endl;
                }

                run_param.feature_descriptor_path = SamplingDatabase(sample_size, dimension);
            }


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
                PyFile << "fastcluster.kmeans(\"" << run_param.cluster_path << "\", \"" << run_param.feature_descriptor_path << "\", " << km_cluster_size << ", " << km_iteration << ");" << endl;
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
            while (!is_path_exist(run_param.cluster_path))
            {
                ls2null(run_param.cluster_path);
                usleep(1000000); // 1 second
            }

            // Load written cluster
            LoadCluster(run_param.cluster_path);
        }
        else    // FLANN HKM ckustering
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
                SaveCluster(run_param.cluster_path);
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
        LoadCluster(run_param.cluster_path);
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

void ImageFeaturesQuantization(bool save_quantized)
{
    // Release memory
    ReleaseQuantizedDatasetBuffer();

    // Check existing poolinfo
    if (feature_count_per_pool.size() == 0)
        LoadPoolinfo(run_param.poolinfo_path);

    /// Build-load ANN search index
    //Index< ::flann::L2<float> > flann_search_index(AutotunedIndexParams(0.9, 0.01, 0.5, 1));
    Index< ::flann::L2<float> > flann_search_index(KDTreeIndexParams((int)run_param.KDTREE));

    // Check existing search index
    if (!is_path_exist(run_param.searchindex_path))
    {
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
        search_index.save(run_param.searchindex_path);
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Keep search index
        flann_search_index = search_index;
    }
    else // Load existing search index
    {
        cout << "Load FLANN search index..";
        cout.flush();
        startTime = CurrentPreciseTime();
        Index< ::flann::L2<float> > search_index(cluster, SavedIndexParams(run_param.searchindex_path)); // load index with provided dataset
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;

        // Keep search index
        flann_search_index = search_index;
    }

    /// Start ANN search
    SearchParams sparams = SearchParams();
    //sparams.checks = FLANN_CHECKS_AUTOTUNED;
    //sparams.checks = FLANN_CHECKS_UNLIMITED; // for only one tree
    sparams.checks = 512;               // Higher is better, also slower
    sparams.cores = run_param.MAXCPU;   // Take max CPU cores to run
    size_t knn = 1;                     // For hard assignment

    cout << "KNN searching..";
    cout.flush();
    startTime = CurrentPreciseTime();

    /// Per image vector quantization
    // Feature vector per image preparation
    size_t accumulative_feature_amount = 0;
    size_t current_feature_amount = 0;
    bool is_write = false;
    int quantization_buffer_limit = 200;
    int quantization_buffer_left = quantization_buffer_limit;
    //int dimension = dataset_descriptor.cols;
    /// Resuming checkpoint
    // Checking existing quantized data
    size_t resume_idx = 0;
    if (is_path_exist(run_param.quantized_path))
    {
        bool pass = true;
        // Checking quantized
        size_t quantized_count;
        size_t quantized_offset_count;

        ifstream QuantizedFile(run_param.quantized_path.c_str(), ios::binary | ios::in);
        ifstream QuantizedOffsetFile(run_param.quantized_offset_path.c_str(), ios::binary | ios::in);
        if (QuantizedFile.is_open() && QuantizedOffsetFile.is_open())
        {
            // Quantized count
            QuantizedFile.read((char*)(&quantized_count), sizeof(quantized_count));
            QuantizedOffsetFile.read((char*)(&quantized_offset_count), sizeof(quantized_offset_count));
            if (quantized_count == quantized_offset_count)
            {
                /// Checking actual data integrity
                // Reading last quantized_offset
                size_t last_quantized_offset_offset = sizeof(quantized_offset_count) + (quantized_offset_count - 1) * sizeof(size_t);
                size_t last_quantized_offset;
                QuantizedOffsetFile.seekg(last_quantized_offset_offset, QuantizedOffsetFile.beg);
                QuantizedOffsetFile.read((char*)(&last_quantized_offset), sizeof(last_quantized_offset));
                // Reading last quantized count
                size_t feature_count;
                QuantizedFile.seekg(last_quantized_offset, QuantizedFile.beg);
                QuantizedFile.read((char*)(&feature_count), sizeof(feature_count));

                if (int(feature_count) != feature_count_per_image[quantized_count - 1])
                {
                    cout << "Cannot resume, quantized file and quantized_offset file are not match." << endl;
                    pass = false;
                }
                else
                {
                    // Release previous memory
                    ReleaseQuantizedDatasetBuffer();

                    // Accumulating previous feature
                    for (size_t dataset_id = 0; dataset_id < quantized_count; dataset_id++)
                        accumulative_feature_amount += feature_count_per_image[dataset_id];

                    // Set resuming position
                    resume_idx = quantized_count;

                    // Flag for continue appending mode
                    is_write = true;
                }
            }
            else
            {
                cout << "Cannot resume, quantized file and quantized_offset file are not match." << endl;
                pass = false;
            }

            QuantizedOffsetFile.close();
            QuantizedFile.close();
        }

        if (!pass)
            return;

        cout << "Resuming at dataset_id:" << resume_idx << endl;
	}
	if (resume_idx < feature_count_per_image.size())
    {
        // Resuming from resume_idx
        for (size_t dataset_id = resume_idx; dataset_id < feature_count_per_image.size(); dataset_id++)
        {
            current_feature_amount = feature_count_per_image[dataset_id];

            // Old version
            /* Slice features per image from full-size dataset_descriptor to be quantized
            // Prepare feature vector to be quantized
            float* dataset_feature_idx = dataset_descriptor.ptr();
            float* current_feature = new float[current_feature_amount * dimension];
            for (int row = 0; row < current_feature_amount; row++)
                for (int col = 0; col < dimension; col++)
                    current_feature[row * dimension + col] = dataset_feature_idx[(accumulative_feature_amount + row) * dimension + col];
            */
            // New version
            // Load specific dataset_descriptor
            LoadFeature(accumulative_feature_amount, current_feature_amount, LOAD_DESC);

            // Accumulate offset of total feature per image for the next load
            accumulative_feature_amount += current_feature_amount;

            //Matrix<float> feature_data(current_feature, current_feature_amount, dimension);
            Matrix<int> result_index(new int[current_feature_amount * knn], current_feature_amount, knn); // size = feature_amount x knn
            Matrix<float> result_dist(new float[current_feature_amount * knn], current_feature_amount, knn);

            //flann_search_index.knnSearch(feature_data, result_index, result_dist, knn, sparams);
            flann_search_index.knnSearch(dataset_descriptor, result_index, result_dist, knn, sparams);

            // Debug
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

            // Check quantization buffer reach its limit, or reach the last images
            if (--quantization_buffer_left == 0 || dataset_id == feature_count_per_image.size() - 1)
            {
                // Flushing buffer to disk
                SaveQuantizedDataset(is_write);

                // Clear buffer
                ReleaseQuantizedDatasetBuffer();

                is_write = true;
                quantization_buffer_left = quantization_buffer_limit;
            }

            percentout(dataset_id, feature_count_per_image.size(), 1);

            // Release memory
            //delete[] feature_data.ptr();
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
        cout << "Quantizing not necessary. Dataset has been quantized." << endl;

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

void SaveQuantizedDataset(bool append)
{
    size_t quantized_count;
    size_t current_quantized_offset;

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
            quantized_count += dataset_quantized_indices.size();
            OutFile.write(reinterpret_cast<char*>(&quantized_count), sizeof(quantized_count));

            // Go to the end of stream
            OutFile.seekp(0, OutFile.end);
            current_quantized_offset = OutFile.tellp();
        }
        else
        {
            // Write at the beginning of stream for normal mode
            quantized_count = dataset_quantized_indices.size();
            OutFile.write(reinterpret_cast<char*>(&quantized_count), sizeof(quantized_count));

            // Start after read quantized_count
            current_quantized_offset = sizeof(quantized_count);
        }

        // Quantize index and distance
        for (size_t quantized_idx = 0; quantized_idx < dataset_quantized_indices.size(); quantized_idx++)
        {
            // Keep offset
            dataset_quantized_offset.push_back(current_quantized_offset);

            // Feature size
            size_t feature_count = dataset_quantized_indices[quantized_idx].size();
            OutFile.write(reinterpret_cast<char*>(&feature_count), sizeof(feature_count));
            current_quantized_offset += sizeof(feature_count);

            //cout << "feature_count: " << feature_count << endl;
            // Feature quantized index and distance to index
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                // Index
                int quantized_index = dataset_quantized_indices[quantized_idx][feature_idx];
                OutFile.write(reinterpret_cast<char*>(&quantized_index), sizeof(quantized_index));
                current_quantized_offset += sizeof(quantized_index);
                // Dist
                float quantized_dist = dataset_quantized_dists[quantized_idx][feature_idx];
                OutFile.write(reinterpret_cast<char*>(&quantized_dist), sizeof(quantized_dist));
                current_quantized_offset += sizeof(quantized_dist);

                //cout << "quantized_index: " << quantized_index << " quantized_dist: " << quantized_dist << endl;
            }
        }

        // Write offset
        bin_write_vector_SIZET(run_param.quantized_offset_path, dataset_quantized_offset, append);

        // Close file
        OutFile.close();
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

void LoadSpecificQuantizedDataset(size_t start_idx, size_t load_size)
{
    // Load quantized offset
    if (!dataset_quantized_offset_ready)
        LoadQuantizedDatasetOffset();

    // Release memory
    ReleaseQuantizedDatasetBuffer();

    ifstream InFile (run_param.quantized_path.c_str(), ios::binary);
    if (InFile)
    {
        /// Skip to quantized index of specific start_idx
        size_t curr_offset = dataset_quantized_offset[start_idx];
        InFile.seekg(curr_offset, InFile.beg);

        for (size_t dataset_quantized_idx = 0; dataset_quantized_idx < load_size; dataset_quantized_idx++)
        {
            // Feature size
            size_t feature_count;
            InFile.read((char*)(&feature_count), sizeof(feature_count));

            // Feature quantized index and distance to index
            vector<int> dataset_quantized_index;
            vector<float> dataset_quantized_dist;
            for (size_t feature_idx = 0; feature_idx < feature_count; feature_idx++)
            {
                // Index
                int index_data;
                InFile.read((char*)(&index_data), sizeof(index_data));
                dataset_quantized_index.push_back(index_data);
                // Dist
                float dist_data;
                InFile.read((char*)(&dist_data), sizeof(dist_data));
                dataset_quantized_dist.push_back(dist_data);

                //cout << "index_data: " << index_data << " dist_data:" << dist_data << endl;
            }
            dataset_quantized_indices.push_back(dataset_quantized_index);
            dataset_quantized_dists.push_back(dataset_quantized_dist);
            //cout << "dataset_quantized_indices.size(): " << dataset_quantized_indices.size() << endl;
        }

        // Close file
        InFile.close();
    }
}

void ReleaseQuantizedDatasetBuffer()
{
    // Release memory
    if (dataset_quantized_indices.size())
    {
        // Clear offset
        dataset_quantized_offset_ready = false;
        dataset_quantized_offset.clear();

        // Clear buffer
        for (size_t quantized_idx = 0; quantized_idx < dataset_quantized_indices.size(); quantized_idx++)
        {
            dataset_quantized_indices[quantized_idx].clear();
            dataset_quantized_dists[quantized_idx].clear();
        }
        dataset_quantized_indices.clear();
        dataset_quantized_dists.clear();
    }
}

void Bow(bool save_bow)
{
    // Checking existing bow file
    if (!is_path_exist(run_param.bow_path))
    {
        // Checking quantized dataset availability
        if (!is_path_exist(run_param.quantized_path))
        {
            cout << "No quantized dataset available, please run knn first" << endl;
            return;
        }

        /// Building Bow
        cout << "Building Bow..";
        // Create bow builder object
        bow bow_builder(run_param);
        cout.flush();
        startTime = CurrentPreciseTime();
        size_t accumulative_feature_amount = 0;
        int current_feature_amount = 0;
        int current_pool_feature_amount = 0;
        bool is_write = false;
        /// For each image
        for (size_t image_id = 0; image_id < feature_count_per_image.size(); image_id++)
        {
            current_feature_amount = feature_count_per_image[image_id];

            // Load each quantized index
            LoadSpecificQuantizedDataset(image_id);
            // Load each keypoint
            LoadFeature(accumulative_feature_amount, current_feature_amount, LOAD_KP);
            // Convert keypoint to vector< vector<float> >
            size_t dimension = dataset_keypoint.cols;
            float* dataset_keypoint_idx = dataset_keypoint.ptr();
            // Feature collection
            vector< vector<float> > features;
            for (int feature_id = 0; feature_id < current_feature_amount; feature_id++)
            {
                size_t feature_mem_idx = feature_id * dimension;

                // Create feature
                vector<float> feature;
                feature.push_back(dataset_keypoint_idx[feature_mem_idx + 0]);   // x
                feature.push_back(dataset_keypoint_idx[feature_mem_idx + 1]);   // y
                feature.push_back(dataset_keypoint_idx[feature_mem_idx + 2]);   // a
                feature.push_back(dataset_keypoint_idx[feature_mem_idx + 3]);   // b
                feature.push_back(dataset_keypoint_idx[feature_mem_idx + 4]);   // c

                // Keep new feature into its corresponding bin (cluster_id)
                features.push_back(feature);
            }
            accumulative_feature_amount += current_feature_amount;
            // Accumulate pool counter
            current_pool_feature_amount += current_feature_amount;

            // Build BoW
            bow_builder.build_bow(dataset_quantized_indices[0], features);

            // If total features reach total features in the pool, do pooling then flush to disk
            if (current_pool_feature_amount == feature_count_per_pool[ImgListsPoolIds[image_id]])
            {
                // Pooling from internal multiple bow
                bow_builder.build_pool();
                // Flush bow to disk
                bow_builder.flush_bow(is_write);
                // Flush bow_pool to disk
                bow_builder.flush_bow_pool(is_write);

                is_write = true;

                // Reset pool counter
                current_pool_feature_amount = 0;

                // Release memory
                bow_builder.reset_bow();
                bow_builder.reset_bow_pool();
            }

            percentout(image_id, feature_count_per_image.size(), 1);
        }
        cout << "done! (in " << setprecision(2) << fixed << TimeElapse(startTime) << " s)" << endl;
    }
    else
        cout << "Building BoW does not necessary. BoW has been built." << endl;
}

void build_invert_index()
{
    cout << "Build Invert Index" << endl;
    cout << "Path " << run_param.inv_path << endl;

    // Checking bag_of_word avalibality
    if (!is_path_exist(run_param.bow_path))
    {
        cout << "No bag of word available" << endl;
        return;
    }
    else
        cout << "BOW OK" << endl;

    // Provide invert index directory
    make_dir_available(run_param.inv_path);

    // Create invert index database
    invert_index invert_hist;
    invert_hist.reset();
    invert_hist.init(run_param.CLUSTER_SIZE, run_param.inv_path);

    cout << "Building invert index database..";
    // Create bow builder object (use as bow loader)
    bow bow_builder(run_param);
    cout.flush();
    startTime = CurrentPreciseTime();
    /// Building inverted index for each pool
    //cout << "feature_count_per_pool.size(): " << feature_count_per_pool.size() << endl;
    for (size_t pool_id = 0; pool_id < feature_count_per_pool.size(); pool_id++)
    {
        // Read existing bow
        vector<bow_bin_object> read_bow;
        bow_builder.load_specific_bow_pool(pool_id, read_bow);
        //cout << "pool_id: " << pool_id << " read_bow.size(): " << read_bow.size() << endl;

        /*cout << "cluster_id: ";
        for (size_t bin_idx = 0; bin_idx < read_bow.size(); bin_idx++)
            cout << read_bow[bin_idx].cluster_id << " ";
        cout << endl;*/

        // tf and normalize
        bow_builder.logtf_unitnormalize(read_bow);

        // Add to inverted hist
        invert_hist.add(pool_id, read_bow);

        percentout(pool_id, feature_count_per_pool.size(), 20);
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
