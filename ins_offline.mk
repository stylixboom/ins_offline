##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Release
ProjectName            :=ins_offline
ConfigurationName      :=Release
WorkspacePath          := "/home/stylix/Dropbox/MyDocument/SokendaiCourse/Researches/Workspace/code"
ProjectPath            := "/home/stylix/Dropbox/MyDocument/SokendaiCourse/Researches/Workspace/code/ins_offline"
IntermediateDirectory  :=./Release
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=Siriwat Kasamwattanarote
Date                   :=02/10/15
CodeLitePath           :="/home/stylix/Dropbox/MyDocument/SokendaiCourse/Researches/Workspace/configurations/.codelite"
LinkerName             :=g++
SharedObjectLinkerName :=g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=$(PreprocessorSwitch)NDEBUG 
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="ins_offline.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  `pkg-config opencv --libs` `pkg-config --libs lapacke`
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch)/home/stylix/local/include 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)ins $(LibrarySwitch)sifthesaff $(LibrarySwitch)alphautils $(LibrarySwitch)opencv_core $(LibrarySwitch)opencv_features2d $(LibrarySwitch)opencv_highgui $(LibrarySwitch)x264 $(LibrarySwitch)faac $(LibrarySwitch)lapacke $(LibrarySwitch)lapack $(LibrarySwitch)blas $(LibrarySwitch)tmglib $(LibrarySwitch)hdf5 $(LibrarySwitch)hdf5_hl_cpp $(LibrarySwitch)hdf5_cpp $(LibrarySwitch)hdf5_hl $(LibrarySwitch)mpi_cxx $(LibrarySwitch)mpi $(LibrarySwitch)rt $(LibrarySwitch)gomp $(LibrarySwitch)pthread $(LibrarySwitch)dl 
ArLibs                 :=  "libins.a" "libsifthesaff.a" "libalphautils.a" "opencv_core" "opencv_features2d" "opencv_highgui" "x264" "faac" "lapacke" "lapack" "blas" "tmglib" "hdf5" "hdf5_hl_cpp" "hdf5_cpp" "hdf5_hl" "mpi_cxx" "mpi" "rt" "gomp" "pthread" "dl" 
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch)/home/stylix/local/lib $(LibraryPathSwitch)../lib/ins/$(ConfigurationName) $(LibraryPathSwitch)../lib/sifthesaff/$(ConfigurationName) $(LibraryPathSwitch)../lib/alphautils/$(ConfigurationName) 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := ar rcu
CXX      := g++
CC       := gcc
CXXFLAGS :=  -O3 -fopenmp -std=c++11 -Wall `pkg-config --cflags opencv` $(Preprocessors)
CFLAGS   :=  -O2 -Wall $(Preprocessors)
ASFLAGS  := 
AS       := as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/ins_offline.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d "../.build-release/ransac" "../.build-release/sifthesaff" "../.build-release/alphautils" "../.build-release/ins" $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) @$(ObjectsFileList) $(LibPath) $(Libs) $(LinkOptions)

"../.build-release/ransac":
	@$(MakeDirCommand) "../.build-release"
	@echo stam > "../.build-release/ransac"


"../.build-release/sifthesaff":
	@$(MakeDirCommand) "../.build-release"
	@echo stam > "../.build-release/sifthesaff"


"../.build-release/alphautils":
	@$(MakeDirCommand) "../.build-release"
	@echo stam > "../.build-release/alphautils"


"../.build-release/ins":
	@$(MakeDirCommand) "../.build-release"
	@echo stam > "../.build-release/ins"




MakeIntermediateDirs:
	@test -d ./Release || $(MakeDirCommand) ./Release


$(IntermediateDirectory)/.d:
	@test -d ./Release || $(MakeDirCommand) ./Release

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/ins_offline.cpp$(ObjectSuffix): ins_offline.cpp $(IntermediateDirectory)/ins_offline.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/stylix/Dropbox/MyDocument/SokendaiCourse/Researches/Workspace/code/ins_offline/ins_offline.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/ins_offline.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/ins_offline.cpp$(DependSuffix): ins_offline.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/ins_offline.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/ins_offline.cpp$(DependSuffix) -MM "ins_offline.cpp"

$(IntermediateDirectory)/ins_offline.cpp$(PreprocessSuffix): ins_offline.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/ins_offline.cpp$(PreprocessSuffix) "ins_offline.cpp"


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Release/


