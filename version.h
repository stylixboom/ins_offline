#ifndef ins_offline_VERSION_H
#define ins_offline_VERSION_H

namespace ins_offline_AutoVersion{
	
	//Date Version Types
	static const char ins_offline_DATE[] = "21";
	static const char ins_offline_MONTH[] = "11";
	static const char ins_offline_YEAR[] = "2014";
	static const char ins_offline_UBUNTU_VERSION_STYLE[] =  "14.11";
	
	//Software Status
	static const char ins_offline_STATUS[] =  "Alpha";
	static const char ins_offline_STATUS_SHORT[] =  "a";
	
	//Standard Version Type
	static const long ins_offline_MAJOR  = 2;
	static const long ins_offline_MINOR  = 1;
	static const long ins_offline_BUILD  = 706;
	static const long ins_offline_REVISION  = 3529;
	
	//Miscellaneous Version Types
	static const long ins_offline_BUILDS_COUNT  = 1112;
	#define ins_offline_RC_FILEVERSION 2,1,706,3529
	#define ins_offline_RC_FILEVERSION_STRING "2, 1, 706, 3529\0"
	static const char ins_offline_FULLVERSION_STRING [] = "2.1.706.3529";
	
	//These values are to keep track of your versioning state, don't modify them.
	static const long ins_offline_BUILD_HISTORY  = 57;
	

}
#endif //ins_offline_VERSION_H
