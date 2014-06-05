#ifndef ins_offline_VERSION_H
#define ins_offline_VERSION_H

namespace ins_offline_AutoVersion{
	
	//Date Version Types
	static const char ins_offline_DATE[] = "05";
	static const char ins_offline_MONTH[] = "06";
	static const char ins_offline_YEAR[] = "2014";
	static const char ins_offline_UBUNTU_VERSION_STYLE[] =  "14.06";
	
	//Software Status
	static const char ins_offline_STATUS[] =  "Alpha";
	static const char ins_offline_STATUS_SHORT[] =  "a";
	
	//Standard Version Type
	static const long ins_offline_MAJOR  = 1;
	static const long ins_offline_MINOR  = 7;
	static const long ins_offline_BUILD  = 216;
	static const long ins_offline_REVISION  = 836;
	
	//Miscellaneous Version Types
	static const long ins_offline_BUILDS_COUNT  = 283;
	#define ins_offline_RC_FILEVERSION 1,7,216,836
	#define ins_offline_RC_FILEVERSION_STRING "1, 7, 216, 836\0"
	static const char ins_offline_FULLVERSION_STRING [] = "1.7.216.836";
	
	//These values are to keep track of your versioning state, don't modify them.
	static const long ins_offline_BUILD_HISTORY  = 67;
	

}
#endif //ins_offline_VERSION_H
