#ifndef ins_offline_VERSION_H
#define ins_offline_VERSION_H

namespace ins_offline_AutoVersion{
	
	//Date Version Types
	static const char ins_offline_DATE[] = "09";
	static const char ins_offline_MONTH[] = "05";
	static const char ins_offline_YEAR[] = "2014";
	static const char ins_offline_UBUNTU_VERSION_STYLE[] =  "14.05";
	
	//Software Status
	static const char ins_offline_STATUS[] =  "Alpha";
	static const char ins_offline_STATUS_SHORT[] =  "a";
	
	//Standard Version Type
	static const long ins_offline_MAJOR  = 1;
	static const long ins_offline_MINOR  = 6;
	static const long ins_offline_BUILD  = 58;
	static const long ins_offline_REVISION  = 59;
	
	//Miscellaneous Version Types
	static const long ins_offline_BUILDS_COUNT  = 18;
	#define ins_offline_RC_FILEVERSION 1,6,58,59
	#define ins_offline_RC_FILEVERSION_STRING "1, 6, 58, 59\0"
	static const char ins_offline_FULLVERSION_STRING [] = "1.6.58.59";
	
	//These values are to keep track of your versioning state, don't modify them.
	static const long ins_offline_BUILD_HISTORY  = 9;
	

}
#endif //ins_offline_VERSION_H
