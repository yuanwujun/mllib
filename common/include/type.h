// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef BASE_TYPE_H_
#define BASE_TYPE_H_ 
#include <map>
#include <set>
#include <string>
#include <vector>
#include <list>

typedef std::vector<int> VInt;
typedef const VInt VIntC;
typedef std::vector<VInt> VVInt;
typedef const VVInt VVIntC;
typedef double Real;
typedef std::vector<Real> VReal;
typedef const VReal VRealC;
typedef std::vector<VReal> VVReal;
typedef const VVReal VVRealC;
typedef std::vector<VVReal> VVVReal;
typedef const VVVReal VVVRealC;

typedef std::map<int, int> MIntInt;
typedef std::map<int, double> MIntReal;
typedef MIntReal::iterator MIntRealIt;
typedef MIntReal::const_iterator MIntRealCIt;

typedef std::map<int, VReal> MIntVReal;

typedef std::string Str;
typedef const Str StrC;
typedef std::vector<Str> VStr;
typedef const VStr VStrC;
typedef std::vector<VStr> VVStr;
typedef const VVStr VVStrC;

typedef std::list<Str> LStr;

typedef std::set<int> SInt;
typedef std::vector<SInt> VSInt;
typedef std::vector<MIntReal> VMIntReal;
#endif // BASE_TYPE_H_ 
