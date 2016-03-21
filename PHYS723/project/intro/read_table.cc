#include <iostream>
#include <math.h>
#include <string>

using namespace std;

int main(int argc, char **argv){
  //string filename(argv[1]);
  //cout << " filename = " << filename << endl;

  const int nparts_max = 250;
  int idhep[nparts_max];
  int mother[nparts_max];
  int da1[nparts_max];
  int da2[nparts_max];
  float px[nparts_max];
  float py[nparts_max];
  float pz[nparts_max];
  float e[nparts_max];
  float x[nparts_max];
  float y[nparts_max];
  float z[nparts_max];
  float t[nparts_max];

  int ipart, i1, i2, i3;
  int nevts = 0;
  while(cin 
	>> ipart >> idhep[ipart] >> mother[ipart]
	>> da1[ipart] >> da2[ipart] 
	>> px[ipart] >> py[ipart] >> pz[ipart] >> e[ipart]
	>> x[ipart] >> y[ipart] >> z[ipart] >> t[ipart]
	>> i1 >> i2 >> i3){
    if(ipart == 0)nevts++;
  } // End read loop
  cout << " nevts = " << nevts << endl;
} // End int main()
