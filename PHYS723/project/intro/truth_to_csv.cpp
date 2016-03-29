#include <iostream>
#include <math.h>
#include <string>

using namespace std;

// Run with:
//$ make truth_to_csv
//$ ./truth_to_csv < truth.inp > truth.csv
int main(int argc, char **argv){

    const int nparts_max = 250;
    int idhep[nparts_max], mother[nparts_max], da1[nparts_max], da2[nparts_max];
    float px[nparts_max], py[nparts_max], pz[nparts_max], e[nparts_max], x[nparts_max], y[nparts_max], z[nparts_max], t[nparts_max];

    int ipart, i1, i2, i3;
    int nevts = 0;
    int totevts = 0;

    cout << "part,event,ID,mother,daughter1,daughter2,Px,Py,Pz,E,X,Y,Z,T,i1,i2,i3" << endl;

    while(cin >> ipart >> idhep[ipart] >> mother[ipart] >> da1[ipart] >> da2[ipart] 
        >> px[ipart] >> py[ipart] >> pz[ipart] >> e[ipart] 
        >> x[ipart] >> y[ipart] >> z[ipart] >> t[ipart] 
        >> i1 >> i2 >> i3) {
        if(ipart == 0) nevts++;
        totevts++;
        cout << totevts << "," << ipart << "," << idhep[ipart] << "," << mother[ipart] << "," << da1[ipart] << "," << da2[ipart] 
            << "," << px[ipart] << "," << py[ipart] << "," << pz[ipart] << "," << e[ipart] 
            << "," << x[ipart] << "," << y[ipart] << "," << z[ipart] << "," << t[ipart] 
            << "," << i1 << "," << i2 << "," << i3 << endl;
    } 

} 
