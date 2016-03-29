#include <iostream>
#include <math.h>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char **argv){
    //string filename(argv[1]);
    //cout << " filename = " << filename << endl;
    string base_name = "events/evnt_";
    string format = ".csv";
    ofstream myfile;
    myfile.open("");

    const int nparts_max = 250;
    int idhep[nparts_max], mother[nparts_max], da1[nparts_max], da2[nparts_max];
    float px[nparts_max], py[nparts_max], pz[nparts_max], e[nparts_max], x[nparts_max], y[nparts_max], z[nparts_max], t[nparts_max];

    int ipart, i1, i2, i3;
    int nevts = 0;
    int totevts = 0;

    while(cin >> ipart >> idhep[ipart] >> mother[ipart] >> da1[ipart] >> da2[ipart] 
        >> px[ipart] >> py[ipart] >> pz[ipart] >> e[ipart] 
        >> x[ipart] >> y[ipart] >> z[ipart] >> t[ipart] 
        >> i1 >> i2 >> i3) {
        if(ipart == 0) {
            myfile.close();
            nevts++;
            myfile.open(base_name + to_string(nevts) + format);
            myfile << "part,event,ID,mother,daughter1,daughter2,Px,Py,Pz,E,X,Y,Z,T,i1,i2,i3" << endl;
        }
        totevts++;
        myfile << totevts << "," << ipart << "," << idhep[ipart] << "," << mother[ipart] << "," << da1[ipart] << "," << da2[ipart] 
            << "," << px[ipart] << "," << py[ipart] << "," << pz[ipart] << "," << e[ipart] 
            << "," << x[ipart] << "," << y[ipart] << "," << z[ipart] << "," << t[ipart] 
            << "," << i1 << "," << i2 << "," << i3 << endl;
    }
    myfile.close();
    
}
