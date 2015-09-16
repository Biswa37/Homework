#include <string>
#include <iostream>

#include "TLorentzRotation.h"
#include "TF1.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TFile.h"
#include "TStyle.h"

using namespace std;
typedef TLorentzVector T4V;

double BW(double *x, double *par){
	double dr = x[0] - par[1];
	double di = 0.5 * par[2];
	return par[0] / (dr*dr + di*di);
}

int main(int argc, char const *argv[]) {
	const int nmax = 1000;

	T4V *p1;
	T4V *p2;
	T4V *p12;

	double m12_min = 0.9;
	double m12_max = 1.1;

	TH1F *hist_m12 = new TH1F("hist_m12", "m12", 100,0.9,1.1);

	float E1, p1x, p1y, p1z, E2, p2x, p2y, p2z;

	TFile *inputFile = new TFile("data.root");
	TTree *data_tree = (TTree*)inputFile->Get("data");

	data_tree->SetBranchAddress("E1", &E1);
	data_tree->SetBranchAddress("Px1", &p1x);
	data_tree->SetBranchAddress("Py1", &p1y);
	data_tree->SetBranchAddress("Pz1", &p1z);
	data_tree->SetBranchAddress("E2", &E2);
	data_tree->SetBranchAddress("Px2", &p2x);
	data_tree->SetBranchAddress("Py2", &p2y);
	data_tree->SetBranchAddress("Pz2", &p2z);

	for (int i = 0; i < data_tree->GetEntries(); i++){
		data_tree->GetEntry(i);
		p1 = new T4V(p1x,p1y,p1z,E1);
		p2 = new T4V(p2x,p2y,p2z,E2);
		p12 = new T4V;
		*p12 = *p1 + *p2;
		hist_m12->Fill(p12->M());
	}
	gStyle->SetFitFormat("6.10g");
	gStyle->SetOptFit(1111);
	TCanvas c1("c1", "Masses", 200, 10, 700, 500);
	TF1 *bw = new TF1("BW", BW, m12_min, m12_max, 3);
	bw->SetParameter(0, 50.);
	bw->SetParameter(1, 1.019);
	bw->SetParameter(2, 0.00426);
	hist_m12->Fit(bw, "R");

	c1.SaveAs("bw_fit.pdf");

	return 0;
}