#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TLorentzVector.h"
#include "TStyle.h"
#include "TMath.h"
#include <iostream>

using namespace std;

int main(int argc, char const *argv[]) {
	TFile *myFile = new TFile("output.root","RECREATE");

	Float_t E1,Px1,Py1,Pz1,E2,Px2,Py2,Pz2;
	Double_t mass = 0;

	TH1D *m12 = new TH1D("m12","m12",60,0.99,1.08);
	TF1 *myBW = new TF1("myBW","TMath::BreitWigner(x,[0],[1])", 1.01, 1.03);

	myBW->SetParName(0,"Mass");
	myBW->SetParName(1,"#Gamma");

	TFile *inputFile = new TFile("data.root");
	TTree *data_tree = (TTree*)inputFile->Get("data");

	data_tree->SetBranchAddress("E1", &E1);
	data_tree->SetBranchAddress("Px1", &Px1);
	data_tree->SetBranchAddress("Py1", &Py1);
	data_tree->SetBranchAddress("Pz1", &Pz1);
	data_tree->SetBranchAddress("E2", &E2);
	data_tree->SetBranchAddress("Px2", &Px2);
	data_tree->SetBranchAddress("Py2", &Py2);
	data_tree->SetBranchAddress("Pz2", &Pz2);

	TLorentzVector vec1(0.0,0.0,0.0,0.0);
	TLorentzVector vec2(0.0,0.0,0.0,0.0);

	int count = data_tree->GetEntries();

	for (int i = 0; i < count; i++) {	
		data_tree->GetEntry(i);
		vec1.SetPxPyPzE(Px1,Py1,Pz1,E1);
		vec2.SetPxPyPzE(Px2,Py2,Pz2,E2);
		mass = (vec1 + vec2).M();

		m12->Fill(mass);
	}
	myBW->SetParameters(0,1.02,0.05);
	myBW->SetParLimits(0,1.01,1.03);
	m12->Fit("myBW","","",0,2);
	m12->GetXaxis()->SetTitle("Mass (GeV)");
	gStyle->SetOptFit(1111);

	myFile->cd();
	myFile->Write();
	myFile->Close();

	return 0;
}