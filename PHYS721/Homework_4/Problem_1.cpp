#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TLorentzVector.h"
#include "TStyle.h"
#include "TMath.h"
#include <iostream>
#include "TPaveStats.h"

using namespace std;

int main(int argc, char const *argv[]) {
	TFile *myFile = new TFile("output.root","RECREATE");

	Float_t E1,Px1,Py1,Pz1,E2,Px2,Py2,Pz2;
	Double_t mass = 0;

	TH1D *m12 = new TH1D("m12","m12",60,0.99,1.08);
	TF1 *myBW = new TF1("myBW","TMath::BreitWigner(x,[0],[1])", 1, 2);
	TF1 *BW_nonR = new TF1("BW_nonR","((([1]/(2.0*3.14159)))/((x-[0])**2.0 + ([1]/2.0)**2.0))", 1.01, 1.03);

	myBW->SetParName(0,"Mass_R");
	myBW->SetParName(1,"#Gamma_R");
	myBW->SetLineStyle(2);
	myBW->SetLineWidth(4);
	myBW->SetLineColor(kBlue);

	BW_nonR->SetParName(0,"Mass_nonR");
	BW_nonR->SetParName(1,"#Gamma_nonR");
	BW_nonR->SetLineStyle(4);
	BW_nonR->SetLineWidth(4);
	BW_nonR->SetLineColor(kRed);

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

	int counts = data_tree->GetEntries();

	for (int i = 0; i < counts; i++) {
		data_tree->GetEntry(i);
		vec1.SetPxPyPzE(Px1,Py1,Pz1,E1);
		vec2.SetPxPyPzE(Px2,Py2,Pz2,E2);
		mass = (vec1 + vec2).M();

		m12->Fill(mass);
	}
	m12->Draw();
	BW_nonR->SetParameters(0,1.019,0.043);
	BW_nonR->SetParLimits(0,1.01,1.03);
	m12->Fit("BW_nonR","+","sames",1,1.08);

	myBW->SetParameters(0,1.019,0.043);
	myBW->SetParLimits(0,1.01,1.03);
	m12->Fit("myBW","+","sames",0,2);

	m12->GetXaxis()->SetTitle("Mass (GeV)");
	gStyle->SetOptFit(0011);

	TPaveStats *st = ((TPaveStats*)(m12->GetListOfFunctions()->FindObject("stats")));
	st->SetFitFormat("1.8g");
	st->SetX1NDC(0.64); st->SetX2NDC(0.99);
	st->SetY1NDC(0.4); st->SetY2NDC(0.6);

	myFile->cd();
	myFile->Write();
	myFile->Close();

	return 0;
}