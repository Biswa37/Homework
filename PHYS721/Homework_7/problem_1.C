#include "TTree.h"
#include "TFile.h"
#include "TH1D.h"
#include "TCanvas.h"

double background(double *x, double *par) {
	return par[0] + par[1]*x[0] + par[2]*x[0]*x[0];
}
// Lorentzian Peak function
double lorentzianPeak(double *x, double *par) {
	//return (0.5*par[0]*par[1]/TMath::Pi()) / TMath::Max(1.e-10, (x[0]-par[2])*(x[0]-par[2])+ .25*par[1]*par[1]);
	return (0.5*par[0]*par[1]/TMath::Pi()) / TMath::Max(1.e-10, (x[0]-par[2])*(x[0]-par[2])+ .25*par[1]*par[1]);
}
double gaussian(double *x, double *par){
    return (par[0] * TMath::Exp(-(x[0] - par[1])*(x[0] - par[1]) / 2 * par[2] * par[2]));
}
// Sum of background and peak function
double fitFunction(double *x, double *par) {
	return background(x,par) + gaussian(x,&par[3]);
}

void problem_1(){
	//Declare variable for energy
	Float_t energy;

	//declare a histogram for the energy
	TH1D *energy_hist = new TH1D("Energy","Energy",50,100,160);
	//declare the canvas to display hist
	TCanvas *c1 = new TCanvas("c1","c1",10,10,900,900);
	TF1 *fitFcn = new TF1("fitFcn",fitFunction,90,180,6);

	//get the input file
	TFile *inputFile = new TFile("data.root");
	//get the correct tree from the input file
	//depending on the tree name change "data"
	TTree *data_tree = (TTree*)inputFile->Get("data");

	//Get the branch named E1 and put it into the varialbe energy
	data_tree->SetBranchAddress("E", &energy);

	//Get the number of events for the for loop
	int count = data_tree->GetEntries();

	for (int i = 0; i < count; i++) {	
		//get the current event i
		data_tree->GetEntry(i);
		//Fill the histogram
		energy_hist->Fill(energy);
	}
	//label the axis and draw the histogram after it's filled
	energy_hist->GetXaxis()->SetTitle("E (GeV)");
	energy_hist->Draw();
	// first try without starting values for the parameters
	// this defaults to 1 for each param.
	energy_hist->Fit("fitFcn");
	// this results in an ok fit for the polynomial function however
	// the non-linear part (Lorentzian) does not respond well
	// second try: set start values for some parameters
	fitFcn->SetParameter(4,1.66); // width
	fitFcn->SetParameter(5,126.5); // peak
	energy_hist->Fit("fitFcn","V+");

	// improve the picture:
	TF1 *backFcn = new TF1("backFcn",background,100,160,3);
	backFcn->SetLineColor(3);
	backFcn->SetLineStyle(3);
	TF1 *signalFcn = new TF1("signalFcn",gaussian,100,160,3);
	signalFcn->SetLineColor(4);
	Double_t par[6];

	// writes the fit results into the par array
	fitFcn->GetParameters(par);
	backFcn->SetParameters(par);
	backFcn->Draw("same");
	signalFcn->SetParameters(&par[4]);
	signalFcn->Draw("same"); 
	cout << fitFcn->Integral(0,200) - backFcn->Integral(0,200) << endl;

	gStyle->SetOptFit(1111);

}