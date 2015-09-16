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

/* functions from python code
def myBW_2(Energy,Mass,Gamma_0):
    g = ((Mass**2.0 + Gamma_P(Energy,Gamma_0)**2.0)*Mass**2.0)**(1.0/2.0)
    k = (2.0 * 2.0**(1.0/2.0) * Mass * Gamma_P(Energy,Gamma_0) * g)/(np.pi * (Mass**(2.0)+g)**(1.0/2.0))
    return (k/((Energy**2.0-Mass**2.0)**2.0 + (Gamma_P(Energy,Gamma_0)*Mass)**2.0))

def Gamma_P(Energy,Gamma_0):
    m_k = 0.493677
    m_phi = 1.019461
    p = ((Energy**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    p0 = ((m_phi**2.0/4.0)-m_k**2.0)**(1.0/2.0)
    return Gamma_0*(p/p0)**3.0


double gamma_p(double E, double Gamma_0){
	double m_k = 0.493677;
	double m_phi = 1.019461;
	double p = sqrt(((E*E)/4.0)-(m_k*m_k));
	double p0 = sqrt(((m_phi*m_phi)/4.0)-(m_k*m_k));
	return Gamma_0*(p/p0)*(p/p0)*(p/p0);
}

double bw_gamma(Double_t E , double mass, double gamma_0){
	double bw = gamma_p(E,gamma_0)/((E-mass)*(E-mass) + gamma_p(E,gamma_0)*gamma_p(E,gamma_0)/4);
	return bw/(2*TMath::Pi());
	return bw/(2*TMath::Pi());
}
*/
int main(int argc, char const *argv[]) {
	TCanvas *c1 = new TCanvas("c1","#phi -> K^+ K^-",800,800);

	TF1 *myBW = new TF1("myBW","TMath::BreitWigner(x,[0],[1])", 0.9, 1.1);
	TF1 *BW_nonR = new TF1("BW_nonR","((([1]/(2.0*3.14159)))/((x-[0])**2.0 + ([1]/2.0)**2.0))", 0.9, 1.1);
	//TF1 *BW_gamma = new TF1("BW_gamma","([1]*TMath::Power((sqrt(((x*x)/4.0)-(0.493677*0.493677))/sqrt(((1.019461*1.019461)/4.0)-(0.493677*0.493677))),3))/((x-[0])*(x-[0]) + gamma_p(x,[1])*gamma_p(x,[1])/4/(2*TMath::Pi())", 0, 2);

	myBW->SetLineStyle(1);
	myBW->SetLineWidth(4);
	myBW->SetLineColor(kBlue);
	myBW->SetParameter(0,1.019461);
	myBW->SetParameter(1,0.00426);
	myBW->SetTitle("#phi #rightarrow k^{+} k^{-}");

	BW_nonR->SetLineStyle(2);
	BW_nonR->SetLineWidth(4);
	BW_nonR->SetLineColor(kRed);
	BW_nonR->SetParameter(0, 1.019461);
	BW_nonR->SetParameter(1, 0.00426);

	//BW_gamma->SetLineStyle(6);
	//BW_gamma->SetLineWidth(4);
	//BW_gamma->SetLineColor(kBlue);
	//BW_gamma->SetParameter(0, 1.019461);
	//BW_gamma->SetParameter(1, 0.00426);

	myBW->Draw();
	BW_nonR->Draw("same");
	//BW_gamma->Draw();
	c1->SetTitle("#phi -> k^+ k^-");
	c1->Print("phi_decay.pdf","pdf");

	return 0;
}