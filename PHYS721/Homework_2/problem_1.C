void problem_1(){
	//Declare variable for energy
	Float_t energy;

	//declare a histogram for the energy
	TH1D *energy_hist = new TH1D("Energy","Energy",50,0,2.5);
	//declare the canvas to display hist
	TCanvas *c1 = new TCanvas("c1","c1",10,10,900,500);

	//get the input file
	TFile *inputFile = new TFile("data.root");
	//get the correct tree from the input file
	//depending on the tree name change "data"
	TTree *data_tree = (TTree*)inputFile->Get("data");

	//Get the branch named E1 and put it into the varialbe energy
	data_tree->SetBranchAddress("E1", &energy);

	//Get the number of events for the for loop
	int count = data_tree->GetEntries();

	for (int i = 0; i < count; i++)
	{	
		//get the current event i
		data_tree->GetEntry(i);
		//Fill the histogram
		energy_hist->Fill(energy);
	}
	//label the axis and draw the histogram after it's filled
	energy_hist->GetXaxis()->SetTitle("E (GeV)");
	energy_hist->Draw();



}